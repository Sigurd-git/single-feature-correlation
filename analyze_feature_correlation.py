# Standard library imports
import os
import sys
from typing import Dict, List, Optional, Tuple, Union
import warnings
import dill as pickle

# Third-party imports
import numpy as np
import torch
import hdf5storage
from scipy.stats import pearsonr
from tqdm import tqdm
import pandas as pd
import glob

# Local imports
from _prepare import load_features
from _utils import load_config_with_defaults, parse, cached_load_response


def load_model_features(
    project_name: str,
    stimuli: List[str],
    feature_class: str,
    feature_variant: str,
    device: str = "cpu",
    shared_dir: str = "/scratch/snormanh_lab/shared",
) -> torch.Tensor:
    """
    Load features for a specific model and variant using configuration parameters.

    Args:
        project_name: Name of the project
        stimuli: List of stimulus names
        feature_class: Top level feature category (e.g., spectrotemporal, HuBERT)
        feature_variant: Specific version of the feature (e.g., spectempmod_modulus)
        device: Device to use for computations ('cpu' or 'cuda')
        shared_dir: Base directory for shared data

    Returns:
        torch.Tensor: Features tensor reshaped to (time, features) format on GPU
    """
    # Load configuration
    conf_dir = "/scratch/snormanh_lab/shared/Sigurd/encodingmodel/code/conf"
    data = load_config_with_defaults(conf_dir, project_name)
    args_dict = parse(data, print_args=False)

    feature_path = os.path.join(
        shared_dir,
        "projects",
        project_name,
        "analysis/features",
        feature_class,
        feature_variant,
    )
    exclude_before = args_dict.exclude_before
    n_t = args_dict.n_t

    # Load and concatenate features for all stimuli
    features = []
    for stim in stimuli:
        stim_path = os.path.join(feature_path, f"{stim}.mat")
        stim_features = hdf5storage.loadmat(
            stim_path, simplify_cells=True, squeeze_me=True
        )["features"]
        features.append(stim_features[exclude_before : n_t + exclude_before])
    features = np.concatenate(features, axis=0)  # Concatenate along time dimension

    # Reshape features based on feature class
    if feature_class == "spectrotemporal":
        # Spectrotemporal features have shape (time, freq, freq_mod, time_mod, complex)
        n_time = features.shape[0]
        features = features.reshape(n_time, -1)

    # Convert to torch tensor and move to specified device
    features = torch.from_numpy(features).float().to(device)
    return features


def compute_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Pearson correlation between two GPU tensors."""
    x_mean = x.mean()
    y_mean = y.mean()
    x_std = x.std(unbiased=False)
    y_std = y.std(unbiased=False)

    corr = ((x - x_mean) * (y - y_mean)).mean() / (x_std * y_std)
    return corr.item()


def analyze_feature_correlations(
    project_name: str,
    subject: str,
    feature_class: str,
    feature_variant: str,
    device: str = "cpu",
    category_metric: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Analyze correlations between electrode responses and features for a specific model variant.

    Args:
        project_name: Name of the project
        subject: Subject ID
        feature_class: Feature class to analyze
        feature_variant: Feature variant to analyze
        device: Device to use for computations ('cpu' or 'cuda')
        category_metric: Optional metric for selective/non-selective regions

    Returns:
        Dict containing correlation results for the model variant
    """
    # Load configuration
    conf_dir = "/scratch/snormanh_lab/shared/Sigurd/encodingmodel/code/conf"
    data = load_config_with_defaults(conf_dir, project_name)
    args_dict = parse(data, print_args=False)

    # Load response data with cross-validation groups
    y, groups, metas, train_groups, masks = cached_load_response(
        args_dict.project_name,
        subject,
        args_dict.train_stimuli,
        sr=args_dict.sr,
        shared_dir=args_dict.shared_dir,
        lag_y_basis=args_dict.lag_y_basis,
        exclude_before=args_dict.exclude_before,
        n_folds=args_dict.n_folds,
        group_method=args_dict.group_method,
        window=args_dict.window_y,
        lag_step=args_dict.lag_step,
        average_reps=args_dict.average_reps,
        demean=args_dict.demean_y,
        standardize=args_dict.standardize_y,
        train_group_method=args_dict.train_group_method,
        include_after=args_dict.include_after,
        data_type=args_dict.data_type,
        n_t=300,
    )

    # Convert response data to specified device
    y = torch.from_numpy(y).float().to(device)

    # Initialize results dictionary
    results = {f"{feature_class}-{feature_variant}": []}

    print(f"Processing {feature_class}-{feature_variant}")

    # Load features for current variant
    try:
        features = load_model_features(
            args_dict.project_name,
            args_dict.train_stimuli,
            feature_class,
            feature_variant,
            device=device,
        )
    except Exception as e:
        warnings.warn(
            f"Failed to load features for {feature_class}-{feature_variant}: {str(e)}"
        )
        return results

    # Get unique folds for cross-validation
    unique_folds = np.unique(groups)
    groups = torch.from_numpy(groups).to(device)

    # Analyze each electrode
    for electrode_idx in tqdm(
        range(y.shape[1]), desc=f"Processing electrodes for {feature_variant}"
    ):
        electrode_response = y[:, electrode_idx]
        best_feature_correlations = []

        # Cross-validation
        for fold in unique_folds:
            # Split data
            train_mask = groups != fold
            test_mask = groups == fold

            # Get training and testing data
            train_response = electrode_response[train_mask]
            test_response = electrode_response[test_mask]
            train_features = features[train_mask]
            test_features = features[test_mask]

            # Initialize correlation tensor on specified device
            train_correlations = torch.zeros(train_features.shape[1], device=device)
            for i in range(train_features.shape[1]):
                train_correlations[i] = compute_correlation(
                    train_response, train_features[:, i]
                )

            # Get best feature index
            best_feature_idx = torch.argmax(torch.abs(train_correlations))

            # Calculate correlation on test data
            test_correlation = compute_correlation(
                test_response, test_features[:, best_feature_idx]
            )
            best_feature_correlations.append(test_correlation)

        # Average correlations across folds
        mean_correlation = np.nanmedian(best_feature_correlations)
        results[f"{feature_class}-{feature_variant}"].append(mean_correlation)

    # Save results
    output_dir = os.path.join(
        args_dict.shared_dir,
        "Sigurd/encodingmodel/analysis",
        project_name,
        "1-feature_correlations",
    )
    os.makedirs(output_dir, exist_ok=True)

    # Prepare results dictionary with metadata
    results_dict = {
        "results": results,
        "subject": subject,
        "feature_class": feature_class,
        "feature_variant": feature_variant,
        "n_folds": args_dict.n_folds,
        "group_method": args_dict.group_method,
    }

    # Save using pickle
    output_file = os.path.join(
        output_dir,
        f"{subject}_{feature_class}_{feature_variant}_feature_correlations.pkl",
    )
    with open(output_file, "wb") as f:
        pickle.dump(results_dict, f)

    print(f"Results saved to {output_file}")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python analyze_feature_correlation.py project_name subject feature_class feature_variant"
        )
        sys.exit(1)

    project_name = sys.argv[1]
    subject = sys.argv[2]
    feature_class = sys.argv[3]
    feature_variant = sys.argv[4]

    # project_name = "naturalsound-iEEG-153-allsubj"
    # subject = "AMC045"
    # feature_class = "spectrotemporal"
    # feature_variant = "spectempmod_modulus"

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = analyze_feature_correlations(
        project_name=project_name,
        subject=subject,
        feature_class=feature_class,
        feature_variant=feature_variant,
        device=device,
    )
