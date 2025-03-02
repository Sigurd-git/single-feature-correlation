import os
import glob
import pickle
import pandas as pd
from typing import Optional
from tqdm import tqdm


def integrate_feature_correlation_results(
    project_name: str,
    output_path: Optional[str] = None,
    shared_dir: str = "/scratch/snormanh_lab/shared",
) -> pd.DataFrame:
    """
    Integrate all feature correlation results for a project into a single DataFrame.

    Args:
        project_name: Name of the project
        output_path: Optional path to save the integrated DataFrame as CSV
        shared_dir: Base directory for shared data

    Returns:
        pd.DataFrame: Integrated results with metadata
    """
    # Define the directory where results are stored
    results_dir = os.path.join(
        shared_dir,
        "Sigurd/encodingmodel/analysis",
        project_name,
        "1-feature_correlations",
    )

    # Find all result files
    result_files = glob.glob(os.path.join(results_dir, "*_feature_correlations.pkl"))

    if not result_files:
        print(f"No result files found in {results_dir}")
        return pd.DataFrame()

    # Initialize lists to store data
    all_data = []

    # Process each file
    for file_path in tqdm(result_files, desc="Processing result files"):
        try:
            # Extract metadata from filename
            filename = os.path.basename(file_path)
            parts = filename.replace("_feature_correlations.pkl", "").split("_")

            # Load results
            with open(file_path, "rb") as f:
                result_dict = pickle.load(f)

            subject = result_dict["subject"]
            feature_class = result_dict["feature_class"]
            feature_variant = result_dict["feature_variant"]

            # Get the key for results
            result_key = f"{feature_class}-{feature_variant}"

            if result_key in result_dict["results"]:
                # Get correlation values
                correlations = result_dict["results"][result_key]

                # Create a row for each electrode
                for electrode_idx, correlation in enumerate(correlations):
                    all_data.append(
                        {
                            "subject": subject,
                            "electrode_idx": electrode_idx,
                            "feature_class": feature_class,
                            "feature_variant": feature_variant,
                            "correlation": correlation,
                            "abs_correlation": abs(correlation),
                        }
                    )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Save if output path is specified
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Integrated results saved to {output_path}")

    return df


project_name = "naturalsound-iEEG-153-allsubj"
output_path = "/scratch/snormanh_lab/shared/Sigurd/encodingmodel/analysis/naturalsound-iEEG-153-allsubj/1-feature_correlations/correlations.csv"
integrated_df = integrate_feature_correlation_results(
    project_name=project_name, output_path=output_path
)


def load_electrode_selectivity(
    correlation_df: pd.DataFrame,
    project_name: str,
    shared_dir: str = "/scratch/snormanh_lab/shared",
    data_type: str = "broadband_gamma",
    category: str = "music_or_speech_or_song",
) -> pd.DataFrame:
    """
    Load electrode selectivity information and add it to the correlation DataFrame.

    Args:
        correlation_df: DataFrame containing correlation results
        project_name: Name of the project
        shared_dir: Path to shared directory
        data_type: Type of data to load selectivity for
        category: Category type to use for selectivity analysis

    Returns:
        pd.DataFrame: Original DataFrame with added selectivity columns
    """
    # Import the parse_selectivity_file function
    from _utils import parse_selectivity_file

    # Create a copy of the DataFrame to avoid modifying the original
    df = correlation_df.copy()

    # Initialize selectivity columns
    df["selectivity"] = 0.0
    df["selectivity_type"] = "unknown"

    # Get unique subjects
    subjects = df["subject"].unique().tolist()

    # Load selectivity data for each subject
    for subject in subjects:
        subject_mask = df["subject"] == subject

        if not subject_mask.any():
            continue

        try:
            # Call parse_selectivity_file for this subject with music_or_speech_or_song category
            non_selective_bools, selective_bools, selectivities = (
                parse_selectivity_file(
                    subjects=[subject],
                    project_name=project_name,
                    shared_dir=shared_dir,
                    data_type=data_type,
                    category=category,
                )
            )

            # These should be arrays for the single subject
            for electrode_idx in range(len(selective_bools)):
                electrode_mask = df["electrode_idx"] == electrode_idx
                combined_mask = subject_mask & electrode_mask

                if combined_mask.any():
                    # Determine selectivity type
                    is_selective = selective_bools[electrode_idx]
                    is_non_selective = non_selective_bools[electrode_idx]

                    if is_selective:
                        selectivity_type = "selective"
                    elif is_non_selective:
                        selectivity_type = "non-selective"
                    else:
                        selectivity_type = "in-between"

                    selectivity_value = selectivities[electrode_idx]

                    df.loc[combined_mask, "selectivity"] = selectivity_value
                    df.loc[combined_mask, "selectivity_type"] = selectivity_type

        except Exception as e:
            print(f"Error loading selectivity data for subject {subject}: {e}")

    return df


def compare_feature_models_with_barplot(
    correlation_df: pd.DataFrame,
    groupby_cols: list = ["feature_class", "feature_variant"],
    metric: str = "correlation",
    output_dir: Optional[str] = None,
    figsize: tuple = (14, 6),
    palette: str = "viridis",
    title: str = "Feature Model Comparison",
    facet_by_selectivity: bool = False,
    project_name: Optional[str] = None,
    shared_dir: str = "/scratch/snormanh_lab/shared",
) -> None:
    """
    Create bar plots to compare different feature models based on correlation results.

    Args:
        correlation_df: DataFrame containing integrated correlation results
        groupby_cols: Columns to group by for comparison (default: feature_class and feature_variant)
        metric: Metric to use for comparison (default: correlation)
        output_dir: Directory to save the output plots
        figsize: Figure size for the plot
        palette: Color palette to use
        title: Plot title
        facet_by_selectivity: Whether to facet by electrode selectivity
        project_name: Name of the project (required if facet_by_selectivity is True)
        shared_dir: Path to shared directory
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Check if DataFrame is empty
    if correlation_df.empty:
        print("Empty DataFrame provided, cannot create plots")
        return

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Make a copy of the DataFrame to avoid modifying the original
    df = correlation_df.copy()

    # Add selectivity information if requested
    if facet_by_selectivity and project_name:
        print("Loading electrode selectivity data...")
        df = load_electrode_selectivity(
            correlation_df=df, project_name=project_name, shared_dir=shared_dir
        )

    # Group the data for comparison
    if facet_by_selectivity and "selectivity_type" in df.columns:
        # Group by selectivity as well
        grouped_data = (
            df.groupby(groupby_cols + ["selectivity_type"])[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
    else:
        grouped_data = (
            df.groupby(groupby_cols)[metric].agg(["mean", "std", "count"]).reset_index()
        )

    # Sort by performance (mean)
    grouped_data = grouped_data.sort_values("mean", ascending=False)

    # Create a new column combining non-feature_class columns
    if len(groupby_cols) > 1 and "feature_class" in groupby_cols:
        grouped_data["model_variant"] = grouped_data.apply(
            lambda row: "_".join(
                [str(row[col]) for col in groupby_cols if col != "feature_class"]
            ),
            axis=1,
        )

    # Create the plot
    plt.figure(figsize=figsize)

    if facet_by_selectivity and "selectivity_type" in grouped_data.columns:
        # Create a FacetGrid to facet by selectivity_type
        g = sns.FacetGrid(
            grouped_data,
            col="selectivity_type",
            height=figsize[1],
            aspect=figsize[0]
            / figsize[1]
            / len(grouped_data["selectivity_type"].unique()),
            sharey=True,
        )

        # Map barplot onto each facet
        g.map_dataframe(
            sns.barplot,
            x="model_variant",
            y="mean",
            hue="feature_class",
            palette=palette,
        )

        # Customize the facet grid
        g.set_axis_labels("Model Variant", f"Mean {metric.replace('_', ' ').title()}")
        g.set_titles(col_template="{col_name}")
        g.add_legend(title="Feature Class")

        # Rotate x-tick labels
        for ax in g.axes.flat:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        # Set the figure title
        plt.subplots_adjust(top=0.85)
        g.fig.suptitle(title)

    else:
        # Standard barplot without facetting
        sns.barplot(
            x="model_variant",
            y="mean",
            hue="feature_class",
            data=grouped_data,
            palette=palette,
        )
        plt.xlabel("Model Variant")
        plt.ylabel(f"Mean {metric.replace('_', ' ').title()}")
        plt.title(title)
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Feature Class")

    plt.tight_layout()

    # Save plot if output directory is specified
    if output_dir:
        plot_filename = f"{metric}_model_comparison{'_by_selectivity_facet' if facet_by_selectivity else ''}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {plot_path}")

    plt.show()


# Example usage - update the existing call to use facetting by selectivity
output_dir = os.path.join(os.path.dirname(output_path), "plots")
compare_feature_models_with_barplot(
    correlation_df=integrated_df,
    output_dir=output_dir,
    title="Natural Sound iEEG Feature Model Comparison",
    facet_by_selectivity=True,  # Enable facetting by electrode selectivity
    project_name=project_name,  # Provide project name for loading selectivity data
)
