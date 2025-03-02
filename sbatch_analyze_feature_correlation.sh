#!/bin/bash
#SBATCH -p preempt -x bhd0038,bhd0039,bhg0044,bhg0046,bhg0047,bhg0048,bhg0059
#SBATCH -t 16:00:00
#SBATCH -c 8
#SBATCH -a 0-164
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH -o /scratch/snormanh_lab/shared/Sigurd/encodingmodel/analysis/naturalsound-iEEG-153-allsubj/logs/feature_correlation/%a.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=guoyang_liao@urmc.rochester.edu

#----------------------------------------------------------------------------
# Project settings
project_name="naturalsound-iEEG-153-allsubj"
feature_class="spectrotemporal"
variants=(
    "spectempmod_modulus"
    "spectempmod_real"
    "spectempmod_power03"
    "spectempmod_signpower03"
    "spectempmod_rect"
)
subjects=(
    "AMC045"
    "AMC047"
    "AMC056"
    "AMC062"
    "AMC071"
    "AMC078"
    "AMC079"
    "AMC081"
    "AMC082"
    "AMC083"
    "AMC085"
    "AMC086"
    "AMC087"
    "AMC097"
    "AMC101"
    "Einstein01"
    "HBRL720"
    "HBRL741"
    "UR4"
    "UR6"
    "UR7"
    "UR8"
    "UR11"
    "UR12"
    "UR14"
    "UR15"
    "UR16"
    "UR17"
    "UR18"
    "UR19"
    "UR20"
    "UR21"
    "UR22"
)

#----------------------------------------------------------------------------
# Get the variant and subject based on array task ID
variant_idx=$((SLURM_ARRAY_TASK_ID / ${#subjects[@]}))
subject_idx=$((SLURM_ARRAY_TASK_ID % ${#subjects[@]}))
variant=${variants[$variant_idx]}
subject=${subjects[$subject_idx]}

#----------------------------------------------------------------------------
# Activate environment and run analysis
source /scratch/snormanh_lab/shared/Sigurd/encodingmodel/code/shell_code_shared/activate_env.sh

echo "Processing subject: $subject, variant: $variant"
python -u /scratch/snormanh_lab/shared/Sigurd/encodingmodel/analyze_feature_correlation.py \
    "$project_name" \
    "$subject" \
    "$feature_class" \
    "$variant" 