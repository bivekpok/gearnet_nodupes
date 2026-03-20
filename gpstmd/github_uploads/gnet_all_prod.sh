#!/bin/bash
#SBATCH --job-name="gnet_production_cv"
# NOTE: Slurm does not expand shell variables here — edit these paths to match your cluster / WORK_COVER.
#SBATCH --output="/work/hdd/bdja/bpokhrel/new_gearnet/foldseek_train/whole_test_cover/logs/production_v2.%j.%N.out"
#SBATCH --error="/work/hdd/bdja/bpokhrel/new_gearnet/foldseek_train/whole_test_cover/logs/production_v2.%j.%N.err"
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdja-delta-gpu
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mail-user='bivekpok@udel.edu'
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH -t 8:00:00

# --- Path overrides (see PATHS.md and paths.env.example) ---
# 1) Copy paths.env.example -> paths.env (gitignored) next to this script, or
# 2) export WORK_COVER, GEARNET_PDB_ROOT, etc. before sbatch.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${_SCRIPT_DIR}/paths.env" ]]; then
  # shellcheck source=/dev/null
  set -a && source "${_SCRIPT_DIR}/paths.env" && set +a
fi

: "${WORK_COVER:=/work/hdd/bdja/bpokhrel/new_gearnet/foldseek_train/whole_test_cover}"
: "${GEARNET_PDB_ROOT:=/work/hdd/bdja/bpokhrel/gearnet_files}"
: "${CONDA_SH:=/u/bpokhrel/miniconda3/etc/profile.d/conda.sh}"
: "${CONDA_ENV_GEARNET:=/u/bpokhrel/gearnet}"

: "${BASE_SPLIT_DIR:=${WORK_COVER}/production_splitsv2}"
: "${PDB_DIR:=${GEARNET_PDB_ROOT}/pdb_m15_ac}"
: "${SOLUBLE_DIR:=${GEARNET_PDB_ROOT}/watersoluble_proteins_ac}"
: "${OUTPUT_DIR:=${WORK_COVER}/production_models_v2}"

# 1. Clear loaded modules
module purge

# 2. Fix the NumExpr warning
export NUMEXPR_MAX_THREADS=8

# 3. Prevent W&B from filling up your limited home directory quota
export WANDB_DIR=/tmp
export WANDB_CACHE_DIR=/tmp

# 4. Activate your environment
# shellcheck source=/dev/null
source "${CONDA_SH}"
conda activate "${CONDA_ENV_GEARNET}"

# 5. Navigate to the working directory (train_splits.py is run from here)
cd "${WORK_COVER}" || exit 1

# Create directories for models and individual fold logs
mkdir -p "${OUTPUT_DIR}/logs"

echo "Starting Nested Cross-Validation Production Run..."
echo "WORK_COVER=${WORK_COVER}"
echo "BASE_SPLIT_DIR=${BASE_SPLIT_DIR}"
echo "PDB_DIR=${PDB_DIR}"
echo "SOLUBLE_DIR=${SOLUBLE_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

# --- NESTED LOOP: OUTER FOLDS -> INNER FOLDS ---
for outer_path in ${BASE_SPLIT_DIR}/Outer_Fold_*; do
    if [ -d "$outer_path" ]; then
        outer_name=$(basename "$outer_path")

        for inner_path in ${outer_path}/Inner_Fold_*; do
            if [ -d "$inner_path" ]; then
                inner_name=$(basename "$inner_path")

                echo "=========================================================="
                echo "Training: $outer_name -> $inner_name"
                echo "=========================================================="

                FOLD_LOG="${OUTPUT_DIR}/logs/${outer_name}_${inner_name}.log"

                python3 train_splits.py \
                  --pdb_folder "$PDB_DIR" \
                  --soluble_folder_ac "$SOLUBLE_DIR" \
                  --split_dir "$inner_path" \
                  --output_dir "$OUTPUT_DIR" \
                  --learning_rate 0.00000394644981433921 \
                  --weight_decay 0.00008113944975079617 \
                  --mlp_dropout 0.2903210512935248 \
                  --readout "mean" \
                  --num_gearnet_layers 5 \
                  --batch_size 32 \
                  --hidden_dim 512 \
                  --concat_hidden "False" \
                  --knn_k 25 \
                  --spatial_radius 12.0 \
                  --num_epochs 2500 \
                  --seed 56 \
                  --activation "relu" > "$FOLD_LOG" 2>&1

                echo "Finished $outer_name -> $inner_name. (Saved to $FOLD_LOG)"
                echo "----------------------------------------------------------"
            fi
        done
    fi
done

echo "All nested folds completed successfully!"
