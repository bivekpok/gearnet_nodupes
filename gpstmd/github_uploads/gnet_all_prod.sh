#!/bin/bash
# =============================================================================
# Nested CV production training — paths are configurable (universal layout).
#
# 1) Put this script in your "whole_test_cover" (or any) project directory.
# 2) Optional: copy paths.env.example -> paths.env next to this script and set
#    GEARNET_PDB_ROOT, CONDA_SH, CONDA_ENV_GEARNET, etc.
# 3) Submit FROM the directory used for SLURM relative log paths (usually whole_test_cover):
#      mkdir -p logs && sbatch gnet_all_prod.sh
#    If this file lives under github_uploads/, from parent dir:
#      mkdir -p logs && sbatch github_uploads/gnet_all_prod.sh
#    Or override SLURM logs (Slurm does not expand $vars in #SBATCH):
#      sbatch -o /your/logs/out.%j -e /your/logs/err.%j gnet_all_prod.sh
#    Git clone only: set WORK_COVER in paths.env (parent of github_uploads may not hold splits).
# =============================================================================
#SBATCH --job-name="gnet_production_cv"
# Relative to your submission cwd — use "mkdir -p logs" before sbatch, or pass -o/-e.
#SBATCH --output=logs/production_v2.%j.%N.out
#SBATCH --error=logs/production_v2.%j.%N.err
# --- Edit for your cluster (not portable) ---
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

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${_SCRIPT_DIR}/paths.env" ]]; then
  # shellcheck source=/dev/null
  set -a && source "${_SCRIPT_DIR}/paths.env" && set +a
fi

# WORK_COVER = dir containing production_splitsv2/ (runtime cd target).
if [[ -z "${WORK_COVER}" ]]; then
  if [[ "$(basename "${_SCRIPT_DIR}")" == "github_uploads" ]]; then
    _PARENT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
    if [[ -d "${_PARENT}/production_splitsv2" ]]; then
      WORK_COVER="${_PARENT}"
    else
      echo "ERROR: ../production_splitsv2 not found from github_uploads/. Set WORK_COVER in paths.env." >&2
      exit 1
    fi
  else
    WORK_COVER="${_SCRIPT_DIR}"
  fi
fi
# Override in paths.env if your PDBs live elsewhere (example Delta path kept as fallback)
: "${GEARNET_PDB_ROOT:=/work/hdd/bdja/bpokhrel/gearnet_files}"
: "${CONDA_SH:=${HOME}/miniconda3/etc/profile.d/conda.sh}"
: "${CONDA_ENV_GEARNET:=gearnet}"

: "${BASE_SPLIT_DIR:=${WORK_COVER}/production_splitsv2}"
: "${PDB_DIR:=${GEARNET_PDB_ROOT}/pdb_m15_ac}"
: "${SOLUBLE_DIR:=${GEARNET_PDB_ROOT}/watersoluble_proteins_ac}"
: "${OUTPUT_DIR:=${WORK_COVER}/production_models_v2}"

module purge

export NUMEXPR_MAX_THREADS=8
export WANDB_DIR=/tmp
export WANDB_CACHE_DIR=/tmp

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "ERROR: CONDA_SH not found: ${CONDA_SH} — set CONDA_SH in paths.env or export it." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_SH}"
conda activate "${CONDA_ENV_GEARNET}"

cd "${WORK_COVER}" || exit 1

mkdir -p "${OUTPUT_DIR}/logs"

echo "Starting Nested Cross-Validation Production Run..."
echo "WORK_COVER=${WORK_COVER}"
echo "BASE_SPLIT_DIR=${BASE_SPLIT_DIR}"
echo "PDB_DIR=${PDB_DIR}"
echo "SOLUBLE_DIR=${SOLUBLE_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

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
