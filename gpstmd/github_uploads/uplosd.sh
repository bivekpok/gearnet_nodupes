#!/bin/bash
#SBATCH --job-name="pdb_upload"
# Edit log paths for your cluster (Slurm does not expand env vars here).
#SBATCH --output="/work/hdd/bdja/bpokhrel/new_gearnet/foldseek_train/whole_test_cover/github_uploads/logs/old_param.out.%j.%N.out"
#SBATCH --error="/work/hdd/bdja/bpokhrel/new_gearnet/foldseek_train/whole_test_cover/github_uploads/logs/old_param.err.%j.%N.err"
#SBATCH --partition=gpuA40x4
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
#SBATCH -t 3:00:00

# Override before sbatch or set in paths.env (see PATHS.md)
: "${CONDA_SH:=/u/bpokhrel/miniconda3/etc/profile.d/conda.sh}"
: "${CONDA_ENV_UPLOAD:=env_general}"
: "${HF_UPLOAD_FOLDER:=/work/hdd/bdja/bpokhrel/gearnet_files/pdb_m15_ac}"
: "${HF_REPO_ID:=bivek77/protein-dataset}"

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${_SCRIPT_DIR}/paths.env" ]]; then
  # shellcheck source=/dev/null
  set -a && source "${_SCRIPT_DIR}/paths.env" && set +a
fi

module purge
export NUMEXPR_MAX_THREADS=8
# shellcheck source=/dev/null
source "${CONDA_SH}"
conda activate "${CONDA_ENV_UPLOAD}"

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_large_folder(
    folder_path='${HF_UPLOAD_FOLDER}',
    repo_id='${HF_REPO_ID}',
    repo_type='dataset'
)
print('PDBs uploaded!')
"
