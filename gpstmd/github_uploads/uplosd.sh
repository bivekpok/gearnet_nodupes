#!/bin/bash
#SBATCH --job-name="pdb_upload"
#SBATCH --output="/work/hdd/bdja/bpokhrel/new_gearnet/foldseek_train/whole_test_cover/github_uploads/logs/old_param.out.%j.%N.out"
#SBATCH --error="/work/hdd/bdja/bpokhrel/new_gearnet/foldseek_train/whole_test_cover/github_uploads/logs/old_param.err.%j.%N.err"
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdja-delta-gpu    
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1   # , 2
#SBATCH --cpus-per-task=16  
#SBATCH --gpus-per-node=1     #, 2    # always equal to "ntasks-per-node"
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest  
#SBATCH --mail-user='bivekpok@udel.edu'
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH -t 3:00:00


module purge

# 2. Fix the NumExpr warning you were getting
export NUMEXPR_MAX_THREADS=8
source /u/bpokhrel/miniconda3/etc/profile.d/conda.sh
conda activate env_general

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_large_folder(
    folder_path='/work/hdd/bdja/bpokhrel/gearnet_files/pdb_m15_ac',
    repo_id='bivek77/protein-dataset',
    repo_type='dataset'
)
print('PDBs uploaded!')
"
