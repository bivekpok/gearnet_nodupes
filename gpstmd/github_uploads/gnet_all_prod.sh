#!/bin/bash
#SBATCH --job-name="gnet_production_cv"
#SBATCH --output="production_v2.%j.%N.out"
#SBATCH --error="production_v2.%j.%N.err"
#SBATCH --partition=gpuA100x4
#SBATCH --account=YOUR_ALLOCATION_HERE       # <--- USER: Update with your HPC allocation
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16  
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest  
#SBATCH --mail-user='YOUR_EMAIL@DOMAIN.COM'  # <--- USER: Update with your email
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH -t 8:00:00

# 1. Clear loaded modules
module purge

# 2. Fix the NumExpr warning
export NUMEXPR_MAX_THREADS=8

# 3. Prevent W&B from filling up limited home directory quota
export WANDB_DIR=/tmp
export WANDB_CACHE_DIR=/tmp

# 4. Activate environment 
# (Assumes the user has Conda initialized. If submitting via SLURM, 
# it's best to 'conda activate gearnet' before running 'sbatch')
eval "$(conda shell.bash hook)"
conda activate gearnet

# 5. Automatically navigate to the directory where the script was submitted
cd "$SLURM_SUBMIT_DIR"

# Ensure the log directory exists for SLURM output
mkdir -p logs

# --- PATH VARIABLES ---
# Base directories (Users should adjust DATA_DIR to where they downloaded the datasets)
DATA_DIR="../gearnet_files" 
BASE_SPLIT_DIR="./production_splitsv2"
OUTPUT_DIR="./production_models_v2"

PDB_DIR="${DATA_DIR}/pdb_m15_ac"
SOLUBLE_DIR="${DATA_DIR}/watersoluble_proteins_ac"

# Create directories for models and individual fold logs
mkdir -p "${OUTPUT_DIR}/logs"

echo "Starting Nested Cross-Validation Production Run..."

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
                
                # Log file specific to this fold
                FOLD_LOG="${OUTPUT_DIR}/logs/${outer_name}_${inner_name}.log"
                
                # Run Python script with LOCKED Sweep Hyperparameters
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
