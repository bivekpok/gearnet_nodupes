#!/bin/bash

# --- CONFIGURATION ---
# IMPORTANT: Update these paths to match your local data and model locations.
# These paths are relative to where you run this script (usually the project root).

# 1. Path to your PDB folder containing membrane proteins
PDB_FOLDER="./data/pdb_m15_ac"

# 2. Path to your soluble proteins folder
SOLUBLE_FOLDER="./data/watersoluble_proteins_ac"

# 3. Path to the CSV metadata file
CSV_PATH="./data/final_pdb_csv2.csv"

# 4. Directory where evaluation metrics/logs will be stored
OUTPUT_DIR="./results/github_output"

# 5. Directory containing the pre-trained model weights (.pth files)
MODEL_PATH_FOLDER="./results/pretrained_models"

# 6. Model version to load: "best" or "last"
BEST_OR_LAST="best"

# ---------------------
# --- ENVIRONMENT SETUP ---

VENV_PATH="./venv/bin/activate"

# Check if the virtual environment exists and activate it
if [ -f "$VENV_PATH" ]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH"
else
    echo "Warning: Virtual environment not found at $VENV_PATH. Ensure dependencies are installed globally or update VENV_PATH."
fi

# ---------------------

echo "Starting evaluation..."
echo "Model version to load: $BEST_OR_LAST"
echo "Output Directory: $OUTPUT_DIR"

# Execute the Python script in evaluation mode (by omitting the --training flag)
# Assumes protein_classification_reproducible.py is in the current directory or $PATH
python protein_classification_reproducible.py \
    --pdb_folder "$PDB_FOLDER" \
    --soluble_folder_ac "$SOLUBLE_FOLDER" \
    --csv_path "$CSV_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_path_folder "$MODEL_PATH_FOLDER" \
    --best_or_last "$BEST_OR_LAST"

# Optional: Deactivate the environment after the script finishes
if [ -f "$VENV_PATH" ]; then
    deactivate
    echo "Deactivated virtual environment."
fi

echo "Evaluation completed!"
