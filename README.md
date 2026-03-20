# Training: nested CV & production runs

This repository’s **default landing page** describes how to run nested cross-validation with
`train_splits.py` and the SLURM driver `gnet_all_prod.sh`.

**Project overview** (dataset, architecture, `protein_classification_reproducible.py`, citations):
see **[`README_GPSforTMDs.md`](README_GPSforTMDs.md)** (formerly the main `README.md`).

**Paths on your machine:** the repo used to hardcode Delta paths (`/work/hdd/...`, `/u/bpokhrel/...`).
See **[`gpstmd/github_uploads/PATHS.md`](gpstmd/github_uploads/PATHS.md)** and copy
[`paths.env.example`](gpstmd/github_uploads/paths.env.example) → `paths.env` (gitignored).

---

## Layout

| Path | Purpose |
|------|---------|
| `gpstmd/github_uploads/train_splits.py` | Train **one inner fold** from manifest CSVs under `--split_dir` |
| `gpstmd/github_uploads/gearnet_modules.py` | Imported by `train_splits.py` (run from `github_uploads/`) |
| `gpstmd/github_uploads/gnet_all_prod.sh` | SLURM batch: loop all `Outer_Fold_*/Inner_Fold_*` |
| `gpstmd/filter_foldseek_redundant_by_membrane.py` | Optional: Foldseek + membrane-label dedup before splits |

After clone, training scripts live under **`gpstmd/github_uploads/`** (run `train_splits.py` from there).

---

## 1. Clone & environment

```bash
git clone https://github.com/bivekpok/gearnet_nodupes.git
cd gearnet_nodupes
conda env create -f environment.yml   # optional if you use a pre-built env
conda activate gearnet
```

---

## 2. Configure paths (recommended)

```bash
cd gearnet_nodupes/gpstmd/github_uploads
cp paths.env.example paths.env
# Edit paths.env: WORK_COVER, GEARNET_PDB_ROOT, CONDA_SH, CONDA_ENV_GEARNET
```

For `gnet_all_prod.sh`, also edit the **`#SBATCH --output` / `--error`** lines so log paths exist on your cluster (Slurm does not expand `$VAR` there).

---

## 3. Environment variables (interactive / same as batch job)

```bash
module purge
export NUMEXPR_MAX_THREADS=8
export WANDB_DIR=/tmp
export WANDB_CACHE_DIR=/tmp
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV_GEARNET:-gearnet}"
```

---

## 4. Run one inner fold (`train_splits.py`)

`train_splits.py` must run from the folder that contains **`gearnet_modules.py`** (same directory as the script).

```bash
cd gpstmd/github_uploads

# Set these to your layout (see PATHS.md)
export GEARNET_PDB_ROOT="/path/to/gearnet_files"
export WORK_COVER="/path/to/whole_test_cover"

python3 train_splits.py \
  --pdb_folder "${GEARNET_PDB_ROOT}/pdb_m15_ac" \
  --soluble_folder_ac "${GEARNET_PDB_ROOT}/watersoluble_proteins_ac" \
  --split_dir "${WORK_COVER}/production_splitsv2/Outer_Fold_1/Inner_Fold_1" \
  --output_dir "${WORK_COVER}/production_models_v2" \
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
  --activation "relu"
```

`gnet_all_prod.sh` uses the same directories via `WORK_COVER`, `GEARNET_PDB_ROOT`, etc. (defaults match the original Delta layout; override with `paths.env` or exports).

### Generate splits first (`datasplitterv2.py`)

```bash
python3 datasplitterv2.py \
  --csv /path/to/your_metadata.csv \
  --output-root "${WORK_COVER}/production_splitsv2"
```

---

## 5. Submit all nested folds (SLURM)

Edit `#SBATCH` lines and the path variables inside the script if needed, then:

```bash
sbatch gpstmd/github_uploads/gnet_all_prod.sh
```

---

## 6. Optional: Foldseek + membrane deduplication

See **`gpstmd/readme.md`** and **`gpstmd/filter_foldseek_redundant_by_membrane.py`**.

```bash
python gpstmd/filter_foldseek_redundant_by_membrane.py \
  --clusters-tsv path/to/clusters_cluster.tsv \
  --dataset-csv path/to/final_pdb_csv2.csv \
  --output-csv path/to/final_pdb_csv2_nodupes.csv
```
