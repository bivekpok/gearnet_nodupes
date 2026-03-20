# Training: nested CV & production runs

This repository’s **default landing page** describes how to run nested cross-validation with
`train_splits.py` and the SLURM driver `gnet_all_prod.sh`.

**Project overview** (dataset, architecture, `protein_classification_reproducible.py`, citations):
see **[`README_GPSforTMDs.md`](README_GPSforTMDs.md)** (formerly the main `README.md`).

---

## Layout

| Path | Purpose |
|------|---------|
| `gpstmd/github_uploads/train_splits.py` | Train **one inner fold** from manifest CSVs under `--split_dir` |
| `gpstmd/github_uploads/gearnet_modules.py` | Imported by `train_splits.py` (run from `github_uploads/`) |
| `gpstmd/github_uploads/gnet_all_prod.sh` | SLURM batch: loop all `Outer_Fold_*/Inner_Fold_*` |
| `gpstmd/filter_foldseek_redundant_by_membrane.py` | Optional: Foldseek + membrane-label dedup before splits |

**Example local checkout** (Delta):  
`/work/hdd/bdja/bpokhrel/new_gearnet/foldseek_train/whole_test_cover/github_uploads/train_splits.py`

---

## 1. Clone & environment

```bash
git clone https://github.com/bivekpok/gearnet_nodupes.git
cd gearnet_nodupes
conda env create -f environment.yml   # optional if you use a pre-built env
conda activate gearnet
```

---

## 2. Environment variables (from `gnet_all_prod.sh`, Delta / SLURM)

```bash
module purge
export NUMEXPR_MAX_THREADS=8
export WANDB_DIR=/tmp
export WANDB_CACHE_DIR=/tmp
source /u/bpokhrel/miniconda3/etc/profile.d/conda.sh
conda activate /u/bpokhrel/gearnet   # or: conda activate gearnet
```

---

## 3. Run one inner fold (`train_splits.py`)

`train_splits.py` must run from the folder that contains **`gearnet_modules.py`** (same directory as the script).

```bash
cd gpstmd/github_uploads

python3 train_splits.py \
  --pdb_folder "/work/hdd/bdja/bpokhrel/gearnet_files/pdb_m15_ac" \
  --soluble_folder_ac "/work/hdd/bdja/bpokhrel/gearnet_files/watersoluble_proteins_ac" \
  --split_dir "/work/hdd/bdja/bpokhrel/new_gearnet/foldseek_train/whole_test_cover/production_splitsv2/Outer_Fold_1/Inner_Fold_1" \
  --output_dir "/work/hdd/bdja/bpokhrel/new_gearnet/foldseek_train/whole_test_cover/production_models_v2" \
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

Adjust `--pdb_folder`, `--soluble_folder_ac`, `--split_dir`, and `--output_dir` for your machine.
`gnet_all_prod.sh` sets these as variables (`PDB_DIR`, `SOLUBLE_DIR`, `BASE_SPLIT_DIR`, `OUTPUT_DIR`) and loops over every inner fold.

---

## 4. Submit all nested folds (SLURM)

Edit `#SBATCH` lines and the path variables inside the script if needed, then:

```bash
sbatch gpstmd/github_uploads/gnet_all_prod.sh
```

---

## 5. Optional: Foldseek + membrane deduplication

See **`gpstmd/readme.md`** and **`gpstmd/filter_foldseek_redundant_by_membrane.py`**.

```bash
python gpstmd/filter_foldseek_redundant_by_membrane.py \
  --clusters-tsv path/to/clusters_cluster.tsv \
  --dataset-csv path/to/final_pdb_csv2.csv \
  --output-csv path/to/final_pdb_csv2_nodupes.csv
```
