# Path reference (portable setup)

Everything under `gpstmd/github_uploads/` was originally written for a **specific Delta HPC layout** and user home. Below: what each path is for, and how to replace it.

## Suggested layout (any machine)

```text
${DATA_ROOT}/
├── pdb_m15_ac/                 # membrane PDBs
├── watersoluble_proteins_ac/     # soluble PDBs
└── metadata.csv                # e.g. fs_sl_df_noGolgiLysosomeVacuole.csv

${WORK_ROOT}/                   # e.g. whole_test_cover
├── production_splitsv2/        # nested split manifests (Outer_Fold_*/Inner_Fold_*)
├── production_models_v2/       # training outputs (optional)
└── logs/                       # SLURM / training logs
```

## Environment variables (`gnet_all_prod.sh`)

| Variable | Meaning | Example (yours) |
|----------|---------|-----------------|
| `WORK_COVER` | Base dir for splits + default output (script `cd`s here) | `.../foldseek_train/whole_test_cover` |
| `GEARNET_PDB_ROOT` | Parent of membrane + soluble PDB folders | `.../gearnet_files` |
| `PDB_DIR` | Membrane PDBs (overrides default under `GEARNET_PDB_ROOT`) | `.../pdb_m15_ac` |
| `SOLUBLE_DIR` | Soluble PDBs | `.../watersoluble_proteins_ac` |
| `BASE_SPLIT_DIR` | Nested split root | `$WORK_COVER/production_splitsv2` |
| `OUTPUT_DIR` | Checkpoints / results | `$WORK_COVER/production_models_v2` |
| `CONDA_SH` | Conda `conda.sh` | `$HOME/miniconda3/etc/profile.d/conda.sh` |
| `CONDA_ENV_GEARNET` | Env name or prefix for `conda activate` | `gearnet` or `/path/to/env` |

**Before `sbatch`**, you can export these (see `paths.env.example`):

```bash
set -a
source paths.env   # your copy of paths.env.example
set +a
sbatch gnet_all_prod.sh
```

> **SLURM `#SBATCH --output` / `--error`**: Slurm does **not** expand shell variables in those lines. Either edit those two lines to match your `WORK_COVER/logs`, or submit with `sbatch -o ... -e ... gnet_all_prod.sh`.

## File-by-file

| File | Hardcoded / personal bits | What to do |
|------|---------------------------|------------|
| `gnet_all_prod.sh` | `#SBATCH` logs, partition, account, email; conda path; `WORK_COVER` defaults | Edit `#SBATCH` for your cluster; set env vars or change defaults at top of script |
| `uplosd.sh` | Logs, conda, Hugging Face `folder_path`, `repo_id` | Treat as **personal**; set `HF_UPLOAD_FOLDER`, `HF_REPO_ID`, etc., or do not use |
| `datasplitterv2.py` | Was: CSV + output root in `__main__` | Use `python datasplitterv2.py --csv ... --output-root ...` |
| `train_splits.py` | None (paths via CLI) | Pass your `--pdb_folder`, `--soluble_folder_ac`, `--split_dir`, `--output_dir` |
| Root `README.md` | Example absolute paths | Replace with your paths; see variables above |
| `gpstmd/readme.md` | Local reference path | Same |
| `../filter_foldseek_redundant_by_membrane.py` | None (CLI only) | Pass `--clusters-tsv` and `--dataset-csv` |
| Repo `env2.yml` | `prefix: /u/.../miniconda3` | **Remove or change `prefix`** if you share the file; prefer `environment.yml` without prefix |

## Quick copy-paste (generic)

```bash
export WORK_COVER="/path/to/whole_test_cover"
export GEARNET_PDB_ROOT="/path/to/gearnet_files"
export CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_ENV_GEARNET="gearnet"

cd /path/to/gearnet_nodupes/gpstmd/github_uploads
python3 datasplitterv2.py --csv /path/to/metadata.csv --output-root "$WORK_COVER/production_splitsv2"

python3 train_splits.py \
  --pdb_folder "$GEARNET_PDB_ROOT/pdb_m15_ac" \
  --soluble_folder_ac "$GEARNET_PDB_ROOT/watersoluble_proteins_ac" \
  --split_dir "$WORK_COVER/production_splitsv2/Outer_Fold_1/Inner_Fold_1" \
  --output_dir "$WORK_COVER/production_models_v2" \
  # ... hyperparameters ...
```
