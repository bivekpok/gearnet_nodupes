# gpstmd

Training / split utilities live under `github_uploads/`.

**Portable paths:** see [`github_uploads/PATHS.md`](github_uploads/PATHS.md) and `github_uploads/paths.env.example`.

## SLURM: production CV (`gnet_all_prod.sh`)

`github_uploads/gnet_all_prod.sh` is the batch script for GearNet production cross-validation
(Delta-style `#SBATCH` directives, conda env, W&B temp dirs). **Edit paths** (logs, working dir,
conda) if you run on another cluster.

Submit from the directory where your training code expects paths, e.g.:

```bash
sbatch github_uploads/gnet_all_prod.sh
```

## Foldseek + membrane deduplication

`filter_foldseek_redundant_by_membrane.py` implements the same logic as the Colab snippet:
load Foldseek cluster pairs, map `membrane_name_cache` (or another label column) from your
PDB table, drop self-hits, discard **Member** PDBs that share the same membrane label as
their **Representative**, then optionally write a cleaned CSV.

```bash
python filter_foldseek_redundant_by_membrane.py \
  --clusters-tsv path/to/s03tm03_clusters_cluster.tsv \
  --dataset-csv path/to/final_pdb_csv2.csv \
  --output-csv path/to/final_pdb_csv2_nodupes.csv
```

Optional: `--pdb-column`, `--label-column` if your table uses different names.

---

Example split output directory (set `WORK_COVER` / `--output-root` on your machine):

`$WORK_COVER/production_splitsv2`
