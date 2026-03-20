# gpstmd

Training / split utilities live under `github_uploads/`.

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

Local reference path (splits):  
`/work/hdd/bdja/bpokhrel/new_gearnet/foldseek_train/whole_test_cover/production_splitsv2`
