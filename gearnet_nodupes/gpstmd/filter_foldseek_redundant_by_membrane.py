#!/usr/bin/env python3
"""
Remove redundant PDBs using Foldseek cluster membership + membrane location.

Logic (same as Colab notebook):
1. Load Foldseek cluster TSV: two columns, representative vs member.
2. Load a dataset with PDB IDs and membrane labels.
3. Map labels onto cluster pairs; drop self-hits (rep == member).
4. Members that share the same membrane label as their representative are
   treated as redundant; those member PDB IDs are discarded from the dataset.

Example (paths like your Drive layout):
  python filter_foldseek_redundant_by_membrane.py \\
    --clusters-tsv foldseek/s03tm03_clusters_cluster.tsv \\
    --dataset-csv final_pdb_csv2.csv \\
    --output-csv final_pdb_csv2_nodupes.csv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter dataset by Foldseek redundancy + matching membrane label."
    )
    p.add_argument(
        "--clusters-tsv",
        required=True,
        help="Foldseek cluster file: tab-separated, col0=Representative, col1=Member",
    )
    p.add_argument(
        "--dataset-csv",
        required=True,
        help="CSV with pdbid and membrane label columns",
    )
    p.add_argument(
        "--pdb-column",
        default="pdbid",
        help="Column name for PDB ID in dataset (default: pdbid)",
    )
    p.add_argument(
        "--label-column",
        default="membrane_name_cache",
        help="Membrane location column (default: membrane_name_cache)",
    )
    p.add_argument(
        "--output-csv",
        default=None,
        help="If set, write filtered dataset to this CSV",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    seq_clus = pd.read_csv(
        args.clusters_tsv,
        sep="\t",
        header=None,
        names=["Representative", "Member"],
    )

    new_data = pd.read_csv(args.dataset_csv)
    pdb_col = args.pdb_column
    label_col = args.label_column

    if pdb_col not in new_data.columns:
        print(f"Missing column {pdb_col!r} in dataset.", file=sys.stderr)
        return 1
    if label_col not in new_data.columns:
        print(f"Missing column {label_col!r} in dataset.", file=sys.stderr)
        return 1

    label_map = new_data.set_index(pdb_col)[label_col]

    pairs = seq_clus.copy()
    pairs["rep_label"] = pairs["Representative"].map(label_map)
    pairs["mem_label"] = pairs["Member"].map(label_map)

    cross_members = pairs[pairs["Representative"] != pairs["Member"]]
    same_loc_cross_member = cross_members[cross_members["rep_label"] == cross_members["mem_label"]]
    pdbs_to_discard = same_loc_cross_member["Member"].tolist()

    final_clean_df = new_data[~new_data[pdb_col].isin(pdbs_to_discard)]

    print(f"Original dataset size: {len(new_data)}")
    print(f"Redundant PDBs discarded: {len(pdbs_to_discard)}")
    print(f"Final clean dataset size: {len(final_clean_df)}")
    print("\n--- Final class balance ---")
    print(final_clean_df[label_col].value_counts())

    if args.output_csv:
        final_clean_df.to_csv(args.output_csv, index=False)
        print(f"\nWrote: {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
