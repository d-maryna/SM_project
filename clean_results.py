"""Remove duplicate rows from the raw Orca results and fill null_type for old deletion rows.
    python clean_results.py ~/orca_full_out/results.csv sm-project-data/results.csv
"""
import sys
import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
d = pd.read_csv(src).drop_duplicates("vid", keep="last")
d.loc[(d["type"] == "del") & d.null_type.isna(), "null_type"] = "flank"
d.to_csv(dst, index=False)
print(f"{len(d)} variants written to {dst}")