"""Phase B: check the data and make the split files (needs only metadata.csv).

python scripts/prepare_splits.py path/to/metadata.csv --k 3 --out splits/
Writes: splits_k3.csv, data_report.json, client_tables_k3.txt
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fl_skin import CLASSES  # noqa: E402
from fl_skin.data import dataset_report, load_metadata, missingness_audit  # noqa: E402
from fl_skin.splits import REGIMES, client_class_table, make_all_splits  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("metadata")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--min-images", type=int, default=150)
    ap.add_argument("--out", default="splits")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load_metadata(a.metadata)
    rep = dataset_report(df)
    rep["missingness_audit"] = missingness_audit(df)
    bad = [c for c, v in rep["missingness_audit"].items() if v["suspicious"]]
    if bad:
        raise SystemExit(f"STOP: missingness of {bad} differs by class -> label leak")
    with open(os.path.join(a.out, "data_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    sp = make_all_splits(df, k=a.k, seed=a.seed, min_images=a.min_images)
    sp.to_csv(os.path.join(a.out, f"splits_k{a.k}.csv"), index=False)
    with open(os.path.join(a.out, f"client_tables_k{a.k}.txt"), "w") as f:
        f.write(f"global split (images):\n{sp['split'].value_counts().to_string()}\n\n")
        for r in REGIMES:
            if (sp.loc[sp["split"] != "test", r] == -1).all():
                continue
            f.write(f"== {r} ==\n{client_class_table(sp, r, CLASSES).to_string()}\n\n")
    print(json.dumps(rep, indent=1, default=str))
    print(open(os.path.join(a.out, f"client_tables_k{a.k}.txt")).read())
