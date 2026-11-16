"""Bring checkpoints from a previous Kaggle session back into /kaggle/working.

Attach the notebook's own previous output as an input dataset; this copies
every run folder from there that is further along than the local copy.
"""
from __future__ import annotations

import glob
import os
import shutil


def _progress(run_dir: str) -> int:
    if os.path.exists(os.path.join(run_dir, "done.json")):
        return 10 ** 9
    p = os.path.join(run_dir, "log.csv")
    n = 0
    for lp in glob.glob(os.path.join(run_dir, "**", "log.csv"), recursive=True):
        with open(lp) as f:
            n += max(0, sum(1 for _ in f) - 1)
    return n if os.path.exists(p) or n else -1


def restore(input_root: str = "/kaggle/input", out_dir: str = "/kaggle/working") -> list[str]:
    restored = []
    for runs_dir in glob.glob(os.path.join(input_root, "**", "runs"), recursive=True):
        for src in sorted(glob.glob(os.path.join(runs_dir, "*"))):
            if not os.path.isdir(src):
                continue
            dst = os.path.join(out_dir, "runs", os.path.basename(src))
            if _progress(src) > _progress(dst):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                restored.append(os.path.basename(src))
    for extra in ("weights",):
        for src in glob.glob(os.path.join(input_root, "**", extra), recursive=True):
            dst = os.path.join(out_dir, extra)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)
    for res in glob.glob(os.path.join(input_root, "**", "results_shard*.csv"), recursive=True):
        dst = os.path.join(out_dir, "previous_" + os.path.basename(os.path.dirname(res)) + "_" + os.path.basename(res))
        if not os.path.exists(dst):
            shutil.copy(res, dst)
    return restored


if __name__ == "__main__":
    r = restore()
    print(f"restored {len(r)} run folders: {r}")
