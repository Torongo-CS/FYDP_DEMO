"""Command-line runner used on Kaggle (one process per GPU).

python -m fl_skin.run --plan plans/week1.json --data /kaggle/input --splits splits/splits.csv \
    --out /kaggle/working --shard 0 --nshards 2 --hours 11.0

- Runs the plan's runs whose index % nshards == shard, in order.
- Skips runs that already have done.json; resumes runs that have state.pt.
- Stops cleanly before the deadline so Kaggle can save /kaggle/working
  (a notebook that hits the 12 h limit loses its outputs).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch

from .config import load_plan
from .data import build_image_cache, find_dataset_root, load_metadata
from .engine import Budget, Context, Paused, run_one


def build_context(data_dir, splits_path, out_dir, device, img_size=256, crop=224, cache_dir="/tmp",
                  need_weights=True):
    meta_path, index = find_dataset_root(data_dir)
    df = load_metadata(meta_path, index)
    sp = pd.read_csv(splits_path)
    sp["patient_id"] = sp["patient_id"].astype(str)
    df = df.set_index("img_id").loc[sp["img_id"]].reset_index() if "img_id" in sp else df
    df["row"] = np.arange(len(df))
    sp["row"] = np.arange(len(sp))
    assert (df["patient_id"].to_numpy() == sp["patient_id"].to_numpy()).all(), "split file does not match data"
    cache = build_image_cache(df, os.path.join(cache_dir, f"padufes_{img_size}.npy"), img_size)
    images = np.load(cache)
    wp = os.path.join(out_dir, "weights", "mobilenet_v2_imagenet.pth")
    if not need_weights:
        wp = None
    elif not os.path.exists(wp):
        import torchvision
        os.makedirs(os.path.dirname(wp), exist_ok=True)
        m = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
        torch.save(m.state_dict(), wp)
    return Context(df, sp, images, device, crop, weights_path=wp)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--hours", type=float, default=11.0, help="stop starting work after this many hours")
    ap.add_argument("--start", type=float, default=None, help="session start (unix time)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--cache-dir", default="/tmp")
    ap.add_argument("--prepare-only", action="store_true",
                    help="build the image cache and weights once, then exit (run before the shards)")
    a = ap.parse_args(argv)
    if a.prepare_only:
        build_context(a.data, a.splits, a.out, "cpu", a.img_size, a.crop, a.cache_dir)
        print("cache and weights ready", flush=True)
        return

    start = a.start or time.time()
    budget = Budget(start + a.hours * 3600)
    runs = [r for i, r in enumerate(load_plan(a.plan)) if i % a.nshards == a.shard]
    for r in runs:
        assert (r.img_size, r.crop) == (a.img_size, a.crop), f"{r.run_id}: image size differs from session"
    ctx = build_context(a.data, a.splits, a.out, a.device, a.img_size, a.crop, a.cache_dir,
                        need_weights=any(r.pretrained for r in runs))
    res_path = os.path.join(a.out, f"results_shard{a.shard}.csv")
    for cfg in runs:
        print(f"[shard {a.shard}] {cfg.run_id}", flush=True)
        try:
            info = run_one(cfg, ctx, a.out, budget)
        except Paused:
            print(f"[shard {a.shard}] time is up; {cfg.run_id} saved and paused", flush=True)
            info = {"status": "paused", "run_id": cfg.run_id}
            _append(res_path, info)
            break
        except Exception as e:  # keep going with the other runs; the error is logged
            traceback.print_exc()
            info = {"status": "error", "run_id": cfg.run_id, "error": repr(e)[:300]}
        _append(res_path, {k: v for k, v in info.items() if not isinstance(v, (list, dict))})
        print(json.dumps(info, default=str)[:500], flush=True)


def _append(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["time", "json"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), json.dumps(row, default=str)])


if __name__ == "__main__":
    main()
