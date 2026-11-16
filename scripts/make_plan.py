"""Write the run lists (plans) for each stage of the schedule.

python scripts/make_plan.py calib
python scripts/make_plan.py week1a                       # CL arm + metadata baseline
python scripts/make_plan.py week1b --fusion concat       # local-only + FedProx tuning
python scripts/make_plan.py week2  --fusion concat --mu 0.01
python scripts/make_plan.py ablations --fusion concat --mu 0.01
python scripts/make_plan.py ablations5 --fusion concat   # 5-client runs (use splits_k5.csv)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fl_skin.config import RunConfig, save_plan  # noqa: E402
from fl_skin.models import FUSIONS  # noqa: E402
from fl_skin.splits import REGIMES  # noqa: E402

SEEDS = [0, 1, 2]
ALGOS = ["fedavg", "fedprox", "fednova", "scaffold", "fedbn"]


def plan(stage, fusion, mu):
    R = []
    if stage == "calib":
        R += [RunConfig("calib__image", "calib", fusion="image"),
              RunConfig("calib__xattn", "calib", fusion="xattn")]
    elif stage == "week1a":
        R += [RunConfig(f"meta_lr__s{s}", "meta_lr", seed=s) for s in SEEDS]
        R += [RunConfig(f"cl__{f}__s{s}", "cl", fusion=f, seed=s) for s in SEEDS for f in FUSIONS]
    elif stage == "week1b":
        R += [RunConfig(f"tune__fedprox_mu{m}__dir0.5__s0", "fl", fusion=fusion, regime="dir0.5",
                        algo="fedprox", mu=m, seed=0) for m in (0.001, 0.01, 0.1)]
        R += [RunConfig(f"local__{fusion}__{r}__s{s}", "local", fusion=fusion, regime=r, seed=s)
              for s in SEEDS for r in REGIMES]
    elif stage == "week2":
        R += [RunConfig(f"fl__{a}__{fusion}__{r}__s{s}", "fl", fusion=fusion, regime=r, algo=a,
                        mu=mu, seed=s) for s in SEEDS for r in REGIMES for a in ALGOS]
    elif stage == "ablations":
        R += [RunConfig(f"abl_e5__{a}__{fusion}__dir0.1__s{s}", "fl", fusion=fusion, regime="dir0.1",
                        algo=a, local_epochs=5, seed=s) for s in SEEDS for a in ("fedavg", "fedbn")]
        R += [RunConfig(f"abl_img__fedavg__image__natural__s{s}", "fl", fusion="image",
                        regime="natural", algo="fedavg", seed=s) for s in SEEDS]
    elif stage == "ablations5":
        R += [RunConfig(f"abl_k5__{a}__{fusion}__{r}__s{s}", "fl", fusion=fusion, regime=r, algo=a,
                        n_clients=5, seed=s) for s in SEEDS for r in ("iid", "dir0.5")
              for a in ("fedavg", "fedbn")]
    else:
        raise SystemExit(f"unknown stage {stage}")
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--fusion", default="concat")
    ap.add_argument("--mu", type=float, default=0.01)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "plans"))
    a = ap.parse_args()
    runs = plan(a.stage, a.fusion, a.mu)
    path = os.path.join(a.outdir, f"{a.stage}.json")
    save_plan(runs, path)
    print(f"{len(runs)} runs -> {path}")
