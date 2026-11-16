# FYDP_FL_Skin: status, plan, and how to continue

Last updated: 2026-11-16

## 1. What this project is

Benchmark-only study on PAD-UFES-20 (skin lesion images with patient questionnaire metadata).
We compare federated learning (FL) and centralized learning (CL) under several data splits,
using only the questionnaire fields that do not leak the label (`safe` feature set:
age, itch, hurt, bleed, elevation, region). No new method is proposed.

Scope decision (approved): benchmark only. Pixel-wise spatial gating and harm score are NOT part of this work.

## 2. What is already done

| Item | Status | Where |
|---|---|---|
| Source check and pipeline critique | Done | Project doc (Claude Docs, FYDP_1) |
| Leakage fix: `biopsed` removed; 804 images without questionnaire (all benign) handled by a "missing" regime; metadata used with `safe` fields only | Done | `fl_skin/data.py`, `splits/` |
| Patient-level splits (StratifiedGroupKFold), client partitions (IID, Dirichlet 0.5 and 0.1, natural by body region, missing-metadata) | Done | `scripts/prepare_splits.py`, `splits/` |
| Unit tests (25), including leakage and safe-field checks | Done | `tests/` |
| Checkpoint and resume design (state.pt per epoch/round, done.json, time guard) | Done | `fl_skin/engine.py`, `fl_skin/restore.py` |
| Kaggle dataset `fydp-fl-skin-code` (code, splits, plans) | Done | Kaggle: torongocs/fydp-fl-skin-code |
| Kaggle notebook `fydp-fl-runner` (T4 x2, Internet on) | Done | Kaggle: torongocs/fydp-fl-runner |
| Calibration run (2 runs) | Done, passed | Kaggle version 1 "calib" |

Calibration results (T4, MobileNetV2, 1,575 training images):
- about 4.5 s per training epoch and 1 s per validation pass
- overfit check (64 images): training accuracy 1.0, passed
- estimated time: centralized 50 epochs about 4.5 min; federated 30 rounds x 2 local epochs about 4.9 min
- the `best_val_bacc` field in the calibration output is empty (`None`). This is a logging gap in the calibration step only, not a training failure.

## 3. Planned runs (129 total)

| Stage | Runs | Purpose | Est. time (1 T4) |
|---|---|---|---|
| week1a | 15 | Centralized learning (4 fusions x 3 seeds) + metadata-only logistic regression (3 seeds) | about 75 min |
| week1b | 18 | FedProx mu tuning (3 values) + local-only training (5 regimes x 3 seeds) | about 90 min |
| week2 | 75 | Main FL grid: 5 algorithms x 5 regimes x 3 seeds (fusion set by plan) | about 6 h |
| ablations | 9 | Local epochs = 5 on dir0.1; image-only baseline on natural split | about 45 min |
| ablations5 | 12 | 5-client setting (iid and dir0.5, FedAvg and FedBN) | about 1 h |

Total GPU time about 9 h on one T4. The two T4s run two shards in parallel, so the wall-clock time is about 5 h
plus overhead. The weekly quota (30 h) is enough for all stages in one week.

## 4. How to continue (for whoever runs the next stage)

1. Open the Kaggle notebook `fydp-fl-runner` (or create one with the same cells: see `kaggle_runner.ipynb`).
2. Add inputs: the dataset `fydp-fl-skin-code` and the dataset "Skin Cancer(PAD-UFES-20)" (source: kaggle.com/datasets/mahdavi1202/skin-cancer, CC BY 4.0).
3. Settings: Accelerator GPU T4 x2, Internet on, Persistence none.
4. Add the previous version's output as an input so `restore.py` can resume finished runs (`restored N run folders`).
5. In cell 1 set `PLAN = "week1a"` (then week1b, week2, ablations, ablations5), `SPLITS = "splits_k3.csv"`, `HOURS = 11.0`.
6. Save Version, Save & Run All. Each session stops itself before 12 hours and writes checkpoints.
7. When a session stops with unfinished runs, repeat step 4 with the newest output as input and run again.
8. Download the output (runs/ folders with `done.json`, results csv, and checkpoints) and commit it into `results/<stage>/` in this repo.

Runs that already have `done.json` are skipped automatically.

## 5. Decisions made so far

- Metadata: safe fields only (no `biopsed`, no questionnaire-derived leak).
- Optimizer: SGD momentum 0.9, lr 0.01, weight decay 1e-4.
- Scope: benchmark only. No new method.
- Test set: sealed, used once in the final analysis (Phase E).

## 6. What is left

- Phase D: week1a, week1b, week2, ablations, ablations5 (all on Kaggle, with approval before each GPU run).
- Phase E: sealed test analysis (bootstrap CIs, paired differences with Holm correction), results into the project doc.
- Write the results section into the project doc.

## 7. Repository contents

- `fl_skin/`: code (data, splits, models, training, FL algorithms, metrics, runner, restore)
- `scripts/`: split preparation and plan generation
- `plans/`: run lists (calib, week1a; other stages generated with `scripts/make_plan.py <stage>`)
- `splits/`: fixed splits (k=3, k=5), client tables, data report, metadata reference copy
- `tests/`: 25 tests
- `kaggle_runner.ipynb`: the Kaggle notebook
- `results/`: Kaggle outputs per stage (added after each stage)

Not included in the repo: the PAD-UFES-20 images (download from Kaggle, see section 4).
