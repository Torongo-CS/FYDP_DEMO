# FYDP — Federated vs Centralized learning on PAD-UFES-20

Code for the pipeline in the project doc "FL vs CL on PAD-UFES-20 — Corrected Pipeline".

## Folder layout

| Path | What it is |
| --- | --- |
| `fl_skin/data.py` | Load PAD-UFES-20, drop leaking columns (`biopsed`, ids), encode metadata with "was missing" flags |
| `fl_skin/splits.py` | Patient-level test/val/train split and the 5 client regimes (iid, natural, dir0.5, dir0.1, missing) |
| `fl_skin/models.py` | MobileNetV2 + image-only / concat / gated / cross-attention fusion |
| `fl_skin/train.py` | GPU-side augmentation, training loop, prediction |
| `fl_skin/engine.py` | CL, local-only, FL (FedAvg, FedProx, FedNova, SCAFFOLD, FedBN, FedAvg+fine-tune), metadata-only LR, calibration; checkpoints + resume |
| `fl_skin/metrics.py` | Balanced accuracy, F1, AUC, MCC, ECE, Brier, cancer sens/spec, patient bootstrap CIs, paired bootstrap, Holm |
| `fl_skin/run.py` | Command-line runner (one process per GPU, time guard) |
| `fl_skin/restore.py` | Copies checkpoints from the previous Kaggle session back into `/kaggle/working` |
| `scripts/prepare_splits.py` | Phase B: data report + split files from `metadata.csv` |
| `scripts/make_plan.py` | Writes the run lists in `plans/` |
| `kaggle_runner.ipynb` | The notebook run on Kaggle each session |
| `tests/` | 24 tests on a fake dataset (`python -m pytest -q tests`) |

## How checkpoints survive on Kaggle

1. Every run writes `runs/<run_id>/state.pt` after each epoch (CL) or round (FL), plus `best.pt` (best on validation).
2. The runner stops starting new work after `HOURS` (default 11 h), so the session ends before Kaggle's 12 h limit and **Kaggle saves `/kaggle/working` as the notebook's output**. (A session that hits the limit loses its output.)
3. The next session attaches the notebook's own previous output as an input; cell 3 copies every run folder back, finished runs are skipped and paused runs resume from their last epoch/round with the same data order.

## Design choices made in code (beyond the doc)

- **One optimizer for all methods:** SGD, momentum 0.9, lr 0.01, cosine. FedNova and SCAFFOLD are defined for SGD steps, so AdamW would make them invalid; using SGD for CL too keeps the comparison fair.
- **Client validation sets:** the global validation set is split between clients with the same rule as training, so CL's train/val are exactly the union of the clients'.
- **Metadata scaling in FL:** built from per-client sums that are added (only totals shared), so all clients use the same scale. Missing values are filled with the mean (0 after scaling) plus a "was missing" flag.
- **Test set is sealed:** test probabilities are written to `preds_sealed.npz` at the end of each run and are not read until the final analysis. Model selection only uses validation.
- **SCAFFOLD with momentum:** control variates use FedNova's effective step count for momentum SGD.
