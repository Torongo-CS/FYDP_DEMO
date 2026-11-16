# Guide: continuing the FL vs CL skin-lesion benchmark

This guide is for continuing the project on your own Kaggle quota (30 GPU hours/week).
Read `STATUS.md` first for the full plan and decisions. This file covers only the steps to run.

## 0. What you need

- A Kaggle account with phone verification (needed for GPU).
- The code: `git clone https://github.com/Torongo-CS/FYDP_DEMO.git`
- The data: PAD-UFES-20 from Kaggle, "Skin Cancer (PAD-UFES-20)" (source: kaggle.com/datasets/mahdavi1202/skin-cancer, CC BY 4.0). The images are NOT in the repo. Add this dataset to your notebook.
- Upload the code as a Kaggle dataset: create dataset, zip the repo folder (without `.git` and `data/`), name it `fydp-fl-skin-code`.

## 1. Set up the Kaggle notebook

1. New notebook. Settings: Accelerator = GPU T4 x2, Internet = On, Persistence = No persistence.
2. Add input: your `fydp-fl-skin-code` dataset.
3. Add input: "Skin Cancer (PAD-UFES-20)".
4. Cell 1 (the bootstrap cell). Set the stage name and splits:

```python
PLAN = "week1a"          # then week1b, week2, ablations, ablations5
SPLITS = "splits_k3.csv"
HOURS = 11.0
import json, glob
NBP = [p for p in glob.glob("/kaggle/input/**/kaggle_runner.ipynb", recursive=True) if "code_snapshot" not in p][0]
CELLS = [c for c in json.load(open(NBP))["cells"] if c["cell_type"] == "code"][1:]
exec("\n".join("".join(c["source"]) for c in CELLS))
```

The cells after cell 1 come from `kaggle_runner.ipynb` in the repo, so the code is loaded from the dataset.

## 2. Resume from earlier output (IMPORTANT)

Runs that finished are skipped automatically, but only if their output is visible to the new session.

- Before running, add the **previous version's output** as an input (in the notebook: Add Input → Notebook output → pick the last version).
- The log should then say `restored N run folders` with N > 0.
- If it says 0 after a previous run finished, the previous output was not attached. Fix the input first.

## 3. Run

1. Save Version → Save & Run All (Commit).
2. A session stops itself at about 11 hours and saves checkpoints. Runs that did not finish are continued next time.
3. When a session ends with unfinished runs, repeat step 2 with the newest output as input.
4. Stages run in this order: week1a (15 runs), week1b (18), week2 (75), ablations (9), ablations5 (12).

## 4. Save results back to the repo

1. Download the output of the finished version (Output tab → Download all).
2. Unzip it into `results/<stage>/` in your local copy, e.g. `results/week1a/`.
3. Commit with a message such as `results: week1a finished (15 runs)`.
4. Push: `git add results && git commit -m "..." && git push`.

Keep `done.json` and the results csv in git. Keep checkpoint files (`state.pt`) out of git (they are large); they are only needed to resume a run in the middle.

## 5. Checking progress

Each run has a `done.json` file when it finishes. In the notebook log, look for lines like:
- `N runs finished`
- `done` in the status table
- `exit codes: [0, 0]` (both GPU workers finished without error)

Non-zero exit code: open the log, copy the traceback, and send it to the project owner.

## 6. Rules to keep the results valid

- Do not change `fl_skin/data.py` (`FEATURE_SET = "safe"` must stay). The `biopsed` field leaks the label and must not be used.
- Do not use the sealed test set until Phase E (the final analysis). Results are read from validation until then.
- Do not re-split the data. Use `splits_k3.csv` from the repo.
- If you change any setting, record it in `STATUS.md` under "Decisions" before running.

## 7. Quota planning

- One stage at a time. Check the quota on the Kaggle accelerator panel before starting.
- Estimated GPU time: week1a about 1 h, week1b about 1.5 h, week2 about 6 h, ablations about 0.75 h, ablations5 about 1 h. Total about 10 h on one T4 (roughly 5–6 h on two T4s).
- Two stages fit in one week's quota.

## 8. If something breaks

- `restored 0 run folders` after a finished run: the previous output was not added as input (see step 2).
- `CUDA out of memory`: lower `batch` in the plan, or use the smaller model setting, and note it in `STATUS.md`.
- Test failures (`pytest tests/`) on your machine: run `pip install -r requirements.txt` if present, else `pip install torch torchvision pandas numpy scikit-learn pillow pytest`, then `pytest -q tests/`.
- Anything else: send the notebook log and the `STATUS.md` line you were on to the project owner.
