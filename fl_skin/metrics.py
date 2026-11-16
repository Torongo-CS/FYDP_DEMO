"""Metrics, bootstrap confidence intervals and paired comparisons (Stage 7 of the doc)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score

from . import CANCER_CLASSES, CLASSES

CANCER_IDX = [CLASSES.index(c) for c in CANCER_CLASSES]


def balanced_accuracy(y, p, sample_weight=None):
    pred = p.argmax(1)
    rec = []
    for c in range(p.shape[1]):
        m = y == c
        if not m.any():
            continue
        if sample_weight is not None and sample_weight[m].sum() <= 0:
            continue  # class absent from this client's view
        rec.append(np.mean(pred[m] == c))
    return float(np.mean(rec))


def weighted_accuracy(y, p, sample_weight):
    """Accuracy with every test image weighted to match a client's class mix."""
    return float(np.average(p.argmax(1) == y, weights=sample_weight))


def ece(y, p, n_bins=15):
    conf = p.max(1)
    acc = (p.argmax(1) == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def brier(y, p):
    oh = np.eye(p.shape[1])[y]
    return float(((p - oh) ** 2).sum(1).mean())


def cancer_score(p):
    return p[:, CANCER_IDX].sum(1)


def cancer_threshold(y_val, p_val, target_sens=0.90):
    """Highest threshold on validation that still reaches the target cancer sensitivity."""
    yc = np.isin(y_val, CANCER_IDX)
    s = cancer_score(p_val)
    for t in np.sort(np.unique(s))[::-1]:
        if (s[yc] >= t).mean() >= target_sens:
            return float(t)
    return 0.0


def all_metrics(y, p, threshold=None):
    pred = p.argmax(1)
    n_cls = p.shape[1]
    out = {
        "bacc": balanced_accuracy(y, p),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(n_cls)), zero_division=0)),
        "acc": float((pred == y).mean()),
        "mcc": float(matthews_corrcoef(y, pred)),
        "ece": ece(y, p),
        "brier": brier(y, p),
    }
    present = [c for c in range(n_cls) if (y == c).any()]
    try:
        out["macro_auc"] = float(roc_auc_score(y, p[:, present] / p[:, present].sum(1, keepdims=True),
                                               multi_class="ovr", labels=present, average="macro"))
    except ValueError:
        out["macro_auc"] = float("nan")
    f1s = f1_score(y, pred, average=None, labels=list(range(n_cls)), zero_division=0)
    for c in range(n_cls):
        out[f"f1_{CLASSES[c]}"] = float(f1s[c])
        m = y == c
        out[f"recall_{CLASSES[c]}"] = float((pred[m] == c).mean()) if m.any() else float("nan")
        out[f"n_{CLASSES[c]}"] = int(m.sum())
    yc = np.isin(y, CANCER_IDX)
    s = cancer_score(p)
    out["cancer_auc"] = float(roc_auc_score(yc, s)) if 0 < yc.sum() < len(yc) else float("nan")
    if threshold is not None:
        out["cancer_sens"] = float((s[yc] >= threshold).mean())
        out["cancer_spec"] = float((s[~yc] < threshold).mean())
    return out


def _patient_resample(groups, rng):
    ug = np.unique(groups)
    pick = rng.choice(ug, size=len(ug), replace=True)
    # map each picked patient to its rows
    order = np.argsort(groups, kind="stable")
    sg = groups[order]
    starts = np.searchsorted(sg, pick, "left")
    ends = np.searchsorted(sg, pick, "right")
    return np.concatenate([order[s:e] for s, e in zip(starts, ends)])


def bootstrap_ci(y, p, groups, metric=balanced_accuracy, n_boot=2000, seed=0, alpha=0.05):
    """Resample test PATIENTS with replacement; returns (point, low, high)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = _patient_resample(groups, rng)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(metric(y[idx], p[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(metric(y, p)), float(lo), float(hi)


def paired_bootstrap_diff(y, p_a, p_b, groups, metric=balanced_accuracy, n_boot=2000, seed=0):
    """Difference metric(A) - metric(B) on the same resampled patients.

    Returns (diff, low, high, p_two_sided). p is the share of resamples whose
    difference falls on the other side of zero, doubled (percentile method)."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = _patient_resample(groups, rng)
        d.append(metric(y[idx], p_a[idx]) - metric(y[idx], p_b[idx]))
    d = np.asarray(d)
    point = metric(y, p_a) - metric(y, p_b)
    lo, hi = np.quantile(d, [0.025, 0.975])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(point), float(lo), float(hi), float(min(1.0, p))


def holm(pvals: dict) -> dict:
    keys = sorted(pvals, key=pvals.get)
    m = len(keys)
    adj, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return adj


def client_view_weights(y, client_class_props):
    """Per-image weights that make the test set mimic a client's class mix (label-skew regimes)."""
    test_props = np.bincount(y, minlength=len(client_class_props)) / len(y)
    ratio = np.where(test_props > 0, client_class_props / np.maximum(test_props, 1e-12), 0)
    return ratio[y]


def worst_and_spread(per_client: list[float]):
    a = np.asarray(per_client, float)
    return float(a.min()), float(a.std(ddof=0))
