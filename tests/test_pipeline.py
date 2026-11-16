"""Run with:  python -m pytest -q tests"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from fake_data import make_fake  # noqa: E402
from fl_skin import CLASSES  # noqa: E402
from fl_skin import engine as E  # noqa: E402
from fl_skin.config import RunConfig, save_plan  # noqa: E402
from fl_skin.data import (LEAK_OR_ID_COLUMNS, MetaEncoder, MetaStats, build_image_cache,  # noqa: E402
                          find_dataset_root, load_metadata)
from fl_skin.metrics import (all_metrics, balanced_accuracy, bootstrap_ci, cancer_threshold,  # noqa: E402
                             holm, paired_bootstrap_diff)
from fl_skin.splits import REGIMES, make_all_splits  # noqa: E402

TINY = dict(img_size=64, crop=56, pretrained=False, epochs=3, patience=10, warmup=1,
            rounds=3, local_epochs=1, batch_size=16, amp=False)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("pad"))
    make_fake(root, n_images=420)
    meta_path, index = find_dataset_root(root)
    df = load_metadata(meta_path, index)
    sp = make_all_splits(df, k=3, seed=1, min_images=40)
    sp_path = os.path.join(root, "splits_k3.csv")
    sp.to_csv(sp_path, index=False)
    cache = build_image_cache(df, os.path.join(root, "cache64.npy"), 64)
    ctx = E.Context(df, sp, np.load(cache), "cpu", crop=56)
    return {"root": root, "df": df, "sp": sp, "ctx": ctx, "sp_path": sp_path}


# ------------------------------------------------------------------ data ---

def test_no_patient_leak(data):
    sp = data["sp"]
    assert (sp.groupby("patient_id")["split"].nunique() == 1).all()
    for r in REGIMES:
        nt = sp[sp.split != "test"]
        assert (nt.groupby("patient_id")[r].nunique() == 1).all()
        assert set(nt[r].unique()) == {0, 1, 2}


def test_no_leak_columns_in_features(data):
    names = data["ctx"].enc.names
    for col in LEAK_OR_ID_COLUMNS:
        assert not any(n == col or n.startswith(col + "__") or n.startswith(col + "=") for n in names)


def test_missing_regime(data):
    sp = data["sp"]
    nt = sp[sp.split != "test"]
    c2 = nt[nt["missing"] == 2]["meta_blank__missing"]
    assert (c2 == 1).all(), "client 2 must have no metadata"
    other = nt[nt["missing"] != 2]["meta_blank__missing"].mean()
    assert 0.15 < other < 0.45
    X = data["ctx"].meta["missing"]
    miss_cols = [i for i, n in enumerate(data["ctx"].enc_missing.names) if n.endswith("__missing")]
    rows = nt[nt["missing"] == 2]["row"].to_numpy()
    assert (X[np.ix_(rows, miss_cols)] == 1).all()


def test_stats_are_additive(data):
    df = data["df"]
    a, b = df.iloc[:200], df.iloc[200:]
    s1 = (MetaStats.from_frame(a) + MetaStats.from_frame(b)).mean_std()
    s2 = MetaStats.from_frame(df).mean_std()
    for k in s1:
        assert np.allclose(s1[k], s2[k])


def test_encoder_blank_rows(data):
    enc = MetaEncoder(MetaStats.from_frame(data["df"]))
    X = enc.transform(data["df"].iloc[:5], blank=np.ones(5, bool))
    flags = [i for i, n in enumerate(enc.names) if n.endswith("__missing")]
    assert (X[:, flags] == 1).all()


# ----------------------------------------------------------- aggregation ---

def test_fedavg_and_fednova_equal_when_equal_steps():
    g = {"w": torch.zeros(3), "n": torch.tensor(5)}
    sds = [{"w": torch.tensor([1.0, 2, 3]), "n": torch.tensor(6)},
           {"w": torch.tensor([3.0, 2, 1]), "n": torch.tensor(7)}]
    avg = E.aggregate(sds, [0.25, 0.75], set(), "fedavg", g, trainable={"w"})
    assert torch.allclose(avg["w"], torch.tensor([2.5, 2, 1.5]))
    assert int(avg["n"]) == 7
    nova = E.aggregate(sds, [0.25, 0.75], set(), "fednova", g, taus=[10, 10], rho=0.9, trainable={"w"})
    assert torch.allclose(nova["w"], avg["w"], atol=1e-6)
    nova2 = E.aggregate(sds, [0.25, 0.75], set(), "fednova", g, taus=[5, 20], rho=0.9, trainable={"w"})
    assert not torch.allclose(nova2["w"], avg["w"])


def test_fedbn_skips_bn():
    g = {"bn.weight": torch.zeros(2), "fc.weight": torch.zeros(2)}
    sds = [{"bn.weight": torch.ones(2), "fc.weight": torch.ones(2)}] * 2
    out = E.aggregate(sds, [0.5, 0.5], {"bn.weight"}, "fedbn", g, trainable=set(g))
    assert (out["bn.weight"] == 0).all() and (out["fc.weight"] == 1).all()


# -------------------------------------------------------------- runs -------

def _cfg(run_id, kind, **kw):
    return RunConfig(run_id, kind, **{**TINY, **kw})


@pytest.mark.parametrize("cfg", [
    _cfg("meta", "meta_lr"),
    _cfg("cl_image", "cl", fusion="image"),
    _cfg("cl_concat", "cl", fusion="concat"),
    _cfg("cl_gated", "cl", fusion="gated"),
    _cfg("cl_xattn", "cl", fusion="xattn"),
    _cfg("local_nat", "local", fusion="concat", regime="natural"),
    _cfg("fl_fedavg", "fl", algo="fedavg", regime="dir0.5"),
    _cfg("fl_fedprox", "fl", algo="fedprox", regime="iid", mu=0.1),
    _cfg("fl_fednova", "fl", algo="fednova", regime="dir0.1"),
    _cfg("fl_scaffold", "fl", algo="scaffold", regime="natural"),
    _cfg("fl_fedbn", "fl", algo="fedbn", regime="missing", fusion="xattn"),
    _cfg("calib", "calib", fusion="image", overfit_n=16),
], ids=lambda c: c.run_id)
def test_every_run_type(data, tmp_path, cfg):
    res = E.run_one(cfg, data["ctx"], str(tmp_path), E.Budget(None))
    d = tmp_path / "runs" / cfg.run_id
    assert (d / "done.json").exists()
    if cfg.kind == "calib":
        assert "seconds_per_epoch" in res
        return
    sealed = np.load(d / "preds_sealed.npz")
    keys = list(sealed.keys())
    assert any(k.endswith("test_full") for k in keys)
    for k in keys:
        if k != "rows":
            assert np.allclose(sealed[k].sum(1), 1, atol=1e-3)
    assert not list(d.rglob("state.pt")), "resume state should be removed after finishing"
    if cfg.kind == "fl":
        log = pd.read_csv(d / "log.csv")
        assert len(log) == cfg.rounds and log["mb_uploaded"].is_monotonic_increasing
    if cfg.algo == "fedbn" and cfg.kind == "fl":
        p = torch.load(d / "personal.pt", weights_only=False)["personal"]
        bn = [k for k in p[0] if k.endswith("running_mean")][0]
        assert not torch.allclose(p[0][bn], p[1][bn]), "FedBN clients should keep different BN stats"


class PauseAfter(E.Budget):
    def __init__(self, n):
        super().__init__(None)
        self.n = n

    def check(self, need_s):
        self.n -= 1
        if self.n < 0:
            raise E.Paused()


@pytest.mark.parametrize("cfg", [
    _cfg("resume_cl", "cl", fusion="concat", epochs=4),
    _cfg("resume_fl", "fl", algo="scaffold", regime="dir0.5", rounds=4),
    _cfg("resume_fedbn", "fl", algo="fedbn", regime="iid", rounds=4),
], ids=lambda c: c.run_id)
def test_resume_gives_identical_result(data, tmp_path, cfg):
    ctx = data["ctx"]
    E.run_one(cfg, ctx, str(tmp_path / "a"), E.Budget(None))
    with pytest.raises(E.Paused):
        E.run_one(cfg, ctx, str(tmp_path / "b"), PauseAfter(2))
    assert not (tmp_path / "b" / "runs" / cfg.run_id / "done.json").exists()
    E.run_one(cfg, ctx, str(tmp_path / "b"), E.Budget(None))
    la = pd.read_csv(tmp_path / "a" / "runs" / cfg.run_id / "log.csv")
    lb = pd.read_csv(tmp_path / "b" / "runs" / cfg.run_id / "log.csv")
    col = [c for c in la.columns if c.startswith("val_bacc")]
    assert len(la) == len(lb)
    assert np.allclose(la[col].to_numpy(), lb[col].to_numpy(), atol=1e-4)
    pa = np.load(tmp_path / "a" / "runs" / cfg.run_id / "preds_sealed.npz")
    pb = np.load(tmp_path / "b" / "runs" / cfg.run_id / "preds_sealed.npz")
    for k in pa.keys():
        assert np.allclose(pa[k], pb[k], atol=1e-4), k


def test_runner_cli_and_restore(data, tmp_path):
    from fl_skin import restore, run
    plan = [_cfg("cli_cl", "cl", fusion="image", epochs=2), _cfg("cli_fl", "fl", rounds=2)]
    pp = tmp_path / "plan.json"
    save_plan(plan, str(pp))
    out = tmp_path / "out"
    common = ["--plan", str(pp), "--data", data["root"], "--splits", data["sp_path"], "--out", str(out),
              "--img-size", "64", "--crop", "56", "--cache-dir", str(tmp_path), "--device", "cpu"]
    run.main(common + ["--nshards", "2", "--shard", "0"])
    run.main(common + ["--nshards", "2", "--shard", "1"])
    assert (out / "runs" / "cli_cl" / "done.json").exists()
    assert (out / "runs" / "cli_fl" / "done.json").exists()
    # restore into a fresh working dir from an "input" copy of the outputs
    fresh = tmp_path / "fresh"
    got = restore.restore(str(out), str(fresh))
    assert set(got) == {"cli_cl", "cli_fl"}
    run.main(["--plan", str(pp), "--data", data["root"], "--splits", data["sp_path"], "--out", str(fresh),
              "--img-size", "64", "--crop", "56", "--cache-dir", str(tmp_path), "--device", "cpu"])
    lines = (fresh / "results_shard0.csv").read_text()
    assert lines.count("already_done") == 2


# --------------------------------------------------------------- metrics ---

def test_metrics():
    from sklearn.metrics import balanced_accuracy_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 6, 300)
    p = rng.dirichlet(np.ones(6), 300)
    p[np.arange(300), y] += 0.4
    p /= p.sum(1, keepdims=True)
    assert np.isclose(balanced_accuracy(y, p), balanced_accuracy_score(y, p.argmax(1)))
    m = all_metrics(y, p, threshold=cancer_threshold(y, p, 0.9))
    assert m["cancer_sens"] >= 0.89 and 0 <= m["ece"] <= 1
    groups = np.repeat(np.arange(150), 2)
    pt, lo, hi = bootstrap_ci(y, p, groups, n_boot=300)
    assert lo <= pt <= hi
    d, lo, hi, pv = paired_bootstrap_diff(y, p, p, groups, n_boot=200)
    assert d == 0 and pv == 1.0
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == 0.03 and adj["c"] == 0.06 and adj["b"] == 0.06


def test_only_safe_fields_used(data):
    banned = ["smoke", "drink", "background", "pesticide", "gender", "cancer_history", "piped",
              "sewage", "fitspatrick", "diameter", "grew", "changed"]
    for n in data["ctx"].enc.names:
        assert not any(b in n for b in banned), n
