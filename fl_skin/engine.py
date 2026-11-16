"""Runs one experiment (CL, local-only, FL, metadata-only, calibration) with
checkpoint-every-epoch/round, exact resume and a time guard.

Output of a finished run in <out>/runs/<run_id>/:
    log.csv            one line per epoch/round (validation only, never test)
    best.pt            best model(s) by validation
    preds_val.npz      validation probabilities (used for thresholds)
    preds_sealed.npz   test probabilities - not read until the final analysis
    done.json          summary (best score, epochs/rounds used, minutes, MB uploaded)
"""
from __future__ import annotations

import copy
import csv
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from . import CLASSES
from .config import RunConfig
from .data import MetaEncoder, MetaStats
from .models import SkinNet, bn_keys
from .train import (Bank, balanced_accuracy, class_weights, cosine_lr, make_optimizer,
                    predict, train_epochs)

N_CLS = len(CLASSES)


class Paused(Exception):
    """Raised when the session is about to run out of time; state is already saved."""


class Budget:
    def __init__(self, deadline: float | None):
        self.deadline = deadline

    def check(self, need_s: float):
        if self.deadline is not None and time.time() + need_s > self.deadline:
            raise Paused()


def _save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _append_log(path, row: dict):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def _trim_log(path, keep_until: int, key: str):
    """After a resume, drop log lines written after the last saved checkpoint."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        df[df[key] <= keep_until].to_csv(path, index=False)


# ------------------------------------------------------------------ context ---

class Context:
    """Everything shared by all runs of one session: data, splits, image bank."""

    def __init__(self, df: pd.DataFrame, splits: pd.DataFrame, images: np.ndarray, device: str,
                 crop: int = 224, weights_path: str | None = None):
        assert (splits["row"].to_numpy() == df["row"].to_numpy()).all(), "splits/metadata row mismatch"
        self.df, self.sp = df, splits
        self.labels = df["label"].to_numpy()
        self.groups = df["patient_id"].to_numpy()
        self.device = device
        self.weights_path = weights_path
        split = splits["split"].to_numpy()
        self.train = np.where(split == "train")[0]
        self.val = np.where(split == "val")[0]
        self.test = np.where(split == "test")[0]
        stats = MetaStats.from_frame(df.iloc[self.train])
        self.enc = MetaEncoder(stats)
        blank_all = np.ones(len(df), bool)
        self.meta = {
            "full": self.enc.transform(df),
            "blank": self.enc.transform(df, blank=blank_all),
        }
        # missing regime: statistics from the rows whose metadata is visible
        bm = splits["meta_blank__missing"].to_numpy().astype(bool)
        vis_train = self.train[~bm[self.train]]
        self.enc_missing = MetaEncoder(MetaStats.from_frame(df.iloc[vis_train]))
        self.meta["missing"] = self.enc_missing.transform(df, blank=bm)
        self.meta["missing_full"] = self.enc_missing.transform(df)
        self.meta["missing_blank"] = self.enc_missing.transform(df, blank=blank_all)
        self.bank = Bank(images, self.meta["full"], self.labels, device, crop)

    def bank_for(self, regime: str, variant: str = "train") -> Bank:
        if regime == "missing":
            key = {"train": "missing", "full": "missing_full", "blank": "missing_blank"}[variant]
        else:
            key = {"train": "full", "full": "full", "blank": "blank"}[variant]
        return self.bank.with_meta(self.meta[key])

    def clients(self, regime: str, k: int):
        col = self.sp[regime].to_numpy()
        return [(self.train[col[self.train] == c], self.val[col[self.val] == c]) for c in range(k)]

    def build_model(self, cfg: RunConfig) -> SkinNet:
        torch.manual_seed(cfg.seed)
        m = SkinNet(cfg.fusion, self.enc.dim, N_CLS, pretrained=cfg.pretrained,
                    weights_path=self.weights_path)
        return m.to(self.device)


def _use_meta(cfg):
    return cfg.fusion != "image"


def _test_preds(model, ctx: Context, cfg: RunConfig, regime: str):
    um = _use_meta(cfg)
    out = {"test_full": predict(model, ctx.bank_for(regime, "full"), ctx.test, use_meta=um)}
    if um:
        out["test_blank"] = predict(model, ctx.bank_for(regime, "blank"), ctx.test, use_meta=um)
    return out


# --------------------------------------------------------------- supervised ---

def train_supervised(cfg: RunConfig, ctx: Context, bank: Bank, tr: np.ndarray, va: np.ndarray,
                     run_dir: str, budget: Budget, key: int = 0):
    """CL recipe: persistent SGD, warmup + cosine, early stopping on validation balanced accuracy."""
    os.makedirs(run_dir, exist_ok=True)
    state_p, best_p, log_p = (os.path.join(run_dir, f) for f in ("state.pt", "best.pt", "log.csv"))
    model = ctx.build_model(cfg)
    opt = make_optimizer(model, cfg.lr, cfg)
    cw = class_weights(ctx.labels[tr], N_CLS)
    um = _use_meta(cfg)
    ep0, best, bad, avg_t = 0, -1.0, 0, None
    if os.path.exists(state_p):
        st = torch.load(state_p, map_location=ctx.device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        ep0, best, bad, avg_t = st["epoch"] + 1, st["best"], st["bad"], st["avg_t"]
        _trim_log(log_p, st["epoch"], "epoch")
    for ep in range(ep0, cfg.epochs):
        if bad >= cfg.patience:
            break
        budget.check((avg_t or 0) * 1.5 + 120)
        t0 = time.time()
        lr = cosine_lr(cfg.lr, ep, cfg.epochs, cfg.warmup)
        train_epochs(model, bank, tr, 1, lr, cfg, cw, (cfg.seed, key, ep), use_meta=um, opt=opt)
        pv = predict(model, bank, va, use_meta=um)
        score = balanced_accuracy(ctx.labels[va], pv)
        dt = time.time() - t0
        avg_t = dt if avg_t is None else 0.7 * avg_t + 0.3 * dt
        if score > best:
            best, bad = score, 0
            _save({"model": model.state_dict(), "epoch": ep, "score": score}, best_p)
        else:
            bad += 1
        _append_log(log_p, {"epoch": ep, "lr": round(lr, 6), "val_bacc": round(score, 5),
                            "seconds": round(dt, 1)})
        _save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "best": best,
               "bad": bad, "avg_t": avg_t}, state_p)
    bst = torch.load(best_p, map_location=ctx.device, weights_only=False)
    model.load_state_dict(bst["model"])
    return model, {"best_val_bacc": bst["score"], "best_epoch": bst["epoch"]}


def run_cl(cfg, ctx: Context, run_dir, budget):
    bank = ctx.bank_for("iid")
    model, info = train_supervised(cfg, ctx, bank, ctx.train, ctx.val, run_dir, budget)
    um = _use_meta(cfg)
    np.savez_compressed(os.path.join(run_dir, "preds_val.npz"), rows=ctx.val,
                        probs=predict(model, bank, ctx.val, use_meta=um))
    np.savez_compressed(os.path.join(run_dir, "preds_sealed.npz"), rows=ctx.test,
                        **_test_preds(model, ctx, cfg, "iid"))
    return info


def run_local(cfg, ctx: Context, run_dir, budget):
    """Each client trains alone with the CL recipe (the floor for Matrix B)."""
    bank = ctx.bank_for(cfg.regime)
    info, val_out, test_out = {}, {}, {}
    for k, (tr, va) in enumerate(ctx.clients(cfg.regime, cfg.n_clients)):
        model, ik = train_supervised(cfg, ctx, bank, tr, va, os.path.join(run_dir, f"client{k}"),
                                     budget, key=100 + k)
        info[f"client{k}_best_val_bacc"] = ik["best_val_bacc"]
        um = _use_meta(cfg)
        val_out[f"client{k}_rows"] = va
        val_out[f"client{k}_probs"] = predict(model, bank, va, use_meta=um)
        for name, p in _test_preds(model, ctx, cfg, cfg.regime).items():
            test_out[f"client{k}_{name}"] = p
    info["best_val_bacc"] = float(np.mean([v for kk, v in info.items() if kk.endswith("val_bacc")]))
    np.savez_compressed(os.path.join(run_dir, "preds_val.npz"), **val_out)
    np.savez_compressed(os.path.join(run_dir, "preds_sealed.npz"), rows=ctx.test, **test_out)
    return info


def run_meta_lr(cfg, ctx: Context, run_dir, budget):
    from sklearn.linear_model import LogisticRegression

    os.makedirs(run_dir, exist_ok=True)
    X = ctx.meta["full"]
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0, random_state=cfg.seed)
    clf.fit(X[ctx.train], ctx.labels[ctx.train])

    def full_probs(rows, Xs):
        p = np.zeros((len(rows), N_CLS))
        p[:, clf.classes_] = clf.predict_proba(Xs[rows])
        return p

    pv = full_probs(ctx.val, X)
    np.savez_compressed(os.path.join(run_dir, "preds_val.npz"), rows=ctx.val, probs=pv)
    np.savez_compressed(os.path.join(run_dir, "preds_sealed.npz"), rows=ctx.test,
                        test_full=full_probs(ctx.test, X))
    return {"best_val_bacc": balanced_accuracy(ctx.labels[ctx.val], pv)}


# ----------------------------------------------------------------------- FL ---

def nova_a(tau: int, rho: float) -> float:
    """FedNova's effective number of steps for SGD with momentum rho."""
    if rho == 0:
        return float(tau)
    return (tau - rho * (1 - rho ** tau) / (1 - rho)) / (1 - rho)


def aggregate(sds, weights, skip: set, algo: str, global_sd, taus=None, rho=0.0, trainable=None):
    new = {k: v.clone() for k, v in global_sd.items()}
    w = torch.tensor(weights, dtype=torch.float64)
    for k in global_sd:
        if k in skip:
            continue
        if not global_sd[k].is_floating_point():
            new[k] = torch.stack([sd[k] for sd in sds]).max(0).values
            continue
        if algo == "fednova" and k in trainable:
            a = [nova_a(t, rho) for t in taus]
            tau_eff = float(sum(wi * ai for wi, ai in zip(weights, a)))
            d = sum(wi * (global_sd[k].double() - sd[k].double()) / ai
                    for wi, ai, sd in zip(weights, a, sds))
            new[k] = (global_sd[k].double() - tau_eff * d).to(global_sd[k].dtype)
        else:
            new[k] = sum(float(wi) * sd[k].double() for wi, sd in zip(w, sds)).to(global_sd[k].dtype)
    return new


def run_fl(cfg: RunConfig, ctx: Context, run_dir, budget):
    os.makedirs(run_dir, exist_ok=True)
    state_p, best_p, log_p = (os.path.join(run_dir, f) for f in ("state.pt", "best.pt", "log.csv"))
    dev = ctx.device
    bank = ctx.bank_for(cfg.regime)
    clients = ctx.clients(cfg.regime, cfg.n_clients)
    n = np.array([len(tr) for tr, _ in clients], float)
    p = (n / n.sum()).tolist()
    cws = [class_weights(ctx.labels[tr], N_CLS) for tr, _ in clients]
    um = _use_meta(cfg)

    model = ctx.build_model(cfg)
    trainable = {k for k, _ in model.named_parameters()}
    bnk = bn_keys(model) if cfg.algo == "fedbn" else set()
    g = {k: v.detach().clone() for k, v in model.state_dict().items()}
    client_bn = [{k: g[k].clone() for k in bnk} for _ in clients]
    use_sc = cfg.algo == "scaffold"
    c = {k: torch.zeros_like(g[k]) for k in trainable} if use_sc else None
    ci = [{k: torch.zeros_like(g[k]) for k in trainable} for _ in clients] if use_sc else None
    shared = [k for k in g if k not in bnk and g[k].is_floating_point()]
    up_bytes = sum(g[k].numel() * 4 for k in shared) + (
        sum(g[k].numel() * 4 for k in trainable) if use_sc else 0)

    r0, best, avg_t, comm = 0, -1.0, None, 0.0
    if os.path.exists(state_p):
        st = torch.load(state_p, map_location=dev, weights_only=False)
        g, client_bn, c, ci = st["g"], st["client_bn"], st["c"], st["ci"]
        r0, best, avg_t, comm = st["round"] + 1, st["best"], st["avg_t"], st["comm"]
        _trim_log(log_p, st["round"], "round")

    def load_client(k):
        sd = dict(g)
        sd.update(client_bn[k])
        model.load_state_dict(sd)

    for r in range(r0, cfg.rounds):
        budget.check((avg_t or 0) * 1.5 + 120)
        t0 = time.time()
        lr = cosine_lr(cfg.lr, r, cfg.rounds)
        sds, taus = [], []
        for k, (tr, _va) in enumerate(clients):
            load_client(k)
            prox = ({kk: vv.detach().clone() for kk, vv in model.named_parameters()}
                    if cfg.algo == "fedprox" else None)
            steps = train_epochs(model, bank, tr, cfg.local_epochs, lr, cfg, cws[k], (cfg.seed, r, k),
                                 prox_ref=prox, mu=cfg.mu, scaffold_c=c,
                                 scaffold_ci=ci[k] if use_sc else None, use_meta=um)
            sd = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
            if use_sc:
                a = nova_a(steps, cfg.momentum) * lr
                for kk in trainable:
                    new_ci = ci[k][kk] - c[kk] + (g[kk] - sd[kk]) / a
                    sd["__dc__" + kk] = new_ci - ci[k][kk]
                    ci[k][kk] = new_ci
            if bnk:
                client_bn[k] = {kk: sd[kk].clone() for kk in bnk}
            sds.append(sd)
            taus.append(steps)
        if use_sc:
            for kk in trainable:
                c[kk] = c[kk] + sum(sd["__dc__" + kk] for sd in sds) / len(sds)
            sds = [{kk: vv for kk, vv in sd.items() if not kk.startswith("__dc__")} for sd in sds]
        g = aggregate(sds, p, bnk, cfg.algo, g, taus, cfg.momentum, trainable)
        comm += up_bytes * len(clients) / 1e6

        scores = []
        for k, (_tr, va) in enumerate(clients):
            load_client(k)
            scores.append(balanced_accuracy(ctx.labels[va], predict(model, bank, va, use_meta=um)))
        score = float(np.mean(scores))
        dt = time.time() - t0
        avg_t = dt if avg_t is None else 0.7 * avg_t + 0.3 * dt
        if score > best:
            best = score
            _save({"g": g, "client_bn": client_bn, "round": r, "score": score}, best_p)
        row = {"round": r, "lr": round(lr, 6), "val_bacc_mean": round(score, 5),
               **{f"val_bacc_c{k}": round(s, 5) for k, s in enumerate(scores)},
               "mb_uploaded": round(comm, 1), "seconds": round(dt, 1)}
        _append_log(log_p, row)
        _save({"g": g, "client_bn": client_bn, "c": c, "ci": ci, "round": r, "best": best,
               "avg_t": avg_t, "comm": comm}, state_p)

    # ---- final: best round -> predictions (+ personalised models)
    bst = torch.load(best_p, map_location=dev, weights_only=False)
    g, client_bn = bst["g"], bst["client_bn"]
    val_out, test_out = {}, {}
    if cfg.algo != "fedbn":  # one shared model -> Matrix A
        model.load_state_dict(g)
        val_out["rows"], val_out["probs"] = ctx.val, predict(model, bank, ctx.val, use_meta=um)
        test_out.update(_test_preds(model, ctx, cfg, cfg.regime))
    personal = {}
    if cfg.algo == "fedbn":
        for k in range(len(clients)):
            load_client(k)
            personal[k] = copy.deepcopy(model.state_dict())
    elif cfg.algo == "fedavg" and cfg.finetune_epochs > 0:
        budget.check(len(clients) * (avg_t or 0) + 120)
        for k, (tr, _va) in enumerate(clients):
            model.load_state_dict(g)
            train_epochs(model, bank, tr, cfg.finetune_epochs, cfg.finetune_lr, cfg, cws[k],
                         (cfg.seed, 9999, k), use_meta=um)
            personal[k] = copy.deepcopy(model.state_dict())
    info = {"best_val_bacc": bst["score"], "best_round": bst["round"], "mb_uploaded": round(comm, 1)}
    for k, sd in personal.items():
        model.load_state_dict(sd)
        va = clients[k][1]
        val_out[f"client{k}_rows"] = va
        val_out[f"client{k}_probs"] = predict(model, bank, va, use_meta=um)
        info[f"client{k}_personal_val_bacc"] = balanced_accuracy(ctx.labels[va], val_out[f"client{k}_probs"])
        for name, pr in _test_preds(model, ctx, cfg, cfg.regime).items():
            test_out[f"client{k}_{name}"] = pr
    if personal:
        _save({"personal": personal}, os.path.join(run_dir, "personal.pt"))
    np.savez_compressed(os.path.join(run_dir, "preds_val.npz"), **val_out)
    np.savez_compressed(os.path.join(run_dir, "preds_sealed.npz"), rows=ctx.test, **test_out)
    return info


# -------------------------------------------------------------- calibration ---

def run_calib(cfg: RunConfig, ctx: Context, run_dir, budget):
    """Time one CL epoch and one FL round; check the model can memorise 64 images."""
    os.makedirs(run_dir, exist_ok=True)
    um = _use_meta(cfg)
    bank = ctx.bank_for("iid")
    cw = class_weights(ctx.labels[ctx.train], N_CLS)
    model = ctx.build_model(cfg)
    sync = (lambda: torch.cuda.synchronize()) if ctx.device.startswith("cuda") else (lambda: None)
    # warm-up (first CUDA call is slow), then time a full epoch
    train_epochs(model, bank, ctx.train[:64], 1, cfg.lr, cfg, cw, (0, 0), use_meta=um)
    sync(); t0 = time.time()
    train_epochs(model, bank, ctx.train, 1, cfg.lr, cfg, cw, (0, 1), use_meta=um)
    sync(); epoch_s = time.time() - t0
    t0 = time.time(); predict(model, bank, ctx.val, use_meta=um); sync(); eval_s = time.time() - t0
    # memorisation check
    small = ctx.train[np.random.default_rng(0).permutation(len(ctx.train))[: cfg.overfit_n]]
    m2 = ctx.build_model(cfg)
    for ep in range(60):
        train_epochs(m2, bank, small, 1, 0.01, cfg, cw, (0, 2, ep), use_meta=um)
    acc = float((predict(m2, bank, small, use_meta=um).argmax(1) == ctx.labels[small]).mean())
    gpu = torch.cuda.get_device_name(0) if ctx.device.startswith("cuda") else "cpu"
    res = {"gpu": gpu, "fusion": cfg.fusion, "train_images": int(len(ctx.train)),
           "seconds_per_epoch": round(epoch_s, 1), "seconds_per_val_eval": round(eval_s, 1),
           "overfit_train_acc": round(acc, 3), "overfit_ok": acc >= 0.95,
           "est_minutes_cl_50ep": round(50 * (epoch_s + eval_s) / 60, 1),
           "est_minutes_fl_30x2": round(30 * (2 * epoch_s + 3 * eval_s / 3) / 60, 1)}
    with open(os.path.join(run_dir, "calib.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


RUNNERS = {"cl": run_cl, "local": run_local, "fl": run_fl, "meta_lr": run_meta_lr, "calib": run_calib}


def run_one(cfg: RunConfig, ctx: Context, out_dir: str, budget: Budget) -> dict:
    run_dir = os.path.join(out_dir, "runs", cfg.run_id)
    done_p = os.path.join(run_dir, "done.json")
    if os.path.exists(done_p):
        with open(done_p) as f:
            return {"status": "already_done", **json.load(f)}
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=1)
    t0 = time.time()
    info = RUNNERS[cfg.kind](cfg, ctx, run_dir, budget)
    info = {"run_id": cfg.run_id, "minutes_this_session": round((time.time() - t0) / 60, 1), **info}
    with open(done_p, "w") as f:
        json.dump(info, f, indent=1)
    # the resume state is no longer needed once a run is finished
    for root, _d, files in os.walk(run_dir):
        for fn in files:
            if fn == "state.pt":
                os.remove(os.path.join(root, fn))
    return {"status": "done", **info}
