"""Patient-level splits and client partitions.

Every function works on patients, never on single images, so all photos of a
patient (and therefore of each lesion) stay together.

Output of `make_all_splits` (saved as CSV, one row per image):
    row, patient_id, label, split            split in {train, val, test}
    plus one column per regime with the client id (0..K-1, -1 for test rows)
    plus `meta_blank__missing` (1 = this image's metadata is hidden in that regime)
Train AND val patients are partitioned with the same rule, so a client's
validation set comes from its own kind of patients, and CL's train/val are
exactly the union of the clients' train/val.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .data import region_group

REGIMES = ["iid", "natural", "dir0.5", "dir0.1", "missing"]


def _patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient with its most frequent label (used to stratify)."""
    g = df.groupby("patient_id")
    return pd.DataFrame({
        "label": g["label"].agg(lambda s: s.value_counts().idxmax()),
        "n_images": g.size(),
    }).reset_index()


def stratified_group_holdout(df: pd.DataFrame, frac: float, seed: int) -> np.ndarray:
    """Boolean mask over df rows: a ~frac share of patients, balanced by class."""
    n_splits = max(2, int(round(1 / frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, hold_idx = next(sgkf.split(df, df["label"], groups=df["patient_id"]))
    mask = np.zeros(len(df), bool)
    mask[hold_idx] = True
    return mask


def global_split(df: pd.DataFrame, test_frac=0.20, val_frac=0.15, seed=2024) -> pd.Series:
    split = pd.Series("train", index=df.index)
    test = stratified_group_holdout(df, test_frac, seed)
    split[test] = "test"
    rest = df[~test]
    val_mask_rest = stratified_group_holdout(rest, val_frac, seed + 1)
    split.loc[rest.index[val_mask_rest]] = "val"
    return split


# ---------------------------------------------------------------- regimes ---

def partition_iid(pdf: pd.DataFrame, k: int, seed: int) -> dict:
    if len(pdf) < k:
        raise ValueError("fewer patients than clients")
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    out = {}
    # one row per patient, so groups = patient ids
    for c, (_, idx) in enumerate(sgkf.split(pdf, pdf["label"], groups=pdf["patient_id"])):
        for p in pdf["patient_id"].iloc[idx]:
            out[p] = c
    return out


def partition_dirichlet(pdf: pd.DataFrame, k: int, alpha: float, seed: int,
                        min_images: int, props: np.ndarray | None = None,
                        max_tries: int = 2000) -> tuple[dict, np.ndarray]:
    """Per class, share patients between clients with Dirichlet(alpha) proportions.

    Retries until every client has >= min_images images. Returns the mapping
    and the class x client proportion matrix (reused for the val patients so
    that each client's validation set has the same mix as its training set).
    """
    n_cls = int(pdf["label"].max()) + 1
    for t in range(max_tries):
        rng = np.random.default_rng(seed + t)
        P = props if props is not None else rng.dirichlet([alpha] * k, size=n_cls)
        out = {}
        for c in range(n_cls):
            pts = pdf[pdf["label"] == c].sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
            if len(pts) == 0:
                continue
            cuts = (np.cumsum(P[c]) * len(pts)).round().astype(int)[:-1]
            for client, chunk in enumerate(np.split(pts["patient_id"].to_numpy(), cuts)):
                for p in chunk:
                    out[p] = client
        sizes = np.zeros(k)
        for p, cl in out.items():
            sizes[cl] += pdf.set_index("patient_id").at[p, "n_images"]
        if props is not None or sizes.min() >= min_images:
            return out, P
    raise RuntimeError(f"Dirichlet alpha={alpha}: could not give every client {min_images}+ images")


def partition_natural(df_part: pd.DataFrame, groups: list[list[str]]) -> dict:
    """Clients by body-region group; a patient goes to the group of most of their images."""
    rg = pd.Series(region_group(df_part["region"]), index=df_part.index)
    out = {}
    for p, idx in df_part.groupby("patient_id").groups.items():
        g = rg.loc[idx].value_counts().idxmax()
        for c, members in enumerate(groups):
            if g in members:
                out[p] = c
                break
        else:
            out[p] = len(groups) - 1  # 'unknown' region -> last client
    return out


def make_all_splits(df: pd.DataFrame, k: int = 3, seed: int = 2024, min_images: int = 150,
                    natural_groups: list[list[str]] | None = None,
                    missing_frac: float = 0.30) -> pd.DataFrame:
    natural_groups = natural_groups or [["head_neck"], ["trunk"], ["limbs", "unknown"]]
    out = df[["row", "img_id", "patient_id", "label", "region"]].copy()
    out["split"] = global_split(df, seed=seed)
    tr = out[out["split"] == "train"]
    va = out[out["split"] == "val"]
    ptr, pva = _patient_table(tr), _patient_table(va)

    maps = {}
    maps["iid"] = {**partition_iid(ptr, k, seed), **partition_iid(pva, k, seed + 7)}
    maps["missing"] = maps["iid"]
    for a, name in [(0.5, "dir0.5"), (0.1, "dir0.1")]:
        m_tr, P = partition_dirichlet(ptr, k, a, seed, min_images)
        m_va, _ = partition_dirichlet(pva, k, a, seed + 11, 0, props=P)
        maps[name] = {**m_tr, **m_va}
    if k == len(natural_groups):
        maps["natural"] = partition_natural(out[out["split"] != "test"], natural_groups)
    else:  # body-region clients only exist for k == number of region groups
        maps["natural"] = {}

    for r in REGIMES:
        out[r] = out["patient_id"].map(maps[r]).fillna(-1).astype(int)
        out.loc[out["split"] == "test", r] = -1

    # missing-metadata regime: client K-1 has no metadata; others lose missing_frac of rows
    rng = np.random.default_rng(seed + 99)
    blank = np.zeros(len(out), bool)
    nontest = out["split"] != "test"
    blank |= (nontest & (out["missing"] == k - 1)).to_numpy()
    others = (nontest & (out["missing"] != k - 1)).to_numpy()
    blank |= others & (rng.random(len(out)) < missing_frac)
    out["meta_blank__missing"] = blank.astype(int)
    out = out.drop(columns=["region"])
    check_splits(out, k)
    return out


def check_splits(sp: pd.DataFrame, k: int) -> None:
    """Fail loudly on any leakage or empty client."""
    pid_split = sp.groupby("patient_id")["split"].nunique()
    assert (pid_split == 1).all(), "a patient appears in more than one of train/val/test"
    for r in REGIMES:
        nt = sp[sp["split"] != "test"]
        if (nt[r] == -1).all():
            continue  # regime not defined for this number of clients
        assert (nt[r] >= 0).all(), f"{r}: unassigned non-test rows"
        assert (nt.groupby("patient_id")[r].nunique() == 1).all(), f"{r}: patient in 2 clients"
        assert (sp.loc[sp["split"] == "test", r] == -1).all()
        for c in range(k):
            assert ((nt[r] == c) & (nt["split"] == "train")).any(), f"{r}: client {c} has no train"


def client_class_table(sp: pd.DataFrame, regime: str, classes: list[str]) -> pd.DataFrame:
    nt = sp[sp["split"] != "test"]
    t = pd.crosstab([nt[regime], nt["split"]], nt["label"]).reindex(columns=range(len(classes)), fill_value=0)
    t.columns = classes
    t["total"] = t.sum(1)
    return t
