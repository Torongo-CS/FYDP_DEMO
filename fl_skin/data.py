"""Loading, cleaning and encoding PAD-UFES-20.

Key rules (see the project doc):
- Columns that leak the label or identify rows never reach the model:
  `biopsed` (every cancer was biopsied), `diagnostic` (the label), ids.
- Every metadata field gets a 0/1 "was missing" flag.
- Encoding statistics are computed only from training rows. In FL they are
  built from per-client sums that are added up (only totals are shared,
  never rows), so every client uses the same input scale.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import CLASSES

LEAK_OR_ID_COLUMNS = ["biopsed", "diagnostic", "img_id", "lesion_id", "patient_id"]

# Fields of the patient questionnaire (smoke, drink, family background, pesticide,
# gender, cancer histories, water/sewage, Fitzpatrick, diameters) are missing for
# 804 images, and every one of those images is benign (ACK/NEV/SEK, never
# biopsied). "Questionnaire missing" would therefore reveal "not cancer", so the
# study uses only the fields recorded for every image (decision of 16 Nov 2026).
# `grew` and `changed` are also left out: their "UNK" answers are concentrated in
# BCC/SCC (a collection artefact, not a clinical sign).
FEATURE_SETS = {
    "safe": {
        "numeric": ["age"],
        "boolean": ["itch", "hurt", "bleed", "elevation"],
        "categorical": ["region"],
    },
    "full": {  # for reference only; leaks the label through missingness
        "numeric": ["age", "diameter_1", "diameter_2", "fitspatrick"],
        "boolean": ["smoke", "drink", "pesticide", "skin_cancer_history", "cancer_history",
                    "has_piped_water", "has_sewage_system", "itch", "grew", "hurt", "changed",
                    "bleed", "elevation"],
        "categorical": ["gender", "region", "background_father", "background_mother"],
    },
}
FEATURE_SET = "safe"
NUMERIC = FEATURE_SETS[FEATURE_SET]["numeric"]
BOOLEAN = FEATURE_SETS[FEATURE_SET]["boolean"]
# Category vocabularies are fixed up front (not learned from data), so every
# client and every split encodes them identically.
_VOCAB = {
    "gender": ["FEMALE", "MALE"],
    "region": [
        "ABDOMEN", "ARM", "BACK", "CHEST", "EAR", "FACE", "FOOT", "FOREARM",
        "HAND", "LIP", "NECK", "NOSE", "SCALP", "THIGH",
    ],
    "background_father": ["BRAZIL", "GERMANY", "ITALY", "POMERANIA", "PORTUGAL", "NETHERLANDS"],
    "background_mother": ["BRAZIL", "GERMANY", "ITALY", "POMERANIA", "PORTUGAL", "NETHERLANDS"],
}
CATEGORICAL = {k: _VOCAB[k] for k in FEATURE_SETS[FEATURE_SET]["categorical"]}

REGION_GROUPS = {
    "head_neck": ["FACE", "NOSE", "EAR", "LIP", "SCALP", "NECK"],
    "trunk": ["CHEST", "BACK", "ABDOMEN"],
    "limbs": ["ARM", "FOREARM", "HAND", "THIGH", "FOOT"],
}


def find_dataset_root(base: str) -> tuple[str, dict[str, str]]:
    """Find metadata.csv under `base` and index every image file by name."""
    meta_path, images = None, {}
    for root, _dirs, files in os.walk(base):
        for f in files:
            if f == "metadata.csv" and meta_path is None:
                meta_path = os.path.join(root, f)
            elif f.lower().endswith((".png", ".jpg", ".jpeg")):
                images.setdefault(f, os.path.join(root, f))
    if meta_path is None:
        raise FileNotFoundError(f"metadata.csv not found under {base}")
    return meta_path, images


def load_metadata(meta_path: str, image_index: dict[str, str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(meta_path)
    df["diagnostic"] = df["diagnostic"].astype(str).str.upper().str.strip()
    df = df[df["diagnostic"].isin(CLASSES)].copy()
    df["label"] = df["diagnostic"].map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    df["patient_id"] = df["patient_id"].astype(str)
    if image_index is not None:
        df["path"] = df["img_id"].map(image_index)
        missing = df["path"].isna().sum()
        if missing:
            raise FileNotFoundError(f"{missing} images listed in metadata were not found")
    df = df.reset_index(drop=True)
    df["row"] = np.arange(len(df))
    return df


def dataset_report(df: pd.DataFrame) -> dict:
    rep = {
        "images": int(len(df)),
        "patients": int(df["patient_id"].nunique()),
        "lesions": int(df["lesion_id"].nunique()) if "lesion_id" in df else None,
        "class_counts": df["diagnostic"].value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict(),
        "missing_fraction": {
            c: round(float(_missing_mask(df[c]).mean()), 3)
            for c in NUMERIC + BOOLEAN + list(CATEGORICAL) if c in df
        },
        "fitzpatrick_counts": df["fitspatrick"].value_counts(dropna=False).sort_index().to_dict()
        if "fitspatrick" in df else None,
        "region_counts": df["region"].value_counts(dropna=False).to_dict() if "region" in df else None,
    }
    return rep


def missingness_audit(df: pd.DataFrame, max_gap: float = 0.15) -> dict:
    """For every model input field, the missing rate per class.

    A field whose missing rate differs between classes by more than `max_gap`
    can leak the label through its "was missing" flag."""
    out = {}
    for c in NUMERIC + BOOLEAN + list(CATEGORICAL):
        rates = pd.Series(_missing_mask(df[c]), index=df.index).groupby(df["diagnostic"]).mean()
        out[c] = {"by_class": rates.round(3).to_dict(), "gap": round(float(rates.max() - rates.min()), 3),
                  "suspicious": bool(rates.max() - rates.min() > max_gap)}
    return out


def _missing_mask(s: pd.Series) -> np.ndarray:
    st = s.astype(str).str.strip().str.upper()
    return (s.isna() | st.isin(["", "NAN", "UNK", "UNKNOWN", "NONE"])).to_numpy()


def _bool_values(s: pd.Series) -> np.ndarray:
    st = s.astype(str).str.strip().str.upper()
    return st.isin(["TRUE", "1", "1.0", "YES"]).to_numpy().astype(np.float32)


@dataclass
class MetaStats:
    """Sufficient statistics for numeric columns: sums that can be added across clients."""
    n: dict = field(default_factory=dict)
    s: dict = field(default_factory=dict)
    ss: dict = field(default_factory=dict)

    @staticmethod
    def from_frame(df: pd.DataFrame) -> "MetaStats":
        st = MetaStats()
        for c in NUMERIC:
            m = ~_missing_mask(df[c])
            v = pd.to_numeric(df[c][m], errors="coerce").dropna().to_numpy(dtype=np.float64)
            st.n[c], st.s[c], st.ss[c] = float(len(v)), float(v.sum()), float((v ** 2).sum())
        return st

    def __add__(self, other: "MetaStats") -> "MetaStats":
        out = MetaStats()
        for c in NUMERIC:
            out.n[c] = self.n.get(c, 0.0) + other.n.get(c, 0.0)
            out.s[c] = self.s.get(c, 0.0) + other.s.get(c, 0.0)
            out.ss[c] = self.ss.get(c, 0.0) + other.ss.get(c, 0.0)
        return out

    def mean_std(self) -> dict:
        res = {}
        for c in NUMERIC:
            n = self.n.get(c, 0.0)
            if n < 2:
                res[c] = (0.0, 1.0)
                continue
            mu = self.s[c] / n
            var = max(self.ss[c] / n - mu ** 2, 1e-8)
            res[c] = (mu, float(np.sqrt(var)))
        return res


class MetaEncoder:
    """Turns the metadata table into a float matrix.

    Layout: [numeric z-scores | numeric missing flags | boolean values |
    boolean missing flags | one-hot categories incl. an UNK slot].
    Missing numerics are filled with the training mean (0 after scaling).
    """

    def __init__(self, stats: MetaStats):
        self.ms = stats.mean_std()
        self.names = (
            NUMERIC + [f"{c}__missing" for c in NUMERIC] + BOOLEAN
            + [f"{c}__missing" for c in BOOLEAN]
            + [f"{k}={v}" for k, vs in CATEGORICAL.items() for v in vs + ["UNK"]]
        )
        assert not any(n.split("=")[0].split("__")[0] in LEAK_OR_ID_COLUMNS for n in self.names)

    @property
    def dim(self) -> int:
        return len(self.names)

    def transform(self, df: pd.DataFrame, blank: np.ndarray | None = None) -> np.ndarray:
        """`blank`: boolean per row; True = treat all metadata of that row as missing."""
        n = len(df)
        blank = np.zeros(n, bool) if blank is None else np.asarray(blank, bool)
        cols = []
        num_vals, num_miss = [], []
        for c in NUMERIC:
            miss = _missing_mask(df[c]) | blank
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
            miss |= np.isnan(v)
            mu, sd = self.ms[c]
            z = np.where(miss, 0.0, (v - mu) / sd)
            num_vals.append(np.clip(z, -5, 5))
            num_miss.append(miss.astype(np.float64))
        b_vals, b_miss = [], []
        for c in BOOLEAN:
            miss = _missing_mask(df[c]) | blank
            b_vals.append(np.where(miss, 0.0, _bool_values(df[c])))
            b_miss.append(miss.astype(np.float64))
        cats = []
        for k, vocab in CATEGORICAL.items():
            st = df[k].astype(str).str.strip().str.upper().to_numpy()
            miss = _missing_mask(df[k]) | blank
            oh = np.zeros((n, len(vocab) + 1))
            for j, v in enumerate(vocab):
                oh[:, j] = (st == v) & ~miss
            oh[:, -1] = oh[:, :-1].sum(1) == 0
            cats.append(oh)
        cols = num_vals + num_miss + b_vals + b_miss
        X = np.concatenate([np.stack(cols, 1), *cats], 1).astype(np.float32)
        assert X.shape[1] == self.dim
        return X


def region_group(region: pd.Series) -> np.ndarray:
    st = region.astype(str).str.strip().str.upper()
    out = np.full(len(st), "unknown", dtype=object)
    for g, regs in REGION_GROUPS.items():
        out[st.isin(regs).to_numpy()] = g
    return out


# ---------------------------------------------------------------- images ---

def load_image_square(path: str, size: int) -> np.ndarray:
    """Pad to square with the image's mean colour, then resize. Returns HxWx3 uint8."""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w, h = im.size
    side = max(w, h)
    mean = tuple(int(x) for x in np.asarray(im).reshape(-1, 3).mean(0))
    canvas = Image.new("RGB", (side, side), mean)
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    return np.asarray(canvas.resize((size, size), Image.BICUBIC), dtype=np.uint8)


def build_image_cache(df: pd.DataFrame, out_path: str, size: int = 256, workers: int = 4) -> str:
    """Decode every image once into one uint8 array (rows follow df['row'])."""
    if os.path.exists(out_path):
        arr = np.load(out_path, mmap_mode="r")
        if arr.shape[0] == len(df) and arr.shape[1] == size:
            return out_path
    from concurrent.futures import ThreadPoolExecutor

    paths = df.sort_values("row")["path"].tolist()
    arr = np.zeros((len(paths), size, size, 3), np.uint8)
    with ThreadPoolExecutor(workers) as ex:
        for i, img in enumerate(ex.map(lambda p: load_image_square(p, size), paths)):
            arr[i] = img
    tmp = out_path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, out_path)
    return out_path
