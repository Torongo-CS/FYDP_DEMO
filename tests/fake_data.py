"""A small fake PAD-UFES-20 (same columns, same structure) for testing the code without the real data."""
import os

import numpy as np
import pandas as pd
from PIL import Image

CLASS_SHARE = {"ACK": 730, "BCC": 845, "MEL": 52, "NEV": 244, "SCC": 192, "SEK": 235}
COLOURS = {"ACK": (200, 80, 80), "BCC": (80, 200, 80), "MEL": (40, 40, 40),
           "NEV": (150, 100, 50), "SCC": (80, 80, 200), "SEK": (200, 200, 80)}
REGIONS = ["FACE", "NOSE", "BACK", "CHEST", "ARM", "FOREARM", "HAND", "THIGH", "NECK", "EAR"]


def make_fake(root: str, n_images: int = 420, size: int = 80, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, "images", "imgs_part_1"), exist_ok=True)
    classes = list(CLASS_SHARE)
    probs = np.array(list(CLASS_SHARE.values()), float)
    probs = probs ** 0.6
    probs /= probs.sum()
    rows, pid, lid, i = [], 0, 0, 0
    while i < n_images:
        pid += 1
        for _ in range(rng.integers(1, 3)):          # lesions per patient
            lid += 1
            dx = classes[rng.choice(len(classes), p=probs)]
            for _ in range(rng.integers(1, 3)):      # images per lesion
                if i >= n_images:
                    break
                img_id = f"PAT_{pid}_{lid}_{i}.png"
                col = np.array(COLOURS[dx]) + rng.normal(0, 25, 3)
                arr = np.clip(rng.normal(col, 30, (size, size + rng.integers(-10, 10), 3)), 0, 255)
                Image.fromarray(arr.astype(np.uint8)).save(os.path.join(root, "images", "imgs_part_1", img_id))
                full = rng.random() < 0.65                # ~35% of rows lack the clinical form
                b = lambda: rng.choice(["True", "False"]) if full else np.nan  # noqa: E731
                rows.append({
                    "patient_id": f"PAT_{pid}", "lesion_id": lid, "smoke": b(), "drink": b(),
                    "background_father": rng.choice(["POMERANIA", "GERMANY", "ITALY", "BRAZIL", "SPAIN"]) if full else np.nan,
                    "background_mother": rng.choice(["POMERANIA", "GERMANY", "ITALY", "BRAZIL"]) if full else np.nan,
                    "age": int(rng.integers(20, 90)), "pesticide": b(),
                    "gender": rng.choice(["MALE", "FEMALE"]) if full else np.nan,
                    "skin_cancer_history": b(), "cancer_history": b(), "has_piped_water": b(),
                    "has_sewage_system": b(), "fitspatrick": float(rng.integers(1, 5)) if full else np.nan,
                    "region": rng.choice(REGIONS), "diameter_1": float(rng.integers(3, 20)) if full else np.nan,
                    "diameter_2": float(rng.integers(3, 20)) if full else np.nan, "diagnostic": dx,
                    "itch": rng.choice(["True", "False", "UNK"]), "grew": rng.choice(["True", "False", "UNK"]),
                    "hurt": rng.choice(["True", "False"]), "changed": rng.choice(["True", "False", "UNK"]),
                    "bleed": rng.choice(["True", "False"]), "elevation": rng.choice(["True", "False"]),
                    "img_id": img_id, "biopsed": dx in ("BCC", "MEL", "SCC"),
                })
                i += 1
    pd.DataFrame(rows).to_csv(os.path.join(root, "metadata.csv"), index=False)
    return root
