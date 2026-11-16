"""One RunConfig per experiment. Plans are JSON lists of these (see make_plan.py)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields


@dataclass
class RunConfig:
    run_id: str
    kind: str                 # "meta_lr" | "cl" | "local" | "fl" | "calib"
    fusion: str = "concat"    # image | concat | gated | xattn
    regime: str = "iid"       # iid | natural | dir0.5 | dir0.1 | missing (cl ignores it)
    algo: str = "fedavg"      # fedavg | fedprox | fednova | scaffold | fedbn
    seed: int = 0
    n_clients: int = 3
    # optimisation (same optimizer for CL and FL: SGD + momentum)
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    batch_size: int = 32
    amp: bool = True
    # CL / local-only
    epochs: int = 50
    warmup: int = 2
    patience: int = 10
    # FL
    rounds: int = 30
    local_epochs: int = 2
    mu: float = 0.01          # FedProx strength
    finetune_epochs: int = 1  # FedAvg + fine-tune personalisation
    finetune_lr: float = 0.001
    # images
    img_size: int = 256
    crop: int = 224
    pretrained: bool = True
    # calibration only
    overfit_n: int = 64

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "RunConfig":
        names = {f.name for f in fields(RunConfig)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        return RunConfig(**d)


def load_plan(path: str) -> list[RunConfig]:
    with open(path) as f:
        return [RunConfig.from_dict(d) for d in json.load(f)]


def save_plan(runs: list[RunConfig], path: str) -> None:
    ids = [r.run_id for r in runs]
    assert len(ids) == len(set(ids)), "duplicate run ids"
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in runs], f, indent=1)
