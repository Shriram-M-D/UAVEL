"""F1: logical UAV client simulator.

Each client is constructed with ONLY its own rows and pixels, so it cannot read another client's data.
Per round it: loads the global shared encoder, trains locally with Shreyas's masked-reconstruction loop,
records a ledger 'train' event keyed by every (zone, time_bucket) it trained on, and returns an update
containing only shared parameters. Decoder and zone adapters never leave the client.
"""
from __future__ import annotations

import copy
from typing import Optional

import torch

from . import _paths  # noqa: F401
from src.learning.local_ssl import train_local  # Shreyas
from uav_backend import Event, Key  # Shriram


def is_shared(name: str, prefixes) -> bool:
    return name.startswith(tuple(prefixes))


def shared_state(model: torch.nn.Module, prefixes) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if is_shared(k, prefixes)}


class UAVClient:
    def __init__(self, client_id: str, rows: list[dict], x: torch.Tensor, model: torch.nn.Module,
                 prefixes, backend=None):
        assert all(r["client_id"] == client_id for r in rows), "client given another client's rows"
        assert len(rows) == len(x)
        self.client_id, self.rows, self.x, self.prefixes, self.b = client_id, rows, x, tuple(prefixes), backend
        self.model = copy.deepcopy(model)  # local copy: decoder + adapters persist on the UAV

    @property
    def sample_ids(self) -> set[str]:
        return {r["sample_id"] for r in self.rows}

    def units(self) -> list[tuple[int, int]]:
        return sorted({(int(r["zone_id"]), int(r["time_bucket"])) for r in self.rows})

    def local_round(self, global_shared: dict, round_id: str, model_version: str, epochs: int = 1,
                    seed: int = 0) -> dict:
        self.model.load_state_dict(global_shared, strict=False)  # only shared keys are overwritten
        _, info = train_local(self.x, epochs=epochs, model=self.model, seed=seed, log=lambda *_: None)
        if self.b is not None:
            self.b.record(Event("train", keys=[Key(self.client_id, z, t) for z, t in self.units()],
                                client_id=self.client_id, model_version=model_version, parent_version=model_version,
                                touched_shared_encoder=True,
                                payload={"round_id": round_id, "epochs": epochs, "n_samples": len(self.x),
                                         "final_recon_loss": info["loss_history"][-1]}))
        return {"client_id": self.client_id, "round_id": round_id, "model_version": model_version,
                "num_samples": len(self.x),
                "zone_ids": sorted({z for z, _ in self.units()}),
                "time_buckets": sorted({t for _, t in self.units()}),
                "state_dict": shared_state(self.model, self.prefixes),
                "metrics": {"recon_loss": float(info["loss_history"][-1]),
                            "train_seconds": float(info["train_seconds"])}}


def build_clients(manifest_rows: list[dict], x_all: torch.Tensor, model, prefixes, backend=None,
                  split: str = "ssl_train", client_ids: Optional[list[str]] = None) -> dict[str, UAVClient]:
    """Partition the manifest by client; each client gets a copy of only its own pixels."""
    out = {}
    ids = client_ids or sorted({r["client_id"] for r in manifest_rows})
    for cid in ids:
        idx = [i for i, r in enumerate(manifest_rows) if r["client_id"] == cid and r["split"] == split]
        out[cid] = UAVClient(cid, [manifest_rows[i] for i in idx], x_all[idx].clone(), model, prefixes, backend)
    return out
