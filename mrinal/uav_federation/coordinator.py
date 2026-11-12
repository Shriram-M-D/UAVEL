"""F2/F6: FedAvg coordinator over shared parameters only, with round history written to the ledger."""
from __future__ import annotations

from pathlib import Path

import torch

from . import _paths  # noqa: F401
from uav_backend import Artefact, Key  # Shriram


def fedavg(updates: list[dict]) -> dict[str, torch.Tensor]:
    """Sample-weighted average of the shared state_dicts. All updates must carry the same keys."""
    if not updates:
        raise ValueError("no updates to aggregate")
    keys = set(updates[0]["state_dict"])
    for u in updates:
        if set(u["state_dict"]) != keys:
            raise ValueError(f"client {u['client_id']} sent a different parameter set")
    n = sum(u["num_samples"] for u in updates)
    return {k: sum(u["state_dict"][k].float() * (u["num_samples"] / n) for u in updates) for k in keys}


class Coordinator:
    def __init__(self, backend, initial_shared: dict[str, torch.Tensor], version: str = "enc-v1"):
        self.b = backend
        self.shared = {k: v.clone() for k, v in initial_shared.items()}
        self.version = version
        self.dir = Path(backend.artefact_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        if backend.registry.get(version) is None:  # register the pre-trained starting point
            p = self.dir / f"shared_encoder_{version}.pt"
            torch.save(self.shared, p)
            backend.registry.register(Artefact(version, "shared_encoder", version=version, path=str(p),
                                               meta={"note": "initialisation; no project data"}))

    def receive(self, update: dict) -> int:
        """Every update passes Shriram's locality guard before it is accepted. Returns bytes."""
        return self.b.guard.check_update(update)

    def aggregate(self, round_id: str, updates: list[dict], contributions: list[Key], extra: dict) -> str:
        self.shared = fedavg(updates)
        n = len(self.b.registry.list(kind="shared_encoder")) + 1
        new_v = f"enc-v{n}"
        p = self.dir / f"shared_encoder_{new_v}.pt"
        torch.save(self.shared, p)
        new = Artefact(new_v, "shared_encoder", version=new_v, path=str(p), parents=[self.version],
                       meta={"round_id": round_id, "aggregation": "fedavg"})
        self.b.record_aggregate(round_id, contributions, new, parent_version=self.version, payload=extra)
        self.b.registry.set_status([self.version], "superseded")
        self.version = new_v
        return new_v
