"""F2-F6: FederationRunner - the plug-in Shriram's backend calls (POST /federation/rounds).

One round: pick participating clients -> simulate dropouts/delays (F4) -> local training -> every update
through the locality guard (F5) -> FedAvg of shared params (F2) -> 'aggregate' ledger event whose keys are
exactly the (client, zone, time) units that influenced the new encoder (F6).

Delayed clients: their update arrives one round late and is aggregated as a stale update (logged),
unless the same client also sent a fresh update that round - then the stale one is dropped (no double weight).
"""
from __future__ import annotations

import csv
from typing import Optional

import numpy as np
import torch

from . import _paths  # noqa: F401
from .clients import UAVClient, build_clients, shared_state
from .coordinator import Coordinator
from uav_backend import Key  # Shriram


class FederationRunner:
    def __init__(self, backend, clients: Optional[dict[str, UAVClient]] = None, model=None,
                 local_epochs: int = 1, dropout_prob: Optional[float] = None, delay_prob: float = 0.0,
                 seed: Optional[int] = None):
        self.b = backend
        cfg = backend.cfg
        self.prefixes = cfg.federation.shared_param_prefixes
        self.local_epochs = local_epochs
        self.dropout_prob = cfg.federation.dropout_prob if dropout_prob is None else dropout_prob
        self.delay_prob = delay_prob
        self.rng = np.random.default_rng(cfg.seed if seed is None else seed)
        if clients is None:
            model, clients = self._from_manifest()
        self.model = model
        self.clients = clients
        self.coord = Coordinator(backend, shared_state(model, self.prefixes))
        self.pending: list[tuple[dict, list[Key]]] = []  # delayed updates arriving next round
        self.history: list[dict] = []

    def _from_manifest(self):
        from src.data.dataset import load_images, to_tensor  # Shreyas
        from src.models.mae import MAE
        import pandas as pd
        path = self.b.cfg.path(self.b.cfg.dataset.manifest_path)
        df = pd.read_csv(path)
        rows = df.to_dict("records")
        x = to_tensor(load_images(df))
        torch.manual_seed(self.b.cfg.seed)
        model = MAE()
        return model, build_clients(rows, x, model, self.prefixes, self.b)

    def run_round(self, client_ids: Optional[list[str]] = None) -> dict:
        rid = f"round-{len(self.history) + 1}"
        ids = client_ids or sorted(self.clients)
        dropped, delayed, updates, contributions, bytes_up = [], [], [], [], 0
        # stale updates from last round arrive first (bytes count); used only if the client has nothing fresher
        arrived = self.pending
        self.pending = []
        for upd, _ in arrived:
            bytes_up += self.coord.receive(upd)
        for i, cid in enumerate(ids):
            if self.rng.random() < self.dropout_prob:
                dropped.append(cid)  # no connectivity this round: no training, no update
                continue
            c = self.clients[cid]
            upd = c.local_round(self.coord.shared, rid, self.coord.version, self.local_epochs,
                                seed=len(self.history) * 1000 + i)
            keys = [Key(cid, z, t) for z, t in c.units()]
            if self.rng.random() < self.delay_prob:
                delayed.append(cid)
                self.pending.append((upd, keys))
                continue
            bytes_up += self.coord.receive(upd)
            updates.append(upd)
            contributions += keys
        fresh = {u["client_id"] for u in updates}
        stale = [(u, k) for u, k in arrived if u["client_id"] not in fresh]
        superseded = [u["client_id"] for u, _ in arrived if u["client_id"] in fresh]
        for upd, keys in stale:  # one update per client per round, never double-weighted
            updates.append(upd)
            contributions += keys
        v_in = self.coord.version
        if updates:
            n_param_bytes = sum(t.element_size() * t.numel() for t in self.coord.shared.values())
            extra = {"clients": [u["client_id"] for u in updates], "dropped": dropped, "delayed": delayed,
                     "stale_from_previous_round": [u["client_id"] for u, _ in stale],
                     "stale_superseded": superseded, "bytes_up": bytes_up}
            v_out = self.coord.aggregate(rid, updates, contributions, extra)
            bytes_down = n_param_bytes * len(updates)
        else:
            v_out, bytes_down = v_in, 0  # nobody reachable: global model unchanged
        h = {"round_id": rid, "model_version_in": v_in, "model_version_out": v_out,
             "contributions": [{"client_id": k.client_id, "zone_id": k.zone_id, "time_bucket": k.time_bucket}
                               for k in contributions],
             "participants": [u["client_id"] for u in updates], "dropped_clients": dropped,
             "delayed_clients": delayed, "stale_updates": [u["client_id"] for u, _ in stale],
             "stale_superseded": superseded,
             "bytes_up": bytes_up, "bytes_down": bytes_down,
             "client_metrics": {u["client_id"]: u["metrics"] for u in updates}}
        self.history.append(h)
        return h

    def global_model(self) -> torch.nn.Module:
        """Current global encoder inside a fresh model (decoder from the template; encoder is what is probed)."""
        import copy
        m = copy.deepcopy(self.model)
        m.load_state_dict(self.coord.shared, strict=False)
        return m
