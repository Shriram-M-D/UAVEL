"""B8: deterministic stub plug-ins. They do no real ML; they produce the same ledger/registry footprint the
real modules must produce, so the whole pipeline (API -> ledger -> erasure -> report) is testable today.
Every stub output is labelled stub=True so it can never be mistaken for a real result.
"""
from __future__ import annotations

import json, time
from typing import Optional

import numpy as np

from .ledger import Key
from .registry import Artefact


def _write(backend, name: str, data: dict) -> str:
    p = backend.artefact_dir / name
    p.write_text(json.dumps(data))
    return str(p)


def _next_encoder(backend, parent: Optional[Artefact], meta: dict) -> Artefact:
    n = len(backend.registry.list(kind="shared_encoder")) + 1
    v = f"enc-v{n}"
    return Artefact(artefact_id=v, kind="shared_encoder", version=v, parents=[parent.artefact_id] if parent else [],
                    path=_write(backend, f"{v}.json", {"version": v, "stub": True}), meta={"stub": True, **meta})


def ensure_base_encoder(backend) -> Artefact:
    enc = backend.registry.latest_shared_encoder()
    if enc is None:
        enc = _next_encoder(backend, None, {"note": "pre-trained initialisation; no project data"})
        backend.registry.register(enc)
    return enc


class StubLearner:
    """Mimics Shreyas's local SSL: updates the shared encoder (unless adapter_only), a zone adapter and a
    (zone, time) memory partition."""

    def __init__(self, backend):
        self.b = backend

    def train(self, client_id: str, zone_id: int, time_bucket: Optional[int] = None,
              adapter_only: bool = False, steps: int = 1) -> dict:
        b, reg = self.b, self.b.registry
        tb = 0 if time_bucket is None else time_bucket
        enc = ensure_base_encoder(b)
        produced = []
        new_enc = None
        if not adapter_only:
            new_enc = _next_encoder(b, enc, {"trained_by": client_id})
            produced.append(new_enc)
        base = new_enc or enc
        old = [a for a in reg.list(kind="adapter", status="active", client_id=client_id) if a.zone_id == zone_id]
        k = len(reg.list(kind="adapter", client_id=client_id)) + 1
        ad = Artefact(artefact_id=f"{client_id}-adapter-z{zone_id}-v{k}", kind="adapter", client_id=client_id,
                      zone_id=zone_id, version=f"v{k}", parents=[base.artefact_id] + [a.artefact_id for a in old],
                      path=_write(b, f"{client_id}_adapter_z{zone_id}_v{k}.json", {"stub": True}), meta={"stub": True})
        mem = Artefact(artefact_id=f"{client_id}-mem-z{zone_id}-t{tb}-v{k}", kind="memory", client_id=client_id,
                       zone_id=zone_id, time_bucket=tb, parents=[ad.artefact_id],
                       path=_write(b, f"{client_id}_mem_z{zone_id}_t{tb}_v{k}.json", {"prototypes": "stub"}),
                       meta={"stub": True, "contents": "prototypes/statistics only"})
        produced += [ad]
        ev = b.record_train(client_id, zone_id, tb, produced, touched_shared_encoder=not adapter_only,
                            model_version=base.version, parent_version=enc.version,
                            payload={"steps": steps, "adapter_only": adapter_only, "stub": True})
        b.record_memory_write(client_id, zone_id, tb, mem)
        reg.set_status([a.artefact_id for a in old], "superseded")
        if new_enc:
            reg.set_status([enc.artefact_id], "superseded")
        return {"event_id": ev.event_id, "client_id": client_id, "zone_id": zone_id, "time_bucket": tb,
                "shared_encoder": base.artefact_id, "adapter": ad.artefact_id, "memory": mem.artefact_id,
                "touched_shared_encoder": not adapter_only, "stub": True}


class StubFederation:
    """Mimics Mrinal's FedAvg round: each client sends a shared-parameter update through the locality guard,
    the coordinator records an 'aggregate' event keyed by every zone that influenced the new encoder."""

    def __init__(self, backend):
        self.b = backend

    def run_round(self, client_ids: Optional[list[str]] = None) -> dict:
        b = self.b
        client_ids = client_ids or [c["client_id"] for c in b.clients()]
        enc = ensure_base_encoder(b)
        rid = f"round-{len(b.ledger.events(event_type='aggregate')) + 1}"
        rng = np.random.default_rng(0)
        contributions, bytes_up = [], 0
        prev = b.ledger.events(event_type="aggregate", limit=5000)
        since = prev[-1].seq if prev else 0
        for cid in client_ids:
            # only data that changed SHARED params since the last round influences this update;
            # adapter-only training stays on the UAV and is not a contribution
            trained = [e for e in b.ledger.events(event_type="train", limit=5000)
                       if e.client_id == cid and e.touched_shared_encoder and e.seq > since]
            keys = {(k.client_id, k.zone_id, k.time_bucket) for e in trained for k in e.keys}
            update = {"client_id": cid, "round_id": rid, "model_version": enc.version,
                      "num_samples": 1, "zone_ids": sorted({k[1] for k in keys}),
                      "state_dict": {"embed.weight": rng.normal(size=(128, 192)).astype(np.float32),
                                     "blocks.0.mlp.0.weight": rng.normal(size=(512, 128)).astype(np.float32)}}
            bytes_up += b.guard.check_update(update)
            contributions += [Key(*k) for k in sorted(keys, key=str)]
        new = _next_encoder(b, enc, {"round_id": rid, "aggregation": "fedavg-stub"})
        b.record_aggregate(rid, contributions, new, parent_version=enc.version,
                           payload={"clients": client_ids, "bytes_up": bytes_up, "stub": True})
        b.registry.set_status([enc.artefact_id], "superseded")
        return {"round_id": rid, "model_version_in": enc.version, "model_version_out": new.version,
                "contributions": [k.__dict__ for k in contributions], "dropped_clients": [],
                "bytes_up": bytes_up, "bytes_down": 0, "stub": True}


class RollbackUnlearner:
    """Baseline unlearner: purge zone-specific state; repair the shared encoder by rolling back to the
    newest encoder version with no target influence. Honest but crude - retained-zone learning after the
    rollback point is lost unless replayed. Prajwal's module replaces this."""

    def __init__(self, backend):
        self.b = backend

    def purge(self, request: dict, resolution: dict) -> dict:
        reg = self.b.registry
        ids = [i for i in resolution["tainted_artefacts"]
               if (a := reg.get(i)) and a.kind in ("adapter", "memory") and a.zone_id == request["zone_id"]]
        files = reg.purge(ids)
        return {"purged": ids, "files_deleted": len(files)}

    def repair(self, request: dict, resolution: dict) -> dict:
        t0 = time.time()
        reg, tainted = self.b.registry, set(resolution["tainted_artefacts"])
        clean = [a for a in reg.list(kind="shared_encoder") if a.artefact_id not in tainted and a.status != "purged"]
        if not clean:
            return {"strategy": "rollback", "new_artefact_ids": [], "latency_s": time.time() - t0,
                    "not_guaranteed": ["no clean encoder version exists; full retrain required"]}
        target = clean[-1]
        new = _next_encoder(self.b, target, {"repaired_from": target.artefact_id, "strategy": "rollback",
                                             "erasure_request": request["request_id"]})
        self.b.registry.register(new)
        lost = sorted({a.artefact_id for a in reg.list(kind="shared_encoder") if a.artefact_id in tainted})
        return {"strategy": "rollback", "rolled_back_to": target.artefact_id, "new_artefact_ids": [new.artefact_id],
                "latency_s": time.time() - t0, "retained_updates_lost": lost,
                "not_guaranteed": ["retained-zone updates after the rollback point are lost until replayed"]}


class StubVerifier:
    def __init__(self, backend):
        self.b = backend

    def verify(self, request: dict, report: dict) -> dict:
        return {"request_id": request["request_id"], "verifier": "stub", "stub": True,
                "mia_auc_before": None, "mia_auc_after": None, "output_distance_to_reference": None,
                "note": "No real verification ran. Prajwal's Verifier replaces this."}
