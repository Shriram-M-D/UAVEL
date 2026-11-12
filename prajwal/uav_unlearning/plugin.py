"""U2/U3 integration: Unlearner plug-in for Shriram's erasure workflow (PURGED and REPAIRED steps).

Select it in experiment.yaml:   plugins: { unlearner: uav_unlearning.plugin:CheckpointReplayUnlearner }

repair() rolls the shared encoder back to the newest version with no target influence, then lists exactly
which retained (client, zone, time) contributions must be replayed to recover their learning. If a
`replay` callable is attached (e.g. by the federation runner), it performs the replay and the result is
registered as the new encoder; otherwise the report says replay is outstanding instead of claiming success.
"""
from __future__ import annotations

import os, time
from typing import Callable, Optional

from . import _paths  # noqa: F401
from uav_backend import Artefact  # Shriram


class CheckpointReplayUnlearner:
    replay: Optional[Callable] = None  # replay(clean_encoder: Artefact, retained_keys: list[dict]) -> path

    def __init__(self, backend):
        self.b = backend

    def purge(self, request: dict, resolution: dict) -> dict:
        reg = self.b.registry
        target = [i for i in resolution["tainted_artefacts"]
                  if (a := reg.get(i)) and a.kind in ("adapter", "memory") and a.zone_id == request["zone_id"]
                  and (request.get("time_buckets") is None or a.time_bucket in (None, *request["time_buckets"]))]
        files = reg.purge(target)
        leftovers = [i for i in target if reg.get(i).path and os.path.exists(reg.get(i).path)]
        return {"purged": target, "files_deleted": len(files), "files_still_present": leftovers}

    def repair(self, request: dict, resolution: dict) -> dict:
        t0 = time.time()
        reg, ledger = self.b.registry, self.b.ledger
        tainted = set(resolution["tainted_artefacts"])
        encoders = reg.list(kind="shared_encoder")
        clean = [a for a in encoders if a.artefact_id not in tainted and a.status not in ("purged", "invalidated")]
        if not clean:
            return {"strategy": "checkpoint_replay", "new_artefact_ids": [], "latency_s": time.time() - t0,
                    "not_guaranteed": ["no clean encoder checkpoint; full retrain from initialisation required"]}
        base = clean[-1]
        # retained contributions that entered the encoder AFTER the rollback point must be replayed
        lost_versions = {a.version for a in encoders if a.artefact_id in tainted}
        retained = []
        for e in ledger.events(event_type="aggregate", limit=5000) + ledger.events(event_type="train", limit=5000):
            if e.model_version in lost_versions or e.parent_version in lost_versions:
                for k in e.keys:
                    if k.zone_id != request["zone_id"] or (request.get("time_buckets") and
                                                           k.time_bucket not in request["time_buckets"]):
                        retained.append({"client_id": k.client_id, "zone_id": k.zone_id, "time_bucket": k.time_bucket})
        retained = [dict(t) for t in {tuple(sorted(d.items())) for d in retained}]
        path = base.path
        replayed = False
        if self.replay is not None and retained:
            path = self.replay(base, retained)
            replayed = True
        n = len(encoders) + 1
        new = Artefact(f"enc-v{n}", "shared_encoder", version=f"enc-v{n}", path=path, parents=[base.artefact_id],
                       meta={"strategy": "checkpoint_replay", "rolled_back_to": base.artefact_id,
                             "replayed": replayed, "erasure_request": request["request_id"]})
        reg.register(new)
        ng = [] if (replayed or not retained) else [
            f"{len(retained)} retained contribution(s) after {base.artefact_id} are not yet replayed; "
            "the repaired encoder has forgotten them too until replay runs"]
        return {"strategy": "checkpoint_replay", "rolled_back_to": base.artefact_id,
                "new_artefact_ids": [new.artefact_id], "retained_to_replay": sorted(retained, key=str),
                "replayed": replayed, "latency_s": round(time.time() - t0, 4), "not_guaranteed": ng}
