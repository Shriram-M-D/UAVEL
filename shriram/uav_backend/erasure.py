"""B7: erasure workflow state machine.

RECEIVED -> RESOLVED -> STOPPED -> PURGED -> REPAIRED -> INVALIDATED -> VERIFIED

Each transition is persisted and written to the ledger, so a request can be resumed after a crash,
re-run safely (a VERIFIED request is a no-op) and audited. Purge/repair/verify are delegated to the
Unlearner/Verifier plug-ins; resolution, stopping and invalidation are owned here.
"""
from __future__ import annotations

import json, sqlite3, time, uuid
from typing import Optional

from pydantic import BaseModel, Field

from .ledger import Event, Key

STATES = ["RECEIVED", "RESOLVED", "STOPPED", "PURGED", "REPAIRED", "INVALIDATED", "VERIFIED"]
ALWAYS_NOT_GUARANTEED = ["Models or artefacts released before the request cannot be recalled by this protocol."]

SCHEMA = """
CREATE TABLE IF NOT EXISTS erasure_requests (
  request_id TEXT PRIMARY KEY, request TEXT NOT NULL, state TEXT NOT NULL, steps TEXT NOT NULL, updated_ts REAL);
"""


class ErasureRequest(BaseModel):
    zone_id: int
    time_buckets: Optional[list[int]] = None   # None = all time buckets
    client_ids: Optional[list[str]] = None     # None = all clients
    request_id: str = Field(default_factory=lambda: "er-" + uuid.uuid4().hex[:8])
    requested_at: float = Field(default_factory=time.time)


class ErasureWorkflow:
    def __init__(self, backend):
        self.b = backend
        self.conn: sqlite3.Connection = backend.conn
        self.conn.executescript(SCHEMA)

    # ---- persistence -----------------------------------------------------------------------------
    def submit(self, req: ErasureRequest) -> dict:
        if self.get(req.request_id):
            return self.get(req.request_id)  # idempotent
        with self.conn:
            self.conn.execute("INSERT INTO erasure_requests VALUES (?,?,?,?,?)",
                              (req.request_id, req.model_dump_json(), "RECEIVED", "{}", time.time()))
        self._audit(req.model_dump(), "RECEIVED", {})
        return self.get(req.request_id)

    def get(self, request_id: str) -> Optional[dict]:
        r = self.conn.execute("SELECT request, state, steps FROM erasure_requests WHERE request_id=?",
                              (request_id,)).fetchone()
        if not r:
            return None
        return {"request": json.loads(r[0]), "state": r[1], "steps": json.loads(r[2])}

    def list(self) -> list[dict]:
        return [self.get(i) for (i,) in self.conn.execute("SELECT request_id FROM erasure_requests ORDER BY updated_ts")]

    def _advance(self, rid: str, state: str, steps: dict) -> None:
        with self.conn:
            self.conn.execute("UPDATE erasure_requests SET state=?, steps=?, updated_ts=? WHERE request_id=?",
                              (state, json.dumps(steps), time.time(), rid))

    def _audit(self, req: dict, state: str, output: dict) -> None:
        keys = [Key(c, req["zone_id"], t) for c in (req.get("client_ids") or [None])
                for t in (req.get("time_buckets") or [None])]
        self.b.ledger.append(Event("verify" if state == "VERIFIED" else "erase", keys=keys,
                                   payload={"request_id": req["request_id"], "state": state, "output": output}))

    # ---- execution -------------------------------------------------------------------------------
    def run(self, request_id: str, until: Optional[str] = None) -> dict:
        rec = self.get(request_id)
        if rec is None:
            raise KeyError(request_id)
        req, steps = rec["request"], rec["steps"]
        handlers = {"RESOLVED": self._resolve, "STOPPED": self._stop, "PURGED": self._purge,
                    "REPAIRED": self._repair, "INVALIDATED": self._invalidate, "VERIFIED": self._verify}
        for state in STATES[STATES.index(rec["state"]) + 1:]:
            t0 = time.time()
            out = handlers[state](req, steps)
            out["seconds"] = round(time.time() - t0, 4)
            steps[state] = out
            self._advance(request_id, state, steps)
            self._audit(req, state, out)
            if state == until:
                break
        return self.report(request_id)

    def report(self, request_id: str) -> dict:
        rec = self.get(request_id)
        s = rec["steps"]
        ng = list(ALWAYS_NOT_GUARANTEED)
        for st in s.values():
            ng += st.get("not_guaranteed", [])
        return {"request": rec["request"], "state": rec["state"], "steps": s,
                "complete": rec["state"] == "VERIFIED",
                "not_guaranteed": list(dict.fromkeys(ng)),
                "results": s.get("VERIFIED", {}).get("results")}

    # ---- steps -----------------------------------------------------------------------------------
    def _resolve(self, req: dict, steps: dict) -> dict:
        evs = self.b.ledger.influenced_by(req["zone_id"], req.get("time_buckets"), req.get("client_ids"))
        direct = sorted({a for e in evs for a in e.artefact_ids})
        tainted = sorted(set(direct) | self.b.registry.descendants(direct))
        enc = [a for a in tainted if (x := self.b.registry.get(a)) and x.kind == "shared_encoder"]
        shared = any(e.touched_shared_encoder for e in evs) or bool(enc)
        return {"events": [e.event_id for e in evs], "event_types": sorted({e.event_type for e in evs}),
                "direct_artefacts": direct, "tainted_artefacts": tainted,
                "tainted_shared_encoders": enc, "shared_encoder_affected": shared,
                "found_influence": bool(evs)}

    def _stop(self, req: dict, steps: dict) -> dict:
        reg = self.b.registry
        ids = [a for a in steps["RESOLVED"]["tainted_artefacts"] if (x := reg.get(a)) and x.status == "active"]
        reg.set_status(ids, "stopped")  # excluded from aggregation and inference from now on
        return {"stopped": ids}

    def _purge(self, req: dict, steps: dict) -> dict:
        return self.b.plugins.unlearner.purge(req, steps["RESOLVED"])

    def _repair(self, req: dict, steps: dict) -> dict:
        if not steps["RESOLVED"]["shared_encoder_affected"]:
            return {"required": False, "reason": "no ledger event shows target data touching the shared encoder"}
        return {"required": True, **self.b.plugins.unlearner.repair(req, steps["RESOLVED"])}

    def _invalidate(self, req: dict, steps: dict) -> dict:
        reg = self.b.registry
        ids = [a for a in steps["RESOLVED"]["tainted_artefacts"]
               if (x := reg.get(a)) and x.status in ("stopped", "superseded")]
        reg.set_status(ids, "invalidated")
        return {"invalidated": ids}

    def _verify(self, req: dict, steps: dict) -> dict:
        results = self.b.plugins.verifier.verify(req, {"steps": steps})
        self.b.save_metrics(req["request_id"], results)
        return {"results": results}
