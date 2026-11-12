"""B6: REST API consumed by Skanda's dashboard. Spec is exported to docs/api/openapi.yaml.

Run (from shriram/):  python -m uvicorn uav_backend.api:app --reload
Config path can be set with the UAV_CONFIG environment variable.
"""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .core import Backend
from .erasure import ErasureRequest


class TrainBody(BaseModel):
    zone_id: int
    time_bucket: Optional[int] = None
    adapter_only: bool = False
    steps: int = 1


class RoundBody(BaseModel):
    client_ids: Optional[list[str]] = None


class ErasureBody(BaseModel):
    zone_id: int
    time_buckets: Optional[list[int]] = None
    client_ids: Optional[list[str]] = None
    run: bool = True  # execute the full workflow immediately (Phase 1); async jobs + SSE come in Phase 2


def create_app(backend: Optional[Backend] = None) -> FastAPI:
    b = backend or Backend(os.environ.get("UAV_CONFIG"))
    app = FastAPI(title="UAV Privacy PoC API", version="0.1.0",
                  description="Backend for privacy-preserving self-learning UAVs with zone-level unlearning. "
                              "All zones are synthetic unless a client reports otherwise.")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.backend = b

    @app.get("/health")
    def health():
        return {"ok": True, "plugins": b.plugins.sources, "events": b.ledger.count()}

    @app.get("/clients")
    def clients():
        return [{k: v for k, v in c.items() if k != "partitions"} | {"n_partitions": len(c["partitions"])}
                for c in b.clients()]

    @app.get("/clients/{client_id}/partitions")
    def partitions(client_id: str):
        for c in b.clients():
            if c["client_id"] == client_id:
                return {**c, "active_model": b.registry.active_model(client_id)}
        raise HTTPException(404, f"unknown client {client_id}")

    @app.post("/clients/{client_id}/train")
    def train(client_id: str, body: TrainBody):
        if client_id not in {c["client_id"] for c in b.clients()}:
            raise HTTPException(404, f"unknown client {client_id}")
        return b.plugins.learner.train(client_id, body.zone_id, body.time_bucket, body.adapter_only, body.steps)

    @app.post("/federation/rounds")
    def federation_round(body: RoundBody):
        return b.plugins.federation.run_round(body.client_ids)

    @app.get("/federation/evidence")
    def evidence():
        return b.guard.evidence()

    @app.post("/erasure-requests")
    def submit_erasure(body: ErasureBody):
        req = ErasureRequest(zone_id=body.zone_id, time_buckets=body.time_buckets, client_ids=body.client_ids)
        b.erasure.submit(req)
        return b.erasure.run(req.request_id) if body.run else b.erasure.report(req.request_id)

    @app.post("/erasure-requests/{request_id}/run")
    def run_erasure(request_id: str, until: Optional[str] = None):
        if not b.erasure.get(request_id):
            raise HTTPException(404, request_id)
        return b.erasure.run(request_id, until)

    @app.get("/erasure-requests")
    def list_erasures():
        return [b.erasure.report(r["request"]["request_id"]) for r in b.erasure.list()]

    @app.get("/erasure-requests/{request_id}")
    def get_erasure(request_id: str):
        if not b.erasure.get(request_id):
            raise HTTPException(404, request_id)
        return b.erasure.report(request_id)

    @app.get("/ledger")
    def ledger(event_type: Optional[str] = None, zone_id: Optional[int] = None,
               limit: int = Query(500, le=5000), offset: int = 0):
        return [asdict(e) for e in b.ledger.events(event_type, zone_id, limit, offset)]

    @app.get("/artefacts")
    def artefacts(kind: Optional[str] = None, status: Optional[str] = None, client_id: Optional[str] = None):
        return [asdict(a) for a in b.registry.list(kind, status, client_id)]

    @app.get("/metrics/{key}")
    def metrics(key: str):
        m = b.get_metrics(key)
        if m is None:
            raise HTTPException(404, key)
        return m

    @app.post("/reset")
    def reset():
        b.reset()
        return {"ok": True}

    return app


app = create_app() if os.environ.get("UAV_NO_APP") is None else None
