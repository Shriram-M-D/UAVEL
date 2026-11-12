# Shriram (Backend & Integration) - progress: Phase 0 + Phase 1 (+ B7, B9 early)

```
cd shriram
pip install -r requirements.txt
python -m pytest -q tests            # 18 tests
python demo_backend.py               # end-to-end with stub ML plug-ins
python -m uvicorn uav_backend.api:app --reload   # API on :8000, docs at /docs
python export_openapi.py             # regenerate docs/api/openapi.yaml
```

| Task | Status | Where |
|---|---|---|
| B1 governance & CI | Done: CODEOWNERS (own entries filled), per-folder CI, PR template. Teammates' usernames still placeholders. | `.github/` |
| B2 contracts | v1 written, proposed for freeze; package naming fixed to avoid `src` clashes. | `docs/INTERFACES.md` |
| B3 config | Done: Pydantic-validated YAML, run_id + snapshot. | `uav_backend/config.py`, `configs/experiment.yaml` |
| B4 ledger | Done: append-only SQLite (triggers block UPDATE/DELETE), multi-key events, `influenced_by(zone, time, clients)`. | `uav_backend/ledger.py` |
| B5 registry | Done: artefact DAG, status lifecycle, `descendants()`, `purge()`, `active_model()`. | `uav_backend/registry.py` |
| B6 API | Done (sync): 14 endpoints, OpenAPI exported. SSE progress = Phase 2. | `uav_backend/api.py`, `docs/api/openapi.yaml` |
| B7 erasure state machine | Done early: 7 states, persisted, resumable, idempotent, every transition in the ledger. | `uav_backend/erasure.py` |
| B8 plug-ins & stubs | Done: 4 interfaces, `module:Class` loading, labelled stubs incl. rollback unlearner. | `uav_backend/plugins.py`, `stubs.py` |
| B9 locality guard | Done early: field allow-list, shared-prefix check, image heuristics, comm log + evidence. | `uav_backend/guard.py` |
| B10 demo/docs | Partial: threat-model draft, backend demo. | `docs/THREAT_MODEL.md`, `demo_backend.py` |

**Phase 1 exit criterion met:** a stub training run writes ledger events visible via `GET /ledger` (tested).

## What the demo shows (`results/demo_backend_report.json`)
- 4 client updates accepted (1.44 MB of shared parameters); a payload with an `images` field is blocked.
- **Erase zone 0 (adapter-only):** purges its adapter + memory; ledger shows no shared-encoder influence ->
  no repair; other zones untouched.
- **Erase zone 2 (updated the shared encoder):** taints every later encoder version and every adapter built
  on them; rolls the encoder back to the last clean version; invalidates uav3's zone-3 adapter; reports that
  retained-zone updates after the rollback point are lost until replayed.

This is the design argument for zone adapters, shown by the provenance system itself.

## Limitations
- ML plug-ins are stubs (labelled `stub: true`) until Shreyas/Mrinal/Prajwal plug in.
- Rollback repair is the crude baseline; Prajwal's repair strategies replace it.
- Synchronous API; async jobs and SSE in Phase 2.
