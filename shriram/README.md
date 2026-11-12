# Shriram - Backend & Integration Lead

Owns the system spine: the zone/time provenance ledger, the versioned state registry, the REST API that the frontend consumes, the job/erasure-workflow orchestrator, and integration of everyone else's modules. Also owns the shared contracts (docs/INTERFACES.md, docs/api/openapi.yaml), CI and the repo governance files.

**Out of scope:** Model training code, federation algorithms, unlearning algorithms and metrics (owned by Shreyas, Mrinal, Prajwal) and UI code (Skanda). Shriram calls them through agreed interfaces, using stubs until they land.

> Rule: you only edit files inside your own folder. Anything shared goes through docs/INTERFACES.md (owned by Shriram) via a pull request that all owners review.

Full details: `PRD_Shriram.docx` in this folder.

## Tasks

| ID | Task | Deliverable |
|---|---|---|
| B1 | Repo governance & CI | Green CI on empty folders; CODEOWNERS enforced. |
| B2 | Interface contracts | Contracts v1 signed off by all five by end of Phase 0. |
| B3 | Experiment configuration | `load_config()` + snapshot writer + tests. |
| B4 | Provenance ledger | Ledger library with unit tests for targeted lookup. |
| B5 | State registry | Registry with delete/invalidate tests proving unrelated partitions stay intact. |
| B6 | REST API | OpenAPI spec frozen in Phase 0, implemented by Phase 3. |
| B7 | Job runner & erasure workflow | State machine with audit log per transition. |
| B8 | Integration adapters & stubs | `make demo` works end to end with stubs by Phase 2. |
| B9 | Raw-data locality guard | Guard + tests + data-flow evidence JSON. |
| B10 | Demo, docs & threat model | Demo script, docs/THREAT_MODEL.md, tagged v1.0. |

## Phases

- **Phase 0 (Week 1) - Setup & contracts:** Repo skeleton, CODEOWNERS, CI, contracts v1 (INTERFACES.md + openapi.yaml), config schema, threat-model draft.  
  *Exit:* All five owners approve contracts; CI green.
- **Phase 1 (Weeks 2-3) - Local learning baseline:** Ledger (B4) and state registry (B5) core; config loader; stub adapters for AI module; API skeleton for /clients and /train.  
  *Exit:* A stub training run writes ledger events visible via GET /ledger.
- **Phase 2 (Weeks 4-5) - Parameter-efficient adaptation & simulated federation:** Job runner; federation endpoint; locality guard (B9); wire in Shreyas's real learner and Mrinal's runner behind the interface; SSE progress.  
  *Exit:* Multi-client round runs through the API; bytes logged; guard rejects image payloads.
- **Phase 3 (Weeks 6-7) - Partitioned state & unlearning:** Erasure endpoint + state machine (B7); ledger queries used by Prajwal's unlearner; invalidation of derived artefacts.  
  *Exit:* POST /erasure-requests drives resolve->purge->repair using real components.
- **Phase 4 (Weeks 8-9) - Verification & demo integration:** Verification results endpoint; reset/re-run; full end-to-end demo with Skanda's UI; performance and error handling pass.  
  *Exit:* Live demo: train, erase, verify, reset, repeat.
- **Phase 5 (Week 10) - Ablation, final evaluation & report:** Bug fixing, docs, runbook, limitations, release tag; help Prajwal run experiments via the job runner.  
  *Exit:* v1.0 tag; clean-machine setup test passes.

## Provides / consumes

- **Provides:** Ledger + registry Python library, REST API, orchestration, config loader, locality guard.
- **Consumes:** Shreyas: LocalLearner + checkpoint layout. Mrinal: FederationRunner + round history. Prajwal: Unlearner, Verifier, metrics JSON.
