# Skanda - Frontend: Demo Dashboard

Owns the demo interface that makes the end-to-end workflow visible: fleet and zone view, local training, federation rounds with proof that no raw images leave clients, an erasure console, a verification dashboard comparing the unlearned model with the retrain reference, a ledger viewer, and reset/re-run controls.

**Out of scope:** Any model, federation, unlearning or API server code. Skanda consumes the REST API only and works against a mock server until the backend is ready.

> Rule: you only edit files inside your own folder. Anything shared goes through docs/INTERFACES.md (owned by Shriram) via a pull request that all owners review.

Full details: `PRD_Skanda.docx` in this folder.

## Tasks

| ID | Task | Deliverable |
|---|---|---|
| S1 | App foundation | Running app shell + generated client. |
| S2 | Mock server | `npm run dev:mock`. |
| S3 | Fleet & zones view | Fleet overview page. |
| S4 | Client detail & local training | Client page. |
| S5 | Federation view | Federation page. |
| S6 | Erasure console | Erasure page. |
| S7 | Verification dashboard | Verification page. |
| S8 | Ledger viewer | Ledger page. |
| S9 | Experiments & ablation view | Experiments page. |
| S10 | Demo mode, QA & accessibility | E2E test + demo script. |

## Phases

- **Phase 0 (Week 1) - Setup & contracts:** Review OpenAPI draft; wireframes for six pages; choose chart library and component style.  
  *Exit:* Wireframes approved by team; OpenAPI feedback sent to Shriram.
- **Phase 1 (Weeks 2-3) - Local learning baseline:** S1 app shell + generated client; S2 mock server; S3 fleet view; S4 client detail with mock training.  
  *Exit:* Fleet and client pages demoable on mocks.
- **Phase 2 (Weeks 4-5) - Parameter-efficient adaptation & simulated federation:** S5 federation view; connect to the real API for clients/training; handle SSE.  
  *Exit:* Real training progress visible in UI.
- **Phase 3 (Weeks 6-7) - Partitioned state & unlearning:** S6 erasure console and workflow stepper against the backend state machine; S8 ledger viewer.  
  *Exit:* Submit an erasure request and watch it progress.
- **Phase 4 (Weeks 8-9) - Verification & demo integration:** S7 verification dashboard with real results JSON; polish; error handling; responsive pass.  
  *Exit:* Full live demo with all pages on real data.
- **Phase 5 (Week 10) - Ablation, final evaluation & report:** S9 experiments view; S10 demo mode, accessibility, Playwright test, screenshots.  
  *Exit:* Final demo script rehearsed; E2E passes.

## Provides / consumes

- **Provides:** Web dashboard.
- **Consumes:** Shriram: REST API + openapi.yaml + SSE. Prajwal: results JSON schema. Mrinal: federation data via API.
