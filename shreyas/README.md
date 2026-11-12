# Shreyas - AI Core: Data, Self-Supervised Learning & Inference

Owns everything that makes a simulated UAV 'learn locally': dataset and zone manifest, the lightweight encoder, masked-reconstruction self-supervised learning, zone adapters, partitioned memory, continual-learning baselines, downstream evaluation of representations, and the inference API for the active model.

**Out of scope:** Federated aggregation and DP (Mrinal), deletion/repair and verification attacks (Prajwal), API/ledger (Shriram), UI (Skanda).

> Rule: you only edit files inside your own folder. Anything shared goes through docs/INTERFACES.md (owned by Shriram) via a pull request that all owners review.

Full details: `PRD_Shreyas.docx` in this folder.

## Tasks

| ID | Task | Deliverable |
|---|---|---|
| A1 | Dataset selection & dataset card | docs: dataset card with licence decision. |
| A2 | Zone/time manifest builder | Reproducible manifest + split report. |
| A3 | Preprocessing pipeline | Dataset/DataLoader module + tests. |
| A4 | Encoder + masked-reconstruction SSL | Repeatable single-client baseline (first 'UAV learns locally' demo). |
| A5 | Downstream evaluation harness | Eval script + metrics JSON (schema in INTERFACES.md). |
| A6 | Zone adapters (parameter-efficient fine-tuning) | Adapter module + comparison table. |
| A7 | Partitioned memory | Memory store + deletion unit tests. |
| A8 | Continual-learning baseline | Baseline script + results. |
| A9 | Provenance hooks | Hooks verified against Shriram's ledger stub. |
| A10 | Inference API | Inference module consumed by backend and Prajwal's verifier. |

## Phases

- **Phase 0 (Week 1) - Setup & contracts:** Agree manifest schema and checkpoint layout; dataset licence check; environment + requirements; pick encoder.  
  *Exit:* Manifest schema and checkpoint layout in INTERFACES.md v1.
- **Phase 1 (Weeks 2-3) - Local learning baseline:** A1-A5: manifest, preprocessing, encoder + MAE, single-client SSL, linear-probe eval; frozen-encoder fallback.  
  *Exit:* Local SSL beats frozen baseline on held-out probe (H1) or negative result documented.
- **Phase 2 (Weeks 4-5) - Parameter-efficient adaptation & simulated federation:** A6 adapters, adapter-vs-full comparison, expose LocalLearner interface used by Mrinal's clients; A9 hooks.  
  *Exit:* Per-zone adapters train; shared vs adapter params clearly separated for federation.
- **Phase 3 (Weeks 6-7) - Partitioned state & unlearning:** A7 partitioned memory, A8 continual baseline, cross-zone leakage check of adapters/memory; support Prajwal's retrain-reference runs.  
  *Exit:* Partition deletion leaves others intact; reference model reproducible.
- **Phase 4 (Weeks 8-9) - Verification & demo integration:** A10 inference API used in demo and verifier; performance and memory measurements; bug fixes from integration.  
  *Exit:* Inference works on active model after erasure.
- **Phase 5 (Week 10) - Ablation, final evaluation & report:** Ablations touching adapters/memory/full FT; multi-seed runs; document model limitations.  
  *Exit:* Final utility tables and plots delivered to Prajwal.

## Provides / consumes

- **Provides:** Manifest, LocalLearner, ActiveModel inference, adapters, memory store, checkpoints, utility metrics.
- **Consumes:** Shriram: ledger hooks, config, registry. Mrinal: aggregated shared-encoder weights. Prajwal: repaired checkpoints.
