# Prajwal - Unlearning Algorithms & Verification

Owns the project's core contribution: resolving a zone/time erasure request, purging adapters and memory, repairing the shared encoder, building the retrain-without-target reference, and verifying erasure with multiple methods. Also runs the final experiments, ablations, statistics, novelty search and the final report.

**Out of scope:** Encoder/SSL training (Shreyas), federation/DP (Mrinal), API/ledger (Shriram), UI (Skanda).

> Rule: you only edit files inside your own folder. Anything shared goes through docs/INTERFACES.md (owned by Shriram) via a pull request that all owners review.

Full details: `PRD_Prajwal.docx` in this folder.

## Tasks

| ID | Task | Deliverable |
|---|---|---|
| U1 | Erasure specification | Design note + schema in INTERFACES.md. |
| U2 | Request resolver & purge | Resolver + purge with unit tests. |
| U3 | Shared-encoder repair | Unlearner with selectable strategy + latency numbers. |
| U4 | Retrain-without-target reference | Reference runner. |
| U5 | Comparison baselines | Baseline results. |
| U6 | Membership-inference verification | MIA module + report. |
| U7 | Output & representation comparison | Distance metrics. |
| U8 | Residual-artefact inspection & utility | Verifier returning one results JSON. |
| U9 | Experiment runner & ablations | experiments/ with configs + results files. |
| U10 | Final report & novelty check | Final report + novelty-check log. |

## Phases

- **Phase 0 (Week 1) - Setup & contracts:** Agree Unlearner/Verifier interfaces and results-JSON schema; write unlearning design note; threat-model review with Shriram.  
  *Exit:* Interfaces + metrics schema in INTERFACES.md v1.
- **Phase 1 (Weeks 2-3) - Local learning baseline:** Prototype MIA and distance metrics on Shreyas's single-client model; define the reference-training protocol.  
  *Exit:* Metrics run on a baseline checkpoint.
- **Phase 2 (Weeks 4-5) - Parameter-efficient adaptation & simulated federation:** U4 reference runner; U5 SISA baseline; early study of what a shared-encoder repair needs from Mrinal's round history.  
  *Exit:* Reference model trainable end to end.
- **Phase 3 (Weeks 6-7) - Partitioned state & unlearning:** U2 resolver/purge, U3 repair strategies, U1 finalised; hook into Shriram's erasure state machine.  
  *Exit:* Erasure request works on real adapters, memory and shared encoder.
- **Phase 4 (Weeks 8-9) - Verification & demo integration:** U6-U8 full verification; produce results JSON consumed by backend and Skanda's dashboard; tune and fix.  
  *Exit:* Verifier output shown live in demo.
- **Phase 5 (Week 10) - Ablation, final evaluation & report:** U9 multi-seed runs and ablations; U10 report and novelty check.  
  *Exit:* Final report and reproducible experiment configs.

## Provides / consumes

- **Provides:** Unlearner, Verifier, reference runner, results JSON, experiments, final report.
- **Consumes:** Shriram: ledger/registry/state machine. Shreyas: ActiveModel, adapters, memory, checkpoints. Mrinal: round history, federated baselines.
