# Mrinal - Federation Simulation & Privacy (DP)

Owns the multi-UAV simulation: isolated logical clients, FedAvg aggregation of shared components, intermittent connectivity, communication accounting, the privacy baselines (local-only, federated, DP-federated) and the optional simulated resource policy. Produces the round history the unlearner needs.

**Out of scope:** Encoder/SSL internals (Shreyas), deletion & verification (Prajwal), API (Shriram), UI (Skanda).

> Rule: you only edit files inside your own folder. Anything shared goes through docs/INTERFACES.md (owned by Shriram) via a pull request that all owners review.

Full details: `PRD_Mrinal.docx` in this folder.

## Tasks

| ID | Task | Deliverable |
|---|---|---|
| F1 | Client simulator | ClientSim class + tests that a client never reads another client's partition. |
| F2 | Coordinator & FedAvg | Aggregator with unit tests. |
| F3 | Framework decision & Flower integration | Decision note + working runner. |
| F4 | Intermittent connectivity | Config-driven dropout/delay model. |
| F5 | Communication accounting & locality evidence | comm_log.json + data-flow report. |
| F6 | Round history for unlearning | Round-history schema in INTERFACES.md + emitter. |
| F7 | Differential privacy (controlled experiment) | DP runs + utility/privacy trade-off table + assumptions note. |
| F8 | Baselines | Baseline results JSON. |
| F9 | Update-leakage check (basic) | Short leakage note with caveats. |
| F10 | Optional simulated resource policy | Optional module + clearly labelled results. |

## Phases

- **Phase 0 (Week 1) - Setup & contracts:** Agree FederationRunner interface and round-history schema; environment; read FL/DP references.  
  *Exit:* Interface + schema in INTERFACES.md v1.
- **Phase 1 (Weeks 2-3) - Local learning baseline:** F1 client simulator against Shreyas's single-client learner; local-only baseline runs per client.  
  *Exit:* N clients train locally with isolated data.
- **Phase 2 (Weeks 4-5) - Parameter-efficient adaptation & simulated federation:** F2-F5: FedAvg, framework decision, dropouts, communication logging, locality evidence, F6 round history; integrate with Shriram's API.  
  *Exit:* Multi-client federated run reproducible; bytes logged; ledger aggregate events present.
- **Phase 3 (Weeks 6-7) - Partitioned state & unlearning:** F7 DP experiments; F8 baselines; supply round history to Prajwal's repair logic.  
  *Exit:* DP/non-DP comparison table with accountant output.
- **Phase 4 (Weeks 8-9) - Verification & demo integration:** Federation view data for the dashboard; F9 leakage check; fix integration issues.  
  *Exit:* Frontend shows rounds and bytes from real runs.
- **Phase 5 (Week 10) - Ablation, final evaluation & report:** Multi-seed federation/DP runs for ablations; F10 only if time remains; write limitations.  
  *Exit:* Final federation/DP result files delivered.

## Provides / consumes

- **Provides:** FederationRunner, ClientSim, round history, comm logs, DP accounting, baselines.
- **Consumes:** Shreyas: LocalLearner/checkpoints. Shriram: config, ledger, guard. Prajwal: unlearning requirements for round history.
