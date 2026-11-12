# F3 - Federation framework decision: lightweight custom loop (not Flower, for now)

**Decision:** implement FedAvg as a small in-process loop (`uav_federation/runner.py`), behind the same
`FederationRunner.run_round(client_ids)` interface that a Flower version would expose.

| Criterion | Custom loop | Flower simulation |
|---|---|---|
| Runs on the team's Windows laptops, CPU-only | Yes | Simulation engine relies on Ray; extra setup and memory |
| Every update can pass Shriram's locality guard | Direct call before aggregation | Needs a custom Strategy hook |
| Exact control of dropouts/delays/stale updates (F4) | Yes, ~30 lines | Possible, more configuration |
| Ledger round history (F6) | Written in the same place aggregation happens | Needs a custom Strategy |
| Extra dependencies | None | `flwr`, `ray` |
| "Uses a standard FL framework" for the report | No | Yes |

**Revisit when:** clients must run as separate processes, or the report needs a standard-framework baseline.
Because the interface is fixed, a Flower runner can be added as `uav_federation.flower_runner:FlowerRunner`
and selected in `experiment.yaml` without changing the backend, the frontend, or the tests' expectations.

**What is aggregated:** shared-encoder parameters only (`embed.`, `pos`, `blocks.`, `norm.`).
The MAE decoder and zone adapters stay on each UAV. This keeps zone-specific state local, which is what
makes adapter-only erasure cheap (see Shriram's demo, erasure A).
