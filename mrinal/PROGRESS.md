# Mrinal (Federation & Privacy) - progress: Phase 1 + Phase 2 (+ first F8 baseline)

```
cd mrinal
python -m pytest -q tests          # 8 tests (tiny random data, ~5 s)
python run_federation.py           # FedAvg 10 rounds x 6 local epochs vs local-only (~6 min CPU)
python run_connectivity.py         # dropouts + delayed/stale clients demo (~2 min)
```
Package: `uav_federation` (imports Shreyas's `src` and Shriram's `uav_backend`; never edits them).

| Task | Status | Where |
|---|---|---|
| F1 client simulator | Done: each UAV gets a **copy of only its own rows/pixels** (tested). Trains with Shreyas's MAE loop. | `uav_federation/clients.py` |
| F2 FedAvg | Done: sample-weighted, **shared encoder only**; decoder + zone adapters stay on the UAV. | `coordinator.py` |
| F3 framework decision | Done: lightweight custom loop now, Flower-compatible interface. | `docs/F3_framework_decision.md` |
| F4 intermittent connectivity | Done: dropout + delayed (stale) updates; a fresh update supersedes a stale one (no double weight). | `runner.py`, `results/connectivity_demo.json` |
| F5 comms accounting | Done: every update through Shriram's locality guard; bytes up/down per round; data-flow report. | `comms.py` |
| F6 round history -> ledger | Done: each round = one `aggregate` event keyed by exactly the (client, zone, time) units that reached the new encoder; encoder versions saved as `.pt` with parents. | `coordinator.py` |
| F7 DP (Opacus) | Not started (Phase 3). Opacus not installed; `nn.MultiheadAttention` needs Opacus's DP replacement. | |
| F8 baselines | Local-only vs FedAvg (non-DP) done. DP and continual-replay baselines pending. | `results/fedavg_metrics.json` |
| F9, F10 | Not started (optional). | |

## Results (EuroSAT proxy, synthetic zones, 4 UAVs, seed 0, same init as Shreyas)
| Model | Linear-probe acc |
|---|---|
| Frozen, no learning | 0.609 |
| Local-only, mean of 4 UAVs (60 epochs each) | 0.641 |
| **FedAvg global encoder** (10 rounds x 6 local epochs) | **0.637** |

- One shared model for all four UAVs matches the average local model, with no raw image ever sent, and was
  still improving at round 10 (`results/fedavg_vs_local.png`). It dips in round 2 before recovering.
- **Data flow:** 40 updates accepted, 0 rejected, 0 raw-image bytes sent.
- **Honest caveat:** 132 MB of parameters were uploaded, more than the 51 MB of raw pixels held across all
  UAVs. On a dataset this small, FL costs more bandwidth than sharing images; its benefit here is
  privacy/data locality, not communication.
- **Connectivity demo:** with 25% dropout + 25% delay, the ledger records which zones reached each encoder
  version (e.g. round 1 only zones {2, 6}), which is what Prajwal's erasure needs.

## Bug found and fixed during the demo
A delayed client that also sent a fresh update in the next round was aggregated twice. Now the stale update
is still received (bytes counted) but dropped from aggregation; logged as `stale_superseded`; test added.

## Limitations / next
- Single seed. No DP yet - nothing here is a formal privacy guarantee (FL alone is not one).
- Next: F7 Opacus DP-SGD with accountant logging, F8 DP baseline, multi-seed runs.
