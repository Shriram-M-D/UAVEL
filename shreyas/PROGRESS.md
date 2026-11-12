# Shreyas (AI core) - progress report: Phase 1 + Phase 2 prototype

Run on a laptop CPU (RTX 3050 not used; torch CPU build). Reproduce from `shreyas/`:

```
python run_phase1.py 60     # local SSL per UAV client + linear probe
python run_phase2.py        # per-zone adapters + isolation test
```

## What was built (PRD tasks)
| Task | Status | Where |
|---|---|---|
| A1 dataset | Done (proxy): EuroSAT 6,000-image subset stands in for UAVid/VisDrone (no registration needed). **No real geotags -> zones are synthetic and flagged.** UAVid/VisDrone licence check still open. | `src/data/manifest.py` |
| A2 zone/time manifest | Done: 8 synthetic zones, 4 UAV clients (2 zones each), 4 time buckets, sample-level splits (`ssl_train` / `probe_train` / `test`). Labels are used only by the probe. | `results/manifest.csv` |
| A3 preprocessing | Done: fixed 64x64 resize + constant normalisation, cached. | `src/data/dataset.py` |
| A4 encoder + masked-reconstruction SSL | Done: tiny ViT-MAE (0.95 M params, 75% masking), trained **per client on its own unlabeled partition only**. | `src/models/mae.py`, `src/learning/local_ssl.py` |
| A5 downstream eval | Done: linear probe on a held-out labelled split. | `src/evaluation/probe.py` |
| A6 zone adapters | Done (prototype): bottleneck adapters per zone, shared encoder frozen. | `run_phase2.py` |
| A7 memory, A8 continual, A9 ledger hooks, A10 inference API | Not started. | |

## Results
**Local self-learning (H1):** linear-probe accuracy on the held-out set

| Model | Probe acc |
|---|---|
| Frozen, untrained encoder | 0.609 |
| uav0 after local SSL | 0.629 |
| uav1 after local SSL | 0.624 |
| uav2 after local SSL | 0.646 |
| uav3 after local SSL | 0.663 |

Every UAV improves (+1.5 to +5.4 points) using only its own unlabeled images; raw images never leave the client partition. Gains are modest: the model is tiny and trained for 60 epochs on ~1k images, single seed, so no claim beyond "direction is positive". Reconstruction loss falls on every client (`results/phase1_loss_curves.png`, `phase1_reconstruction.png`).

**Zone adapters (A6):** each adapter has 16,960 parameters vs 951,232 for the encoder (1.8%). Adapter-only training lowers masked-reconstruction loss on its zone (zone 0: 0.188 -> 0.176; zone 4: 0.457 -> 0.446).

**Isolation test (first unlearning-relevant evidence):** deleting zone 0's adapter leaves zone 4's output **bit-identical** and returns zone 0 exactly to the frozen baseline.

## Honest limitations
- Adapter-only training was on top of an encoder already trained with SSL on these zones, so **shared-encoder influence from the zone remains**. Removing the adapter does not remove that; this is exactly Prajwal's shared-encoder repair problem.
- Zone-0 adapter also slightly lowers loss on zone 4 (0.4565 -> 0.4475), so adapters are not perfectly zone-specific in effect (only in storage/deletion).
- EuroSAT is satellite imagery at 64x64, not UAV video. Single seed; no confidence intervals yet.

## Hand-offs / next
- Shriram: ledger hooks (A9) need the event schema; checkpoints already follow `results/checkpoints/adapters/zone{N}.pt`.
- Prajwal: `*_local_encoder.pt` + adapters are ready to use for erasure and retrain-reference experiments.
- Next for me: A7 partitioned memory, A9 hooks, A10 inference API, multi-seed runs.
