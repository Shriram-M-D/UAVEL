# Prajwal (Unlearning & Verification) - progress: Phases 0-3 (+ core of Phase 4 verification)

```
cd prajwal
python -m pytest -q tests              # 7 tests (~10 s)
python run_erasure_experiment.py       # 3 seeds x 5 models (~7 min CPU)
python plot_results.py                 # redraw results/erasure_comparison.png from the JSON
```
Package: `uav_unlearning` (imports Shreyas's `src` and Shriram's `uav_backend`; never edits them).

| Task | Status | Where |
|---|---|---|
| U1 erasure specification | Done (v1) | `docs/U1_erasure_spec.md` |
| U2 resolver & purge | Done: plug-in for Shriram's workflow; purge verified (`files_still_present == []`). Resolution itself comes from Shriram's ledger. | `uav_unlearning/plugin.py` |
| U3 shared-encoder repair | Done: rollback+replay (exact), fine-tune-on-retained (approximate), purge-only (baseline). Backend plug-in lists exactly which retained contributions need replay and never claims success until replay runs. | `strategies.py`, `plugin.py` |
| U4 retrain reference | Done: same init, same mission schedule, target removed. | `strategies.py` |
| U5 SISA / FedEraser baselines | Not started. Rollback+replay is the SISA-style idea on missions; a sharded baseline is next. | |
| U6 membership inference | Done: loss-threshold MIA, bootstrap 95% CI; sanity-checked (AUC > 0.8 on a memorised model). | `verify.py` |
| U7 output/representation distance | Done: cosine distance + linear CKA to the reference. | `verify.py` |
| U8 residual influence & utility | Done: **zone-knowledge gap** (new), retained-zone loss, downstream probe, cost ratio. | `verify.py` |
| U9 multi-seed runner | Done for one target (3 seeds). More targets/sizes and ablations pending. | `run_erasure_experiment.py` |
| U10 report & novelty check | Not started. | |

## Experiment
uav0 flies **mission 1** over zone 4 only, then **mission 2** over zones 0 and 4 (30 epochs each).
Request: **forget zone 0**. The reference is retrained from the same start without any zone-0 image.

| Strategy (mean of 3 seeds) | Zone knowledge left | Distance to reference | Cost vs retrain | Exact? |
|---|---|---|---|---|
| Original / purge only | 0.036 +/- 0.005 | 0.052 | ~0 | No - encoder still knows zone 0 |
| Fine-tune on retained (10 ep) | 0.033 +/- 0.003 | 0.073 | 0.33 | No - barely forgets, drifts further from reference |
| **Rollback + replay** | **0.000** | **0.000** | **0.49** | **Yes - bit-identical to reference (also unit-tested)** |
| Retrain (reference) | 0 | 0 | 1.00 | Yes |

*Zone knowledge left* = reference loss - model loss on zone-0 images neither model trained on.

### Findings
1. **Deleting the zone's adapter/memory is not enough** when the zone also trained the shared encoder:
   the encoder still reconstructs zone 0 measurably better than a model that never saw it.
2. **Cheap fine-tuning on retained data does not remove it** (0.036 -> 0.033) and moves the model *further*
   from the reference.
3. **Checkpoint rollback + replay gives exact erasure at ~half the retraining cost**, because mission 1 (before
   zone 0 appeared) does not need to be redone. The saving grows with how late the target data arrived.
4. **Per-image membership inference finds nothing, even before erasure** (AUC ~0.48, CI includes 0.5): the model
   learned zone 0 as a whole, not individual images (member loss 0.255 vs non-member 0.249). For zone-level
   requests a zone-level test is needed - hence the zone-knowledge gap. This is a negative result we report.
5. Forgetting costs utility: the downstream probe drops 0.622 -> 0.580 after exact erasure, which is the
   reference's own level (less training data), not collateral damage.

### Limitations
- One client, one target zone, 3 seeds, tiny model, synthetic zones on EuroSAT.
- Wall-clock times of seeds 0-1 overlapped with other jobs on the same laptop; strategy/reference ratios were
  measured back-to-back and are stable (0.49 +/- 0.004), absolute seconds are not.
- Rollback+replay is exact only because training is deterministic given seeds and data order; with
  federation, replay must also redo the affected rounds (the backend plug-in lists them).
- The committed `zone_knowledge_gap` values were computed from the per-run losses saved by this run (same
  quantity the updated script now computes directly).
