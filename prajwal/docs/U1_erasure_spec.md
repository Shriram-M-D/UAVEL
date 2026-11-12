# U1 - Erasure specification (v1)

## Request
`{request_id, zone_id, time_buckets?, client_ids?}` - forget data from zone Z (optionally only time buckets T,
only some clients). Schema and workflow states: `docs/INTERFACES.md` sections 7-8.

## What "erased" means here
The post-erasure model should be **indistinguishable, by the tests we run, from a model retrained from the
same initialisation on the same schedule without the target data** (the reference, U4).

| Stored state | Action | Exact? |
|---|---|---|
| Zone adapter / (zone, time) memory | Delete file + registry `purged` | Yes - stored separately, verified by `files_still_present == []` |
| Shared encoder, target never touched it (ledger) | Nothing | Yes |
| Shared encoder, target touched it | Roll back to newest clean checkpoint, replay retained contributions after it | Exact w.r.t. the schedule if replay uses the same data order/seeds (tested bit-for-bit) |
| Shared encoder, no clean checkpoint | Retrain from initialisation | Exact, full cost |
| Artefacts derived from tainted state (other zones' adapters built on a tainted encoder, cached embeddings) | Invalidate; retrain/rebuild | Exact once rebuilt; reported until then |
| Approximate alternative: fine-tune on retained data | Cheap, **not exact**; distance to reference must be reported | No |

## Never claimed
- That copies released before the request are erased.
- That an MIA AUC near 0.5 proves erasure (it is one signal; we also report distance to the reference and
  retained utility).
- That DP (if enabled) implies deletion.

## Verification (U6-U8), always all of:
1. MIA AUC on target members vs. same-zone non-members, with bootstrap 95% CI.
2. Representation distance (cosine, linear CKA) to the reference, on target-zone and retained-zone data.
3. Retained-zone utility (reconstruction loss, downstream probe) to catch collateral damage.
4. Latency as a fraction of full retraining.
