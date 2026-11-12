# Shared interface contracts - v1

Owner: Shriram. Changes only through a PR approved by all five owners.
Status: **v1, proposed for freeze at end of Phase 0.** Implemented by `shriram/uav_backend` (see its tests).

## 0. Packages and imports (avoids name clashes)
| Owner | Folder | Import as | How |
|---|---|---|---|
| Shriram | `shriram/` | `uav_backend` | `sys.path.insert(0, "<repo>/shriram")` |
| Shreyas | `shreyas/` | `src` (existing) | `sys.path.insert(0, "<repo>/shreyas")` |
| Mrinal | `mrinal/` | `uav_federation` | `sys.path.insert(0, "<repo>/mrinal")` |
| Prajwal | `prajwal/` | `uav_unlearning` | `sys.path.insert(0, "<repo>/prajwal")` |

Nobody imports from another person's folder anything that is not listed in this file.

## 1. Manifest CSV (producer: Shreyas)
`sample_id, client_id, zone_id, time_bucket, split, path, zone_is_synthetic, label`
- `label` is for the downstream probe only; never used in SSL or unlearning.
- A zone belongs to exactly one client in the current simulation (`client = uav{zone % n_clients}`).

## 2. Checkpoint layout (producer: Shreyas)
```
<artefact_dir>/shared_encoder_v{N}.pt
<artefact_dir>/adapters/{client_id}_zone{zone_id}_v{K}.pt
<artefact_dir>/memory/{client_id}_zone{zone_id}_t{time_bucket}_v{K}.npz   # prototypes/statistics only
```
Every file written must also be registered (section 4) - an unregistered file cannot be erased.

## 3. Ledger event (owner: Shriram; producers: everyone via hooks)
```
event_id, seq, ts, run_id, event_type, client_id, model_version, parent_version,
artefact_ids[], touched_shared_encoder, payload{}, keys[{client_id, zone_id, time_bucket}]
```
- `event_type` in `train | aggregate | memory_write | checkpoint | erase | verify`.
- `keys` = every (client, zone, time) data unit that influenced the event. An aggregate has many.
- **`touched_shared_encoder` must be true whenever the step changed any shared parameter.** Getting
  this wrong silently breaks unlearning.
- Append-only (enforced by SQLite triggers).

Hooks (do not write SQL yourself):
```python
backend.record_train(client_id, zone_id, time_bucket, produced=[Artefact...], touched_shared_encoder=bool,
                     model_version=..., parent_version=..., payload={})
backend.record_memory_write(client_id, zone_id, time_bucket, memory=Artefact, payload={})
backend.record_aggregate(round_id, contributions=[Key...], new_encoder=Artefact, parent_version=..., payload={})
```

## 4. Registry artefact (owner: Shriram)
```
artefact_id, kind (shared_encoder|adapter|memory|checkpoint|derived), client_id, zone_id, time_bucket,
version, path, status (active|stopped|purged|invalidated|superseded), parents[], meta{}
```
- `parents` = the artefacts this one was computed from. An adapter's parents include the encoder version
  it was trained on top of. Erasure taints all descendants, so missing parents = missed influence.
- When you produce a new version, mark the old one `superseded` (`registry.set_status`).

## 5. Round history (producer: Mrinal)
```
round_id, model_version_in, model_version_out, contributions[{client_id, zone_id, time_bucket}],
dropped_clients[], bytes_up, bytes_down
```
- `contributions` = only (client, zone, time) units whose data changed the **shared** parameters sent in
  this round, since that client's previous round. Adapter-only training is not a contribution.
- Every client update must go through `backend.guard.check_update(payload)` before aggregation.

## 6. Client update payload (checked by the locality guard)
Allowed fields only: `client_id, round_id, model_version, num_samples, zone_ids, time_buckets, state_dict, metrics`.
- `state_dict` keys must start with a shared prefix (`federation.shared_param_prefixes` in the config).
- `metrics` values must be scalars. Anything array-/image-like elsewhere is rejected.

## 7. Erasure request and workflow (consumer: Prajwal)
```
request_id, zone_id, time_buckets[] | null, client_ids[] | null, requested_at
```
States: `RECEIVED -> RESOLVED -> STOPPED -> PURGED -> REPAIRED -> INVALIDATED -> VERIFIED`.
The backend owns RESOLVED / STOPPED / INVALIDATED. Plug-ins own PURGED / REPAIRED / VERIFIED.

`resolution` passed to plug-ins:
```
events[], event_types[], direct_artefacts[], tainted_artefacts[], tainted_shared_encoders[],
shared_encoder_affected, found_influence
```

## 8. Python plug-in interfaces (constructed as `Class(backend)`; set in `experiment.yaml` as `module:Class`)
```python
class LocalLearner:      # Shreyas
    def train(self, client_id, zone_id, time_bucket=None, adapter_only=False, steps=1) -> dict: ...
class FederationRunner:  # Mrinal
    def run_round(self, client_ids=None) -> dict: ...            # returns section 5
class Unlearner:         # Prajwal
    def purge(self, request, resolution) -> dict: ...            # {purged[], files_deleted}
    def repair(self, request, resolution) -> dict: ...           # {strategy, new_artefact_ids[], latency_s, not_guaranteed[]}
class Verifier:          # Prajwal
    def verify(self, request, report) -> dict: ...               # section 9
```
Stubs for all four live in `shriram/uav_backend/stubs.py`; they are labelled `stub: true` in every output.

## 9. Results JSON (producer: Prajwal)
```
request_id, strategy, latency_s, retrain_latency_s, latency_ratio, mia_auc_before, mia_auc_after, mia_ci,
output_distance_to_reference, retained_zone_utility_before, retained_zone_utility_after,
residual_findings[], not_guaranteed[]
```

## 10. REST API (owner: Shriram)
Generated spec: `docs/api/openapi.yaml` (run `python shriram/export_openapi.py`).
`GET /health, GET /clients, GET /clients/{id}/partitions, POST /clients/{id}/train, POST /federation/rounds,
GET /federation/evidence, POST /erasure-requests, GET /erasure-requests, GET /erasure-requests/{id},
POST /erasure-requests/{id}/run, GET /ledger, GET /artefacts, GET /metrics/{key}, POST /reset`.
Progress streaming (SSE) is planned for Phase 2.

## Changelog
- v1: package names; ledger keys; registry parents/status; contribution rule; guard payload; split
  Unlearner into purge/repair so the backend owns the state machine; erasure `time_buckets` instead of `time_range`.
