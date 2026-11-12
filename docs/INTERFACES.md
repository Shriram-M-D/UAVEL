# Shared interface contracts (v1 DRAFT)

Owner: Shriram. Changes only through a PR approved by all five owners. Frozen at end of Phase 0.

## 1. Manifest CSV (producer: Shreyas)
`sample_id, client_id, zone_id, time_bucket, split, path, zone_is_synthetic`

## 2. Checkpoint layout (producer: Shreyas)
```
checkpoints/shared_encoder_v{N}.pt
checkpoints/adapters/{zone_id}.pt
checkpoints/memory/{zone_id}_{time_bucket}.npz   # prototypes/statistics only, never raw images
```

## 3. Ledger event (producer: everyone via hooks; owner: Shriram)
`event_id, ts, client_id, zone_id, time_bucket, event_type (train|aggregate|memory_write|checkpoint|erase|verify), model_version, parent_version, artefact_ids[], touched_shared_encoder`

## 4. Round history (producer: Mrinal)
`round_id, model_version_in, model_version_out, contributions[{client_id, zone_ids[], time_buckets[]}], dropped_clients[], bytes_up, bytes_down`

## 5. Erasure request (producer: Frontend/API; consumer: Prajwal)
`request_id, zone_id, time_range (optional), client_ids (optional), requested_at`

## 6. Results JSON (producer: Prajwal)
`request_id, strategy, latency_s, retrain_latency_s, latency_ratio, mia_auc_before, mia_auc_after, mia_ci, output_distance_to_reference, retained_zone_utility_before, retained_zone_utility_after, residual_findings[], not_guaranteed[]`

## 7. Python interfaces
```python
class LocalLearner:            # Shreyas
    def train(self, client_id, zone_id, steps) -> TrainResult: ...
class FederationRunner:        # Mrinal
    def run_round(self, client_ids) -> RoundHistory: ...
class Unlearner:               # Prajwal
    def erase(self, request) -> ErasureReport: ...
class Verifier:                # Prajwal
    def verify(self, request, report) -> ResultsJSON: ...
```

## 8. REST API (owner: Shriram, spec in docs/api/openapi.yaml)
GET /clients, GET /clients/{id}/partitions, POST /clients/{id}/train, POST /federation/rounds, POST /erasure-requests, GET /erasure-requests/{id}, GET /ledger, GET /metrics/{run_id}, POST /reset. Progress via server-sent events.
