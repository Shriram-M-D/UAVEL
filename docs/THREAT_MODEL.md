# Threat model (Step 1 deliverable) - draft v1

Owner: Shriram. Reviewed by Prajwal (unlearning) and Mrinal (federation/DP).

## Erasure unit
A zone `Z`, optionally restricted to time buckets `T` and/or clients: request `(Z, T)`.
Zones are **synthetic** in the current dataset (EuroSAT proxy). No legal/geographic compliance claim.

## What counts as what
| Term | Meaning in this system |
|---|---|
| Local data | Images in a client's partition. Never leave the client (enforced by the locality guard). |
| Shared model state | Shared-encoder versions produced by local SSL or FedAvg. |
| Zone state | Per-zone adapters and (zone, time) memory partitions (prototypes/statistics, never raw images). |
| Checkpoint / derived artefact | Any stored state computed from the above; tracked with parents in the registry. |
| Optimizer state | Not persisted across versions in the PoC; if added, must be registered. |
| Released artefact | Anything copied outside the system before a request. **Cannot be recalled.** |

## Actors and capabilities
| Actor | Can see | In scope? |
|---|---|---|
| Honest-but-curious coordinator | All client updates (shared params, metadata) and aggregated models | Yes - updates leak test (Mrinal F9), DP optional |
| Passive communication observer | Same as coordinator, in transit | Yes - same evidence as above |
| Attacker who captures a UAV | That client's stored encoder, adapters, memory | Yes - post-erasure inspection (Prajwal U8) |
| Post-erasure auditor/attacker | Model after erasure; query access | Yes - membership inference + reference comparison |
| Malicious coordinator (deviates from protocol) | Anything | **No** - out of scope, stated explicitly |
| Poisoning client | - | **No** - out of scope |
| Physical/side-channel attacks | - | **No** - out of scope |

## Success definition
Erasure is successful when the post-erasure model is measurably close to a model retrained without the
target data (output distance, membership-inference AUC near chance on target vs. held-out, retained-zone
utility preserved). File deletion alone is **not** success.

## Claims we will not make
- FL alone makes updates private.
- Deleting source files or ledger rows removes learned influence (the ledger is append-only for this reason).
- DP guarantees exact deletion; one attack proves complete erasure.
- Synthetic zones are real places; simulated budgets are measured energy.
