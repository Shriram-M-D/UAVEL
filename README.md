# Privacy-Preserving Self-Learning UAVs with Zone-Level Machine Unlearning

Software-only proof of concept: simulated UAVs learn locally from unlabeled aerial images, keep raw data local, partition model state by zone/time, and support verified zone-level unlearning.

## Team folders (one owner each - no overlapping edits)

| Folder | Owner | Role |
|---|---|---|
| `shriram/` | Shriram | Backend & Integration Lead |
| `shreyas/` | Shreyas | AI Core: Data, Self-Supervised Learning & Inference |
| `mrinal/` | Mrinal | Federation Simulation & Privacy (DP) |
| `prajwal/` | Prajwal | Unlearning Algorithms & Verification |
| `skanda/` | Skanda | Frontend: Demo Dashboard |

## Rules

1. Edit only your own folder. Root files and `docs/` belong to Shriram.
2. Branch as `feature/<folder>-<topic>`, rebase on `main`, squash-merge via PR.
3. Cross-module changes go through `docs/INTERFACES.md` and need all owners' approval.
4. No datasets, checkpoints or secrets in git.

PRDs: `<folder>/PRD_<Name>.docx`. Plan: 6 phases (0-5) over ~10 weeks - see each PRD.
