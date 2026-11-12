"""U6-U8: erasure verification from several angles. No single number is treated as proof.

- membership inference (loss-threshold attack on masked-reconstruction loss) with bootstrap CI
- representation distance to the retrain reference (cosine distance, linear CKA)
- target-zone reconstruction loss vs. reference (does the model still 'know' the zone better than it should?)
- retained-zone utility (reconstruction loss + downstream linear probe) to catch collateral damage
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from . import _paths  # noqa: F401
from src.evaluation.probe import linear_probe, embed  # Shreyas


@torch.no_grad()
def per_sample_recon_loss(model, x, seed: int = 123, repeats: int = 3, bs: int = 256) -> np.ndarray:
    """Masked-reconstruction loss per image, averaged over `repeats` fixed random masks."""
    model.eval()
    out = np.zeros(len(x))
    for r in range(repeats):
        torch.manual_seed(seed + r)
        for i in range(0, len(x), bs):
            xb = x[i:i + bs]
            t = model.patchify(xb)
            _, pred, mask = model(xb)
            l = (((pred - t) ** 2).mean(-1) * mask).sum(1) / mask.sum(1)
            out[i:i + bs] += l.numpy() / repeats
    return out


def mia_auc(model, members, nonmembers, n_boot: int = 500, seed: int = 0) -> dict:
    """AUC of 'lower loss => member'. 0.5 = attacker cannot tell members from non-members."""
    lm, ln = per_sample_recon_loss(model, members), per_sample_recon_loss(model, nonmembers)
    y = np.r_[np.ones(len(lm)), np.zeros(len(ln))]
    s = -np.r_[lm, ln]
    auc = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            boots.append(roc_auc_score(y[i], s[i]))
    return {"auc": float(auc), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_members": int(len(lm)), "n_nonmembers": int(len(ln)),
            "member_loss": float(lm.mean()), "nonmember_loss": float(ln.mean())}


def zone_knowledge_gap(model, ref_model, x_unseen_target) -> float:
    """Zone-level influence test (better suited than per-sample MIA to zone erasure).

    Mean loss of the reference minus mean loss of the model on target-zone images that NEITHER model
    trained on. >0: the model still reconstructs the zone better than a model that never saw it.
    0: matches the reference. Same fixed masks for both models.
    """
    return float(per_sample_recon_loss(ref_model, x_unseen_target).mean()
                 - per_sample_recon_loss(model, x_unseen_target).mean())


def _linear_cka(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(0), b - b.mean(0)
    hsic = np.linalg.norm(a.T @ b) ** 2
    return float(hsic / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)))


def representation_distance(m1, m2, x) -> dict:
    e1, e2 = embed(m1, x), embed(m2, x)
    cos = (e1 * e2).sum(1) / (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) + 1e-12)
    return {"mean_cosine_distance": float(1 - cos.mean()), "linear_cka": _linear_cka(e1, e2)}


def utility(model, x_retained_test, probe_data) -> dict:
    xtr, ytr, xte, yte = probe_data
    return {"retained_zone_recon_loss": float(per_sample_recon_loss(model, x_retained_test).mean()),
            "probe_acc": linear_probe(model, xtr, ytr, xte, yte)}
