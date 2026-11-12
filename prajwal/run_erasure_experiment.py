"""U3-U9: erase zone 0 from UAV uav0 and verify against a retrain-without-target reference.

Scenario (synthetic zones, EuroSAT proxy):
  Mission 1 (time buckets 0-1): uav0 learns from zone 4 only.          -> checkpoint C1
  Mission 2 (time buckets 2-3): uav0 learns from zones 0 and 4.        -> model M (deployed)
  Request: forget zone 0.

Members (MIA)    = zone-0 images used in mission 2.
Non-members      = zone-0 images never trained on (other time buckets + held-out splits), balanced.
Run from prajwal/:  python run_erasure_experiment.py      (env SEEDS=3 EPOCHS=30 FT_EPOCHS=10)
"""
import json, os, sys, time
from pathlib import Path

os.environ["UAV_NO_APP"] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from uav_unlearning.strategies import Mission, run_schedule, reference, purge_only, rollback_replay, finetune_retained  # noqa: E402
from uav_unlearning.verify import mia_auc, representation_distance, utility, zone_knowledge_gap  # noqa: E402
from src.data.dataset import load_images, to_tensor  # noqa: E402  (Shreyas)
from src.data.manifest import ROOT as SHREYAS_ROOT  # noqa: E402
from src.models.mae import MAE  # noqa: E402

SEEDS = int(os.environ.get("SEEDS", 3))
EPOCHS = int(os.environ.get("EPOCHS", 30))
FT_EPOCHS = int(os.environ.get("FT_EPOCHS", 10))
CLIENT, TARGET, RETAINED = "uav0", 0, 4
torch.set_num_threads(8)
out = HERE / "results"
out.mkdir(exist_ok=True)

df = pd.read_csv(SHREYAS_ROOT / "results" / "manifest.csv")
X = to_tensor(load_images(df))
c = df.client_id == CLIENT
train = c & (df.split == "ssl_train")
m1 = (train & (df.zone_id == RETAINED) & df.time_bucket.isin([0, 1])).values
m2_target = (train & (df.zone_id == TARGET) & df.time_bucket.isin([2, 3])).values
m2_retained = (train & (df.zone_id == RETAINED) & df.time_bucket.isin([2, 3])).values
nonmem = ((df.zone_id == TARGET) & ~((train & df.time_bucket.isin([2, 3])).values)).values
ret_test = ((df.zone_id == RETAINED) & (df.split == "test")).values
probe = (X[(df.split == "probe_train").values], df.label.values[(df.split == "probe_train").values],
         X[(df.split == "test").values], df.label.values[(df.split == "test").values])

rng = np.random.default_rng(0)
mem_idx, non_idx = np.flatnonzero(m2_target), np.flatnonzero(nonmem)
k = min(len(mem_idx), len(non_idx))
mem_eval, non_eval = X[rng.choice(mem_idx, k, replace=False)], X[rng.choice(non_idx, k, replace=False)]
x_target_all = X[(df.zone_id == TARGET).values & (df.split == "test").values]
print(f"mission1 {m1.sum()} imgs (zone {RETAINED}) | mission2 {m2_target.sum()} zone-{TARGET} + {m2_retained.sum()} zone-{RETAINED}"
      f" | MIA eval {k} members vs {k} non-members")

idx2 = np.r_[np.flatnonzero(m2_target), np.flatnonzero(m2_retained)]
is_target2 = np.r_[np.ones(m2_target.sum(), bool), np.zeros(m2_retained.sum(), bool)]


def evaluate(name, model, ref_model, seconds, ref_seconds):
    mia = mia_auc(model, mem_eval, non_eval)
    d_t = representation_distance(model, ref_model, x_target_all)
    d_r = representation_distance(model, ref_model, X[ret_test])
    u = utility(model, X[ret_test], probe)
    return {"strategy": name, "latency_s": round(seconds, 2), "retrain_latency_s": round(ref_seconds, 2),
            "latency_ratio": round(seconds / ref_seconds, 3) if ref_seconds else None,
            "mia": mia, "target_member_loss": mia["member_loss"],
            "zone_knowledge_gap": zone_knowledge_gap(model, ref_model, non_eval),
            "distance_to_reference_target_zone": d_t, "distance_to_reference_retained_zone": d_r, **u}


runs = []
for s in range(SEEDS):
    t_seed = time.time()
    torch.manual_seed(s)
    init = MAE()
    missions = [Mission(X[m1], EPOCHS, 1000 * s + 1), Mission(X[idx2], EPOCHS, 1000 * s + 2, contains_target=True)]
    keep = [np.ones(m1.sum(), bool), ~is_target2]  # per-mission mask of samples that are NOT target

    M = run_schedule(init, missions)
    R = reference(init, missions, keep)
    strategies = {
        "original (no erasure)": (M, M.seconds),
        "purge_only": (purge_only(M), 0.0),
        "rollback_replay": (rr := rollback_replay(init, M, missions, keep), rr.seconds),
        f"finetune_retained_{FT_EPOCHS}ep": (ft := finetune_retained(M, X[np.r_[np.flatnonzero(m1), np.flatnonzero(m2_retained)]], FT_EPOCHS, 7 + s), ft.seconds),
        "reference (retrain w/o target)": (R, R.seconds),
    }
    rows = [evaluate(n, t.model, R.model, sec, R.seconds) for n, (t, sec) in strategies.items()]
    for r in rows:
        r["seed"] = s
        print(f"seed {s} | {r['strategy']:<32} MIA AUC {r['mia']['auc']:.3f} [{r['mia']['ci95'][0]:.2f},{r['mia']['ci95'][1]:.2f}]"
              f" | cos-dist to ref (target) {r['distance_to_reference_target_zone']['mean_cosine_distance']:.4f}"
              f" | retained recon {r['retained_zone_recon_loss']:.4f} | probe {r['probe_acc']:.3f}"
              f" | {r['latency_s']:.1f}s ({r['latency_ratio']})", flush=True)
    runs += rows
    print(f"seed {s} done in {time.time() - t_seed:.0f}s", flush=True)

# aggregate over seeds
names = list(dict.fromkeys(r["strategy"] for r in runs))
def agg(key):
    out_ = {}
    for n in names:
        v = np.array([key(r) for r in runs if r["strategy"] == n], float)
        out_[n] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    return out_
summary = {
    "zone_knowledge_gap": agg(lambda r: r["zone_knowledge_gap"]),
    "mia_auc": agg(lambda r: r["mia"]["auc"]),
    "target_member_recon_loss": agg(lambda r: r["target_member_loss"]),
    "cosine_distance_to_reference_target_zone": agg(lambda r: r["distance_to_reference_target_zone"]["mean_cosine_distance"]),
    "cka_to_reference_target_zone": agg(lambda r: r["distance_to_reference_target_zone"]["linear_cka"]),
    "retained_zone_recon_loss": agg(lambda r: r["retained_zone_recon_loss"]),
    "probe_acc": agg(lambda r: r["probe_acc"]),
    "latency_ratio": agg(lambda r: r["latency_ratio"] or 0.0),
}
res = {"setup": {"client": CLIENT, "target_zone": TARGET, "retained_zone": RETAINED, "seeds": SEEDS,
                 "epochs_per_mission": EPOCHS, "finetune_epochs": FT_EPOCHS, "mia_pairs": int(k),
                 "zones_are_synthetic": True, "dataset": "EuroSAT subset (proxy)"},
       "summary": summary, "runs": runs}
(out / "erasure_metrics.json").write_text(json.dumps(res, indent=2))

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
short = [n.split(" ")[0] for n in names]
for a, (key, title) in zip(ax, [("mia_auc", "MIA AUC on target zone (0.5 = can't tell)"),
                                ("cosine_distance_to_reference_target_zone", "Distance to retrain reference (target zone)"),
                                ("latency_ratio", "Cost vs full retrain")]):
    m = [summary[key][n]["mean"] for n in names]
    e = [summary[key][n]["std"] for n in names]
    a.bar(short, m, yerr=e, capsize=4, color=["#999", "#C55A11", "#2F5496", "#548235", "#333"])
    a.set_title(title, fontsize=10)
    a.tick_params(axis="x", rotation=30, labelsize=8)
ax[0].axhline(0.5, ls=":", color="k")
plt.tight_layout()
plt.savefig(out / "erasure_comparison.png", dpi=130)
plt.close()
print(json.dumps({k_: {n: round(v["mean"], 4) for n, v in d.items()} for k_, d in summary.items()}, indent=1))
