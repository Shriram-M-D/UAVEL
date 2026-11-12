"""Re-draw results/erasure_comparison.png from results/erasure_metrics.json (no retraining)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "results" / "erasure_metrics.json").read_text())
S = d["summary"]
names = [n for n in S["zone_knowledge_gap"] if not n.startswith("original")]
labels = {"purge_only": "purge only", "rollback_replay": "rollback +\nreplay", "reference (retrain w/o target)": "retrain\n(reference)"}
lab = [labels.get(n, n.replace("finetune_retained_", "fine-tune\nretained ")) for n in names]
colors = ["#C55A11", "#2F5496", "#548235", "#333333"]
panels = [("zone_knowledge_gap", "Zone knowledge left\n(0 = same as never trained)", None),
          ("cosine_distance_to_reference_target_zone", "Distance to retrain reference\n(target-zone embeddings)", None),
          ("mia_auc", "Per-image MIA AUC\n(no signal even before erasure)", 0.5),
          ("latency_ratio", "Cost vs full retrain", 1.0)]
fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
for a, (k, title, line) in zip(ax, panels):
    a.bar(lab, [S[k][n]["mean"] for n in names], yerr=[S[k][n]["std"] for n in names], capsize=4, color=colors)
    a.set_title(title, fontsize=10)
    a.tick_params(axis="x", labelsize=8)
    if line is not None:
        a.axhline(line, ls=":", color="k", lw=1)
ax[2].set_ylim(0.4, 0.6)
fig.suptitle(f"Erase zone {d['setup']['target_zone']} from {d['setup']['client']} - mean +/- sd over "
             f"{d['setup']['seeds']} seeds (synthetic zones, EuroSAT proxy)", fontsize=11)
plt.tight_layout()
plt.savefig(HERE / "results" / "erasure_comparison.png", dpi=130)
print("wrote results/erasure_comparison.png")
