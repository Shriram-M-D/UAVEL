"""Phase 1 demo: each simulated UAV learns locally (SSL on its own unlabeled images), then is probed.
Run from shreyas/:  python run_phase1.py [epochs]
"""
import json, copy, sys, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data.manifest import build_manifest, ROOT
from src.data.dataset import load_images, to_tensor
from src.models.mae import MAE
from src.learning.local_ssl import train_local
from src.evaluation.probe import linear_probe

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
torch.set_num_threads(8)
out = ROOT / "results"
out.mkdir(exist_ok=True)
df = build_manifest()
df.to_csv(out / "manifest.csv", index=False)
X = to_tensor(load_images(df))
y = df.label.values
tr, te = (df.split == "probe_train").values, (df.split == "test").values
print(f"manifest: {len(df)} images, clients={sorted(df.client_id.unique())} (synthetic zones)")

torch.manual_seed(0)
base = MAE()
res = {"epochs": EPOCHS, "dataset": "EuroSAT subset (proxy for aerial imagery; synthetic zones)", "clients": {}}
res["frozen_random_encoder_probe_acc"] = linear_probe(base, X[tr], y[tr], X[te], y[te])
print("baseline (frozen, untrained encoder) probe acc:", round(res["frozen_random_encoder_probe_acc"], 4))
curves = {}
for cid in sorted(df.client_id.unique()):
    m = ((df.client_id == cid) & (df.split == "ssl_train")).values  # ONLY this UAV's local unlabeled data
    zones = [int(z) for z in sorted(df[m].zone_id.unique())]
    print(f"[{cid}] local SSL on {int(m.sum())} images, zones={zones}")
    model, info = train_local(X[m], epochs=EPOCHS, model=copy.deepcopy(base))
    acc = linear_probe(model, X[tr], y[tr], X[te], y[te])
    res["clients"][cid] = dict(n_images=int(m.sum()), zones=zones, probe_acc=acc,
                               final_recon_loss=info["loss_history"][-1],
                               train_seconds=round(info["train_seconds"], 1), trainable_params=info["trainable_params"])
    curves[cid] = info["loss_history"]
    torch.save(model.state_dict(), out / f"{cid}_local_encoder.pt")
    print(f"[{cid}] probe acc {acc:.4f} (frozen baseline {res['frozen_random_encoder_probe_acc']:.4f})")
    last = model

plt.figure(figsize=(6, 4))
for c, h in curves.items():
    plt.plot(h, label=c)
plt.xlabel("epoch")
plt.ylabel("masked-patch reconstruction loss")
plt.title("Local SSL per UAV client")
plt.legend()
plt.tight_layout()
plt.savefig(out / "phase1_loss_curves.png", dpi=130)
plt.close()

plt.figure(figsize=(5, 4))
names = ["frozen\n(no learning)"] + list(res["clients"])
vals = [res["frozen_random_encoder_probe_acc"]] + [v["probe_acc"] for v in res["clients"].values()]
plt.bar(names, vals, color=["#999"] + ["#2F5496"] * len(res["clients"]))
plt.ylabel("linear-probe accuracy")
plt.title("Utility before vs after local learning")
plt.tight_layout()
plt.savefig(out / "phase1_probe_accuracy.png", dpi=130)
plt.close()

last.eval()
torch.manual_seed(1)
with torch.no_grad():
    xs = X[te][:6]
    _, pred, mask = last(xs)
    tg, mk = last.patchify(xs), mask[..., None]
    vis = last.unpatchify(tg * (1 - mk))
    comp = last.unpatchify(tg * (1 - mk) + pred * mk)
im = lambda t: (t * 0.2 + 0.4).clamp(0, 1).permute(1, 2, 0).numpy()
fig, ax = plt.subplots(3, 6, figsize=(10, 5))
for i in range(6):
    for r, (t, n) in enumerate([(xs, "original"), (vis, "75% masked"), (comp, "reconstructed")]):
        ax[r, i].imshow(im(t[i]))
        ax[r, i].axis("off")
        if i == 0:
            ax[r, i].set_title(n, fontsize=8, loc="left")
plt.tight_layout()
plt.savefig(out / "phase1_reconstruction.png", dpi=130)
plt.close()
json.dump(res, open(out / "phase1_metrics.json", "w"), indent=2)
print(json.dumps(res, indent=2))
