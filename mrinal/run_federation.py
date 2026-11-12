"""F2-F6 + F8 (first baseline): FedAvg over 4 simulated UAVs vs. Shreyas's local-only SSL.

Fair comparison: same init (seed 0), same data, same local epochs per client in total
(ROUNDS x LOCAL_EPOCHS = 60 = Shreyas's local-only run). Run from mrinal/:  python run_federation.py
"""
import json, os, sys, time
from pathlib import Path

os.environ["UAV_NO_APP"] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from uav_federation import FederationRunner, dataflow_report  # noqa: E402
from uav_backend.core import Backend  # noqa: E402
from src.evaluation.probe import linear_probe  # noqa: E402  (Shreyas)
from src.data.dataset import load_images, to_tensor  # noqa: E402
import pandas as pd  # noqa: E402

ROUNDS, LOCAL_EPOCHS = int(os.environ.get("ROUNDS", 10)), int(os.environ.get("LOCAL_EPOCHS", 6))
torch.set_num_threads(8)
out = HERE / "results"
out.mkdir(exist_ok=True)

b = Backend(HERE / "configs" / "federation.yaml")
b.reset()
t0 = time.time()
runner = FederationRunner(b, local_epochs=LOCAL_EPOCHS)
print(f"clients: { {c: len(v.x) for c, v in runner.clients.items()} } | rounds={ROUNDS} x {LOCAL_EPOCHS} local epochs")

df = pd.read_csv(b.cfg.path(b.cfg.dataset.manifest_path))
X = to_tensor(load_images(df))
y = df.label.values
tr, te = (df.split == "probe_train").values, (df.split == "test").values
probe_by_round = {0: linear_probe(runner.global_model(), X[tr], y[tr], X[te], y[te])}
for r in range(ROUNDS):
    h = runner.run_round()
    loss = sum(m["recon_loss"] for m in h["client_metrics"].values()) / len(h["client_metrics"])
    msg = f"{h['round_id']}: {h['model_version_in']} -> {h['model_version_out']} | mean local recon {loss:.4f} | up {h['bytes_up']/1e6:.2f} MB"
    if (r + 1) % 2 == 0 or r == ROUNDS - 1:
        probe_by_round[r + 1] = linear_probe(runner.global_model(), X[tr], y[tr], X[te], y[te])
        msg += f" | global probe acc {probe_by_round[r + 1]:.4f}"
    print(msg, flush=True)

local = json.loads((HERE.parent / "shreyas" / "results" / "phase1_metrics.json").read_text())
local_accs = {c: v["probe_acc"] for c, v in local["clients"].items()}
raw_bytes = int(sum(c.x.numel() for c in runner.clients.values()))  # uint8-equivalent pixel count
flow = dataflow_report(b, runner.history, raw_bytes)
res = {
    "setup": {"rounds": ROUNDS, "local_epochs_per_round": LOCAL_EPOCHS, "total_local_epochs": ROUNDS * LOCAL_EPOCHS,
              "aggregation": "FedAvg (sample-weighted), shared encoder only", "dp": False,
              "dataset": local["dataset"], "seconds": round(time.time() - t0, 1)},
    "baseline_frozen_probe_acc": local["frozen_random_encoder_probe_acc"],
    "local_only_probe_acc": local_accs,
    "local_only_mean": sum(local_accs.values()) / len(local_accs),
    "federated_global_probe_acc_by_round": probe_by_round,
    "federated_final_probe_acc": probe_by_round[ROUNDS],
    "dataflow": flow,
    "ledger": {"events": b.ledger.count(), "aggregate_events": len(b.ledger.events(event_type="aggregate"))},
    "round_history": [{k: v for k, v in h.items() if k != "contributions"} | {"n_contributions": len(h["contributions"])}
                      for h in runner.history],
}
(out / "fedavg_metrics.json").write_text(json.dumps(res, indent=2))

plt.figure(figsize=(6, 4))
xs = sorted(probe_by_round)
plt.plot(xs, [probe_by_round[k] for k in xs], "o-", label="FedAvg global encoder", color="#2F5496")
plt.axhline(res["local_only_mean"], ls="--", color="#C55A11", label=f"local-only mean ({res['local_only_mean']:.3f})")
plt.axhline(res["baseline_frozen_probe_acc"], ls=":", color="#777", label="frozen, no learning")
plt.xlabel("federated round"); plt.ylabel("linear-probe accuracy (held-out)")
plt.title(f"FedAvg vs local-only ({ROUNDS}x{LOCAL_EPOCHS} = {ROUNDS*LOCAL_EPOCHS} local epochs)")
plt.legend(); plt.tight_layout(); plt.savefig(out / "fedavg_vs_local.png", dpi=130); plt.close()

print(json.dumps({k: res[k] for k in ("local_only_probe_acc", "local_only_mean", "federated_final_probe_acc")}, indent=1))
print("dataflow:", {k: flow[k] for k in ("bytes_up_total", "raw_image_bytes_held_locally", "raw_image_bytes_sent")},
      "| guard:", {k: flow["guard"][k] for k in ("payloads_accepted", "payloads_rejected")})
