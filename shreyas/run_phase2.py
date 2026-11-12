"""Phase 2 demo (A6): per-zone adapters on a frozen shared encoder + cross-zone isolation check.
Needs results/*_local_encoder.pt from run_phase1.py.  Run from shreyas/:  python run_phase2.py
"""
import json, copy, torch
from src.data.manifest import ROOT
from src.data.dataset import load_images, to_tensor
from src.models.mae import MAE
from src.learning.local_ssl import train_local
import pandas as pd

torch.set_num_threads(8)
out = ROOT / "results"
df = pd.read_csv(out / "manifest.csv")
X = to_tensor(load_images(df))
CID = "uav0"
base = MAE()
base.load_state_dict(torch.load(out / f"{CID}_local_encoder.pt"))
for p in base.parameters():
    p.requires_grad = False  # shared encoder is frozen: only adapters learn
zones = [int(z) for z in sorted(df[df.client_id == CID].zone_id.unique())]
total = sum(p.numel() for p in base.parameters())


def recon(model, x, zone=None, seed=123):
    model.eval()
    torch.manual_seed(seed)  # same masks for every comparison
    with torch.no_grad():
        return float(model(x, zone)[0])


def zdata(z, split):
    return X[((df.zone_id == z) & (df.split == split)).values]


res = {"client": CID, "shared_encoder_params": total, "zones": {}}
before = {z: recon(base, zdata(z, "test")) for z in zones}
model = copy.deepcopy(base)
ckpt = out / "checkpoints" / "adapters"
ckpt.mkdir(parents=True, exist_ok=True)
for z in zones:
    model.add_adapter(z)
    ad = model.adapters[str(z)]
    print(f"zone {z}: training adapter only ({sum(p.numel() for p in ad.parameters())} params vs {total} total)")
    _, info = train_local(zdata(z, "ssl_train"), epochs=40, model=model, params=ad.parameters(), zone=z, log=lambda *_: None)
    torch.save(ad.state_dict(), ckpt / f"zone{z}.pt")
    res["zones"][z] = dict(adapter_params=info["trainable_params"], train_seconds=round(info["train_seconds"], 1))

for z in zones:
    own = recon(model, zdata(z, "test"), z)
    other = [o for o in zones if o != z][0]
    # cross-zone: use zone z's adapter on the OTHER zone's data vs that zone's frozen baseline
    cross = recon(model, zdata(other, "test"), z)
    res["zones"][z].update(test_recon_frozen=before[z], test_recon_with_adapter=own,
                           other_zone_recon_frozen=before[other], other_zone_recon_using_this_adapter=cross)
    print(f"zone {z}: frozen {before[z]:.4f} -> adapter {own:.4f} | other zone {other}: frozen {before[other]:.4f}, with zone-{z} adapter {cross:.4f}")

# isolation test: deleting zone 0's adapter must not change zone 4's behaviour; unadapted output returns to frozen baseline
z0, z1 = zones[0], zones[1]
x1 = zdata(z1, "test")
r1_before = recon(model, x1, z1)
del model.adapters[str(z0)]
r1_after = recon(model, x1, z1)
r0_after_delete = recon(model, zdata(z0, "test"), z0)  # zone passes through with no adapter
res["isolation_test"] = dict(deleted_adapter_zone=z0, other_zone=z1,
                             other_zone_recon_before_delete=r1_before, other_zone_recon_after_delete=r1_after,
                             unaffected=abs(r1_before - r1_after) < 1e-9,
                             deleted_zone_recon_after_delete=r0_after_delete, deleted_zone_frozen_baseline=before[z0],
                             deleted_zone_matches_frozen=abs(r0_after_delete - before[z0]) < 1e-9)
print("isolation:", json.dumps(res["isolation_test"], indent=1))
json.dump(res, open(out / "phase2_metrics.json", "w"), indent=2)
