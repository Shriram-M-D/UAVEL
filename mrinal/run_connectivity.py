"""F4 demo: intermittent connectivity - dropped and delayed (stale) clients, all logged in the round history.
Short run (1 local epoch per round); shows logging, not accuracy.  Run from mrinal/:  python run_connectivity.py
"""
import json, os, sys
from pathlib import Path

os.environ["UAV_NO_APP"] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

from uav_federation import FederationRunner  # noqa: E402
from uav_backend.config import load_config  # noqa: E402
from uav_backend.core import Backend  # noqa: E402

torch.set_num_threads(8)
cfg = load_config(HERE / "configs" / "federation.yaml")
cfg.backend.db_path, cfg.backend.artefact_dir = "mrinal/runs/conn.sqlite", "mrinal/runs/conn_artefacts"
b = Backend(cfg)
b.reset()
r = FederationRunner(b, local_epochs=1, dropout_prob=0.25, delay_prob=0.25, seed=3)
rows = []
for _ in range(5):
    h = r.run_round()
    row = {k: h[k] for k in ("round_id", "model_version_in", "model_version_out", "participants",
                             "dropped_clients", "delayed_clients", "stale_updates", "bytes_up")}
    rows.append(row)
    print(row, flush=True)
agg = b.ledger.events(event_type="aggregate")
out = {"dropout_prob": 0.25, "delay_prob": 0.25, "rounds": rows,
       "ledger_aggregate_zone_sets": [sorted({k.zone_id for k in e.keys}) for e in agg]}
(HERE / "results" / "connectivity_demo.json").write_text(json.dumps(out, indent=2))
print("zones per aggregate in ledger:", out["ledger_aggregate_zone_sets"])
