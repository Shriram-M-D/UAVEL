"""B10 (partial): end-to-end backend demo with stub ML plug-ins.  Run from shriram/:  python demo_backend.py

Shows: ledger + registry recording local training, a federated round through the locality guard,
and two erasure requests -
  A) a zone trained adapter-only  -> no shared-encoder repair needed, other zones untouched
  B) a zone that updated the shared encoder -> rollback repair, derived state invalidated
"""
import json, os, sys
from pathlib import Path

os.environ["UAV_NO_APP"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from uav_backend import Backend, ErasureRequest, LocalityViolation  # noqa: E402

HERE = Path(__file__).resolve().parent
b = Backend(HERE / "configs" / "experiment.yaml")
b.reset()
L = b.plugins.learner
print("plugins:", b.plugins.sources)

print("\n1) local training (stub learner)")
L.train("uav0", 0, 0, adapter_only=True)   # zone 0: adapter-only
L.train("uav0", 4, 1, adapter_only=True)   # zone 4: adapter-only
L.train("uav1", 1, 0)                       # zone 1: updates shared encoder
L.train("uav2", 2, 2)                       # zone 2: updates shared encoder
L.train("uav3", 3, 0)                       # zone 3: updates shared encoder
print("   ledger events:", b.ledger.count(), "| artefacts:", len(b.registry.list()))

print("\n2) federated round through the locality guard")
rd = b.plugins.federation.run_round()
print(f"   {rd['round_id']}: {rd['model_version_in']} -> {rd['model_version_out']}, bytes up = {rd['bytes_up']:,}")
try:
    b.guard.check_update({"client_id": "uav0", "round_id": "x", "images": ["raw.jpg"]})
except LocalityViolation as e:
    print("   blocked a payload carrying images:", e)
ev = b.guard.evidence()
print("   evidence:", {k: ev[k] for k in ("payloads_accepted", "payloads_rejected", "bytes_transmitted")})

out = {"round": rd, "locality_evidence": ev, "erasures": {}}
for name, zone in (("A_adapter_only_zone", 0), ("B_shared_encoder_zone", 2)):
    req = ErasureRequest(zone_id=zone)
    b.erasure.submit(req)
    rep = b.erasure.run(req.request_id)
    s = rep["steps"]
    print(f"\n3{name[0]}) erase zone {zone}: state={rep['state']}")
    print("   influenced events:", len(s["RESOLVED"]["events"]), "| tainted artefacts:", len(s["RESOLVED"]["tainted_artefacts"]))
    print("   purged:", s["PURGED"]["purged"])
    print("   repair:", {k: v for k, v in s["REPAIRED"].items() if k in ("required", "rolled_back_to", "new_artefact_ids", "reason")})
    print("   invalidated:", s["INVALIDATED"]["invalidated"])
    print("   not guaranteed:", rep["not_guaranteed"])
    out["erasures"][name] = rep
for c in ("uav0", "uav2"):
    print(f"\nactive model {c}:", b.registry.active_model(c))

res = HERE / "results"
res.mkdir(exist_ok=True)
(res / "demo_backend_report.json").write_text(json.dumps(out, indent=2, default=str))
print("\nwrote results/demo_backend_report.json | total ledger events:", b.ledger.count())
