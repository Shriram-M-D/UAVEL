import sqlite3
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from uav_backend import Event, Key, LocalityViolation, ErasureRequest, Artefact
from uav_backend.api import create_app
from uav_backend.config import load_config, snapshot


# ---- B3 config -----------------------------------------------------------------------------------
def test_config_loads_repo_yaml_and_snapshots(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml")
    rid = snapshot(cfg, tmp_path)
    assert (tmp_path / rid / "config_snapshot.yaml").exists()


# ---- B4 ledger -----------------------------------------------------------------------------------
def test_ledger_is_append_only(backend):
    backend.ledger.append(Event("train", keys=[Key("uav0", 0, 1)]))
    with pytest.raises(sqlite3.IntegrityError):
        with backend.conn:
            backend.conn.execute("DELETE FROM events")
    with pytest.raises(sqlite3.IntegrityError):
        with backend.conn:
            backend.conn.execute("UPDATE events SET event_type='verify'")


def test_ledger_rejects_unknown_event_type(backend):
    with pytest.raises(ValueError):
        backend.ledger.append(Event("teleport"))


def test_ledger_targeted_lookup_by_zone_and_time(backend):
    L = backend.ledger
    a = L.append(Event("train", keys=[Key("uav0", 0, 1)]))
    L.append(Event("train", keys=[Key("uav0", 0, 2)]))
    L.append(Event("train", keys=[Key("uav1", 1, 1)]))
    agg = L.append(Event("aggregate", keys=[Key("uav0", 0, 1), Key("uav1", 1, 1)]))
    assert {e.event_id for e in L.influenced_by(0, [1])} == {a.event_id, agg.event_id}
    assert len(L.influenced_by(0)) == 3
    assert {e.event_id for e in L.influenced_by(1)} == {L.events()[2].event_id, agg.event_id}
    assert L.influenced_by(5) == []


# ---- B5 registry ---------------------------------------------------------------------------------
def test_registry_purge_leaves_other_partitions_byte_identical(backend, tmp_path):
    R = backend.registry
    files = {}
    for z in (0, 1):
        p = tmp_path / f"adapter_{z}.bin"
        p.write_bytes(np.random.default_rng(z).bytes(1024))
        files[z] = p
        R.register(Artefact(f"ad{z}", "adapter", client_id="uav0", zone_id=z, path=str(p)))
    before = files[1].read_bytes()
    R.purge(["ad0"])
    assert not files[0].exists() and R.get("ad0").status == "purged"
    assert files[1].read_bytes() == before and R.get("ad1").status == "active"


def test_registry_descendants(backend):
    R = backend.registry
    R.register(Artefact("e1", "shared_encoder"))
    R.register(Artefact("e2", "shared_encoder", parents=["e1"]))
    R.register(Artefact("a", "adapter", parents=["e2"]))
    R.register(Artefact("m", "memory", parents=["a"]))
    R.register(Artefact("x", "adapter", parents=["e1"]))
    assert R.descendants(["e2"]) == {"a", "m"}
    assert R.descendants(["e1"]) == {"e2", "a", "m", "x"}


# ---- B9 guard ------------------------------------------------------------------------------------
def _upd(**kw):
    p = {"client_id": "uav0", "round_id": "r1", "state_dict": {"blocks.0.w": np.zeros((128, 128), np.float32)}}
    p.update(kw)
    return p


def test_guard_accepts_shared_params_and_counts_bytes(backend):
    assert backend.guard.check_update(_upd()) == 128 * 128 * 4


@pytest.mark.parametrize("bad", [
    {"images": [1, 2]},                                                        # unknown field
    {"metrics": {"preview": np.zeros((64, 64, 3))}},                           # array in metrics
    {"state_dict": {"adapters.0.down.weight": np.zeros((16, 128), np.float32)}},  # adapter must stay local
    {"state_dict": {"blocks.x": np.zeros((8, 3, 64, 64), np.float32)}},         # image batch disguised
    {"state_dict": {"blocks.x": np.zeros((10, 10), np.uint8)}},                 # pixel dtype
    {"num_samples": b"\x89PNG"},                                               # raw bytes
])
def test_guard_rejects_raw_data(backend, bad):
    with pytest.raises(LocalityViolation):
        backend.guard.check_update(_upd(**bad))
    assert backend.guard.evidence()["payloads_rejected"] == 1


# ---- B7 erasure workflow -------------------------------------------------------------------------
def test_erasure_adapter_only_zone_leaves_other_zones_and_encoder(backend):
    L = backend.plugins.learner
    L.train("uav0", 0, 0, adapter_only=True)
    L.train("uav0", 4, 0, adapter_only=True)
    enc_before = backend.registry.latest_shared_encoder().artefact_id
    req = ErasureRequest(zone_id=0)
    backend.erasure.submit(req)
    rep = backend.erasure.run(req.request_id)
    assert rep["state"] == "VERIFIED" and rep["complete"]
    assert rep["steps"]["REPAIRED"]["required"] is False
    am = backend.registry.active_model("uav0")
    assert 0 not in am["adapters"] and 4 in am["adapters"]
    assert am["shared_encoder"] == enc_before


def test_erasure_when_shared_encoder_was_trained_requires_repair(backend):
    L = backend.plugins.learner
    L.train("uav1", 1, 0)            # enc-v2 (clean, from zone 1)
    L.train("uav0", 0, 0)            # enc-v3 (zone 0 -> target)
    L.train("uav1", 5, 0)            # enc-v4 (derived from tainted v3)
    req = ErasureRequest(zone_id=0)
    backend.erasure.submit(req)
    rep = backend.erasure.run(req.request_id)
    s = rep["steps"]
    assert s["RESOLVED"]["shared_encoder_affected"]
    assert set(s["RESOLVED"]["tainted_shared_encoders"]) == {"enc-v3", "enc-v4"}
    assert s["REPAIRED"]["rolled_back_to"] == "enc-v2"
    new_enc = backend.registry.latest_shared_encoder()
    assert new_enc.parents == ["enc-v2"]
    # zone 5's adapter was built on a tainted encoder -> invalidated, not silently kept
    assert any("uav1-adapter-z5" in a for a in s["INVALIDATED"]["invalidated"])
    assert any("rollback" in n for n in rep["not_guaranteed"])


def test_erasure_is_resumable_and_idempotent(backend):
    backend.plugins.learner.train("uav0", 0, 0)
    req = ErasureRequest(zone_id=0)
    backend.erasure.submit(req)
    assert backend.erasure.run(req.request_id, until="STOPPED")["state"] == "STOPPED"
    assert backend.erasure.run(req.request_id)["state"] == "VERIFIED"
    n = backend.ledger.count()
    backend.erasure.submit(req)
    backend.erasure.run(req.request_id)
    assert backend.ledger.count() == n


def test_adapter_only_zone_is_not_a_federation_contribution(backend):
    backend.plugins.learner.train("uav0", 0, 0, adapter_only=True)
    backend.plugins.learner.train("uav1", 1, 0)
    rd = backend.plugins.federation.run_round()
    assert {c["zone_id"] for c in rd["contributions"]} == {1}
    req = ErasureRequest(zone_id=0)
    backend.erasure.submit(req)
    assert backend.erasure.run(req.request_id)["steps"]["REPAIRED"]["required"] is False


# ---- B6 API (Phase 1 exit criterion)---------------------------------------------------------------
def test_api_stub_training_shows_up_in_ledger(backend):
    c = TestClient(create_app(backend))
    clients = c.get("/clients").json()
    assert len(clients) == 4
    r = c.post("/clients/uav0/train", json={"zone_id": 0, "time_bucket": 1})
    assert r.status_code == 200
    events = c.get("/ledger", params={"event_type": "train"}).json()
    assert events and events[0]["keys"][0]["zone_id"] == 0
    rd = c.post("/federation/rounds", json={}).json()
    assert rd["bytes_up"] > 0 and c.get("/federation/evidence").json()["payloads_accepted"] == 4
    er = c.post("/erasure-requests", json={"zone_id": 0}).json()
    assert er["state"] == "VERIFIED"
    assert c.get(f"/erasure-requests/{er['request']['request_id']}").json()["complete"]
    assert c.post("/clients/nope/train", json={"zone_id": 0}).status_code == 404
