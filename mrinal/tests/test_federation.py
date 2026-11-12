import os, sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("UAV_NO_APP", "1")

from uav_federation import FederationRunner, build_clients, fedavg  # noqa: E402
from uav_backend import ErasureRequest, LocalityViolation  # noqa: E402
from uav_backend.config import ExperimentConfig  # noqa: E402
from uav_backend.core import Backend  # noqa: E402
from src.models.mae import MAE  # noqa: E402

PREFIXES = ["embed.", "pos", "blocks.", "norm."]


@pytest.fixture
def backend(tmp_path):
    cfg = ExperimentConfig()
    cfg.backend.db_path = str(tmp_path / "b.sqlite")
    cfg.backend.artefact_dir = str(tmp_path / "art")
    cfg.dataset.manifest_path = str(tmp_path / "none.csv")
    return Backend(cfg)


def tiny_setup(backend, n_per=6):
    """3 clients, 2 zones each, random 64x64 images - fast and needs no dataset."""
    rows, k = [], 0
    for c in range(3):
        for z in (c, c + 3):
            for t in range(n_per // 2):
                rows.append({"sample_id": f"s{k}", "client_id": f"uav{c}", "zone_id": z, "time_bucket": t % 2,
                             "split": "ssl_train"})
                k += 1
    x = torch.randn(len(rows), 3, 64, 64)
    torch.manual_seed(0)
    model = MAE(depth=1, ddepth=1)
    return rows, x, model, build_clients(rows, x, model, PREFIXES, backend)


def test_clients_hold_only_their_own_data(backend):
    rows, x, _, clients = tiny_setup(backend)
    seen = set()
    for cid, c in clients.items():
        assert c.sample_ids == {r["sample_id"] for r in rows if r["client_id"] == cid}
        assert not (c.sample_ids & seen)
        seen |= c.sample_ids
    # pixel copies, not views into a shared tensor
    c0 = clients["uav0"]
    c0.x.zero_()
    assert x.abs().sum() > 0


def test_fedavg_is_sample_weighted():
    u = [{"client_id": "a", "num_samples": 1, "state_dict": {"w": torch.tensor([0.0])}},
         {"client_id": "b", "num_samples": 3, "state_dict": {"w": torch.tensor([4.0])}}]
    assert torch.allclose(fedavg(u)["w"], torch.tensor([3.0]))
    with pytest.raises(ValueError):
        fedavg([u[0], {"client_id": "c", "num_samples": 1, "state_dict": {"v": torch.tensor([1.0])}}])


def test_updates_contain_only_shared_params_and_pass_guard(backend):
    _, _, model, clients = tiny_setup(backend)
    model.add_adapter(0)
    r = FederationRunner(backend, clients=clients, model=model)
    h = r.run_round()
    assert h["participants"] == ["uav0", "uav1", "uav2"] and h["bytes_up"] > 0
    upd = clients["uav0"].local_round(r.coord.shared, "x", r.coord.version)
    assert all(k.startswith(tuple(PREFIXES)) for k in upd["state_dict"])
    assert not any(k.startswith(("d_", "mask_tok", "adapters")) for k in upd["state_dict"])
    ev = backend.guard.evidence()
    assert ev["payloads_accepted"] == 3 and ev["payloads_rejected"] == 0


def test_guard_blocks_a_client_that_leaks_its_adapter(backend):
    _, _, model, clients = tiny_setup(backend)
    r = FederationRunner(backend, clients=clients, model=model)
    upd = clients["uav0"].local_round(r.coord.shared, "x", r.coord.version)
    upd["state_dict"]["adapters.0.0.down.weight"] = torch.zeros(16, 128)
    with pytest.raises(LocalityViolation):
        r.coord.receive(upd)


def test_round_history_in_ledger_lists_exact_contributions(backend):
    _, _, model, clients = tiny_setup(backend)
    r = FederationRunner(backend, clients=clients, model=model, dropout_prob=0.0)
    h = r.run_round(["uav0", "uav2"])
    agg = backend.ledger.events(event_type="aggregate")
    assert len(agg) == 1 and agg[0].model_version == h["model_version_out"]
    zones = {k.zone_id for k in agg[0].keys}
    assert zones == {0, 3, 2, 5}                      # uav1's zones 1 and 4 did not contribute
    assert {e.event_id for e in backend.ledger.influenced_by(1)} == set()
    assert backend.registry.get(h["model_version_out"]).parents == [h["model_version_in"]]


def test_dropout_and_delay_are_logged(backend):
    _, _, model, clients = tiny_setup(backend)
    r = FederationRunner(backend, clients=clients, model=model, dropout_prob=1.0)
    h = r.run_round()
    assert h["dropped_clients"] == ["uav0", "uav1", "uav2"] and h["model_version_out"] == h["model_version_in"]
    r = FederationRunner(backend, clients=clients, model=model, dropout_prob=0.0, delay_prob=1.0)
    h1 = r.run_round(["uav0"])
    assert h1["delayed_clients"] == ["uav0"] and not h1["participants"]
    r.delay_prob = 0.0
    h2 = r.run_round(["uav1"])
    assert h2["stale_updates"] == ["uav0"] and set(h2["participants"]) == {"uav0", "uav1"}


def test_stale_update_superseded_by_fresh_one_is_not_double_counted(backend):
    _, _, model, clients = tiny_setup(backend)
    r = FederationRunner(backend, clients=clients, model=model, dropout_prob=0.0, delay_prob=1.0)
    r.run_round(["uav0"])
    r.delay_prob = 0.0
    h = r.run_round(["uav0", "uav1"])
    assert h["participants"] == ["uav0", "uav1"] and h["stale_superseded"] == ["uav0"]
    assert h["stale_updates"] == []


def test_backend_erasure_sees_federated_influence(backend):
    """Integration with Shriram's workflow: a zone that reached the global model forces encoder repair."""
    _, _, model, clients = tiny_setup(backend)
    r = FederationRunner(backend, clients=clients, model=model)
    r.run_round(["uav1"])          # enc-v2: zones 1, 4 only
    r.run_round()                  # enc-v3: all zones
    req = ErasureRequest(zone_id=0)
    backend.erasure.submit(req)
    rep = backend.erasure.run(req.request_id)
    assert rep["steps"]["RESOLVED"]["tainted_shared_encoders"] == ["enc-v3"]
    assert rep["steps"]["REPAIRED"]["rolled_back_to"] == "enc-v2"
