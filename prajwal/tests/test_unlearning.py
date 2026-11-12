import os, sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("UAV_NO_APP", "1")

from uav_unlearning.strategies import Mission, run_schedule, reference, rollback_replay, finetune_retained  # noqa: E402
from uav_unlearning.verify import mia_auc, representation_distance, zone_knowledge_gap  # noqa: E402
from uav_unlearning.plugin import CheckpointReplayUnlearner  # noqa: E402
from uav_backend import ErasureRequest  # noqa: E402
from uav_backend.config import ExperimentConfig  # noqa: E402
from uav_backend.core import Backend  # noqa: E402
from src.models.mae import MAE  # noqa: E402


def tiny():
    torch.manual_seed(0)
    return MAE(depth=1, ddepth=1)


def test_rollback_replay_equals_reference_exactly():
    """Exact unlearning: replay from the last clean checkpoint reproduces the retrain reference bit-for-bit."""
    init = tiny()
    x1, x2 = torch.randn(8, 3, 64, 64), torch.randn(8, 3, 64, 64)
    missions = [Mission(x1, 2, 1), Mission(x2, 2, 2, contains_target=True)]
    keep = [np.ones(8, bool), np.r_[np.zeros(4, bool), np.ones(4, bool)]]
    M = run_schedule(init, missions)
    R = reference(init, missions, keep)
    RR = rollback_replay(init, M, missions, keep)
    for k, v in R.model.state_dict().items():
        assert torch.equal(v, RR.model.state_dict()[k]), k
    assert not all(torch.equal(v, M.model.state_dict()[k]) for k, v in R.model.state_dict().items())


def test_finetune_retained_changes_model_and_keeps_original():
    init = tiny()
    M = run_schedule(init, [Mission(torch.randn(8, 3, 64, 64), 1, 1, True)])
    before = {k: v.clone() for k, v in M.model.state_dict().items()}
    F = finetune_retained(M, torch.randn(4, 3, 64, 64), 1, 3)
    assert all(torch.equal(v, M.model.state_dict()[k]) for k, v in before.items())
    assert any(not torch.equal(v, F.model.state_dict()[k]) for k, v in before.items())


def test_mia_detects_memorised_members():
    """Sanity check of the attack: a model overfit on 'members' must score AUC well above 0.5."""
    torch.manual_seed(0)
    members, non = torch.randn(8, 3, 64, 64), torch.randn(8, 3, 64, 64)
    M = run_schedule(tiny(), [Mission(members, 300, 1)])  # 300 steps on 8 images: memorised
    r = mia_auc(M.model, members, non, n_boot=100)
    assert r["auc"] > 0.8 and r["member_loss"] < r["nonmember_loss"]


def test_zone_knowledge_gap_positive_when_zone_was_learned():
    torch.manual_seed(0)
    zone = torch.randn(1, 3, 64, 64).repeat(24, 1, 1, 1) + 0.1 * torch.randn(24, 3, 64, 64)  # one "zone" look
    init = tiny()
    M = run_schedule(init, [Mission(zone[:16], 60, 1)])
    R = run_schedule(init, [Mission(torch.randn(16, 3, 64, 64), 60, 1)])
    assert zone_knowledge_gap(M.model, R.model, zone[16:]) > 0      # unseen images of the same zone
    assert zone_knowledge_gap(R.model, R.model, zone[16:]) == 0


def test_identical_models_have_zero_distance():
    m = tiny()
    d = representation_distance(m, m, torch.randn(10, 3, 64, 64))
    assert d["mean_cosine_distance"] < 1e-6 and d["linear_cka"] == pytest.approx(1.0, abs=1e-5)


@pytest.fixture
def backend(tmp_path):
    cfg = ExperimentConfig()
    cfg.backend.db_path = str(tmp_path / "b.sqlite")
    cfg.backend.artefact_dir = str(tmp_path / "a")
    cfg.dataset.manifest_path = str(tmp_path / "none.csv")
    cfg.plugins.unlearner = "uav_unlearning.plugin:CheckpointReplayUnlearner"
    return Backend(cfg)


def test_plugin_in_backend_workflow_reports_outstanding_replay(backend):
    assert backend.plugins.sources["unlearner"] == "uav_unlearning.plugin:CheckpointReplayUnlearner"
    L = backend.plugins.learner                   # Shriram's stub learner builds the ledger
    L.train("uav1", 1, 0)                         # enc-v2 clean
    L.train("uav0", 0, 0)                         # enc-v3 target
    L.train("uav1", 5, 0)                         # enc-v4 retained work after target
    req = ErasureRequest(zone_id=0)
    backend.erasure.submit(req)
    rep = backend.erasure.run(req.request_id)
    rp = rep["steps"]["REPAIRED"]
    assert rp["rolled_back_to"] == "enc-v2" and not rp["replayed"]
    assert {"client_id": "uav1", "zone_id": 5, "time_bucket": 0} in rp["retained_to_replay"]
    assert all(r["zone_id"] != 0 for r in rp["retained_to_replay"])
    assert any("not yet replayed" in n for n in rep["not_guaranteed"])
    assert rep["steps"]["PURGED"]["files_still_present"] == []


def test_plugin_runs_replay_when_attached(backend):
    calls = []
    backend.plugins.unlearner.replay = lambda base, retained: calls.append((base.artefact_id, retained)) or "replayed.pt"
    backend.plugins.learner.train("uav0", 0, 0)
    backend.plugins.learner.train("uav1", 5, 0)
    req = ErasureRequest(zone_id=0)
    backend.erasure.submit(req)
    rep = backend.erasure.run(req.request_id)
    assert calls and calls[0][0] == "enc-v1" and rep["steps"]["REPAIRED"]["replayed"]
    assert not any("not yet replayed" in n for n in rep["not_guaranteed"])
