import os, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("UAV_NO_APP", "1")  # don't build the module-level app on import

from uav_backend.config import ExperimentConfig  # noqa: E402
from uav_backend.core import Backend  # noqa: E402


@pytest.fixture
def backend(tmp_path):
    cfg = ExperimentConfig()
    cfg.backend.db_path = str(tmp_path / "b.sqlite")
    cfg.backend.artefact_dir = str(tmp_path / "artefacts")
    cfg.dataset.manifest_path = str(tmp_path / "no_manifest.csv")  # use config-derived clients
    return Backend(cfg)
