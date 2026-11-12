"""B3: experiment configuration. One YAML file drives every run; each run gets a run_id and a frozen snapshot."""
from __future__ import annotations

import hashlib, json, time
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]


class DatasetCfg(BaseModel):
    name: str = "eurosat_subset"
    manifest_path: str = "shreyas/results/manifest.csv"  # relative to repo root
    zones_are_synthetic: bool = True


class FederationCfg(BaseModel):
    enabled: bool = False
    rounds: int = 5
    clients_per_round: int = 4
    dropout_prob: float = 0.0
    shared_param_prefixes: list[str] = Field(default_factory=lambda: ["embed.", "pos", "blocks.", "norm."])


class DPCfg(BaseModel):
    enabled: bool = False
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    delta: float = 1e-5


class ResourcePolicyCfg(BaseModel):
    enabled: bool = False  # optional, simulated only; never reported as energy


class BackendCfg(BaseModel):
    db_path: str = "shriram/runs/backend.sqlite"
    artefact_dir: str = "shriram/runs/artefacts"


class PluginsCfg(BaseModel):
    """'module:Class' strings. Empty -> built-in stub, so nobody is blocked."""
    learner: Optional[str] = None
    federation: Optional[str] = None
    unlearner: Optional[str] = None
    verifier: Optional[str] = None


class ExperimentConfig(BaseModel):
    run_name: str = "dev"
    seed: int = 0
    n_clients: int = 4
    n_zones: int = 8
    n_time_buckets: int = 4
    dataset: DatasetCfg = DatasetCfg()
    federation: FederationCfg = FederationCfg()
    dp: DPCfg = DPCfg()
    resource_policy: ResourcePolicyCfg = ResourcePolicyCfg()
    backend: BackendCfg = BackendCfg()
    plugins: PluginsCfg = PluginsCfg()

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    data = yaml.safe_load(Path(path).read_text()) or {}
    return ExperimentConfig.model_validate(data)


def run_id_for(cfg: ExperimentConfig) -> str:
    h = hashlib.sha256(json.dumps(cfg.model_dump(), sort_keys=True).encode()).hexdigest()[:8]
    return f"{cfg.run_name}-{time.strftime('%Y%m%d-%H%M%S')}-{h}"


def snapshot(cfg: ExperimentConfig, out_dir: str | Path) -> str:
    """Write the exact config used by a run next to its outputs. Returns the run_id."""
    rid = run_id_for(cfg)
    d = Path(out_dir) / rid
    d.mkdir(parents=True, exist_ok=True)
    (d / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
    return rid
