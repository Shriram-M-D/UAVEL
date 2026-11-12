"""B8: plug-in interfaces and loader.

Teammates implement these interfaces in their own folders and point configs/experiment.yaml at them
('module:Class'). Every plug-in is constructed as Class(backend). Until a real one exists, the stub in
stubs.py is used, so the API, frontend and demo are never blocked.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Optional, Protocol


class LocalLearner(Protocol):  # Shreyas
    def train(self, client_id: str, zone_id: int, time_bucket: Optional[int] = None,
              adapter_only: bool = False, steps: int = 1) -> dict: ...


class FederationRunner(Protocol):  # Mrinal
    def run_round(self, client_ids: Optional[list[str]] = None) -> dict: ...


class Unlearner(Protocol):  # Prajwal
    def purge(self, request: dict, resolution: dict) -> dict: ...
    def repair(self, request: dict, resolution: dict) -> dict: ...


class Verifier(Protocol):  # Prajwal
    def verify(self, request: dict, report: dict) -> dict: ...


@dataclass
class Plugins:
    learner: Any
    federation: Any
    unlearner: Any
    verifier: Any
    sources: dict


def _load(spec: Optional[str], default, backend):
    if not spec:
        return default(backend), f"stub:{default.__name__}"
    mod, _, cls = spec.partition(":")
    return getattr(importlib.import_module(mod), cls)(backend), spec


def load_plugins(backend) -> Plugins:
    from . import stubs
    p = backend.cfg.plugins
    learner, s1 = _load(p.learner, stubs.StubLearner, backend)
    fed, s2 = _load(p.federation, stubs.StubFederation, backend)
    unl, s3 = _load(p.unlearner, stubs.RollbackUnlearner, backend)
    ver, s4 = _load(p.verifier, stubs.StubVerifier, backend)
    return Plugins(learner, fed, unl, ver, {"learner": s1, "federation": s2, "unlearner": s3, "verifier": s4})
