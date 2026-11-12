"""U2-U4: training schedules, the retrain-without-target reference, and unlearning strategies.

A 'schedule' is a list of missions [(x, epochs, seed), ...] run in order from a fixed initialisation;
a checkpoint is kept after every mission. This mirrors repeated UAV missions and lets us compare:

  purge_only        - delete zone adapter/memory, leave the shared encoder untouched (cheap, inexact)
  rollback_replay   - restore the last checkpoint before the target data entered, replay later missions
                      without the target (exact w.r.t. the schedule, partial cost)  [SISA-style idea]
  finetune_retained - keep training the current model on retained data only (cheap, approximate)
  reference         - retrain from the initialisation on the whole schedule minus the target (gold standard)
"""
from __future__ import annotations

import copy, time
from dataclasses import dataclass, field

import torch

from . import _paths  # noqa: F401
from src.learning.local_ssl import train_local  # Shreyas


@dataclass
class Mission:
    x: torch.Tensor
    epochs: int
    seed: int
    contains_target: bool = False


@dataclass
class Trained:
    model: torch.nn.Module
    checkpoints: list = field(default_factory=list)  # state_dict after each mission
    seconds: float = 0.0


def _quiet(*_):
    pass


def run_schedule(init: torch.nn.Module, missions: list[Mission]) -> Trained:
    m = copy.deepcopy(init)
    t0, cps = time.time(), []
    for mi in missions:
        if len(mi.x):
            train_local(mi.x, epochs=mi.epochs, model=m, seed=mi.seed, log=_quiet)
        cps.append({k: v.clone() for k, v in m.state_dict().items()})
    return Trained(m, cps, time.time() - t0)


def without_target(missions: list[Mission], keep: list) -> list[Mission]:
    """Same schedule with target samples removed. keep[i] = boolean mask over missions[i].x."""
    assert len(keep) == len(missions)
    return [Mission(mi.x[torch.as_tensor(k)], mi.epochs, mi.seed, False) for mi, k in zip(missions, keep)]


def reference(init, missions, keep) -> Trained:
    return run_schedule(init, without_target(missions, keep))


def purge_only(trained: Trained) -> Trained:
    return Trained(copy.deepcopy(trained.model), [], 0.0)


def rollback_replay(init, trained: Trained, missions, keep) -> Trained:
    first = next((i for i, mi in enumerate(missions) if mi.contains_target), None)
    if first is None:
        return purge_only(trained)
    t0 = time.time()
    m = copy.deepcopy(init)
    if first > 0:
        m.load_state_dict(trained.checkpoints[first - 1])
    replay = run_schedule(m, without_target(missions[first:], keep[first:]))
    return Trained(replay.model, replay.checkpoints, time.time() - t0)


def finetune_retained(trained: Trained, x_retained: torch.Tensor, epochs: int, seed: int) -> Trained:
    t0 = time.time()
    m = copy.deepcopy(trained.model)
    train_local(x_retained, epochs=epochs, model=m, seed=seed, log=_quiet)
    return Trained(m, [], time.time() - t0)
