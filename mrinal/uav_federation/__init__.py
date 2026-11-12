"""uav_federation - Mrinal's multi-UAV simulation: clients, FedAvg coordinator, round history, comms evidence."""
from . import _paths  # noqa: F401
from .clients import UAVClient, build_clients, shared_state
from .coordinator import Coordinator, fedavg
from .runner import FederationRunner
from .comms import dataflow_report

__all__ = ["UAVClient", "build_clients", "shared_state", "Coordinator", "fedavg", "FederationRunner",
           "dataflow_report"]
