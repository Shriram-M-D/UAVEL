"""Backend context: one object that owns the DB, ledger, registry, guard, config and plugins.

Teammates' modules receive this object and call the record_* hooks (PRD A9 / F6) instead of touching
the database directly.
"""
from __future__ import annotations

import csv, json, shutil, sqlite3, time
from pathlib import Path
from typing import Optional

from .config import ExperimentConfig, load_config
from .guard import Guard
from .ledger import Event, Key, Ledger
from .registry import Artefact, Registry

METRICS_SCHEMA = "CREATE TABLE IF NOT EXISTS metrics (key TEXT PRIMARY KEY, ts REAL, data TEXT NOT NULL);"


class Backend:
    def __init__(self, cfg: ExperimentConfig | str | Path | None = None):
        self.cfg = cfg if isinstance(cfg, ExperimentConfig) else load_config(cfg)
        self.db_path = self.cfg.path(self.cfg.backend.db_path)
        self.artefact_dir = self.cfg.path(self.cfg.backend.artefact_dir)
        self._open()
        from .plugins import load_plugins  # local import: plugins need a Backend
        self.plugins = load_plugins(self)

    def _open(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artefact_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(METRICS_SCHEMA)
        self.ledger = Ledger(self.conn)
        self.registry = Registry(self.conn)
        self.guard = Guard(self.conn, self.cfg.federation.shared_param_prefixes)
        from .erasure import ErasureWorkflow
        self.erasure = ErasureWorkflow(self)

    def reset(self) -> None:
        """Wipe this backend's own DB and artefact directory only (POST /reset, re-run demo)."""
        self.conn.close()
        self.db_path.unlink(missing_ok=True)
        shutil.rmtree(self.artefact_dir, ignore_errors=True)
        self._open()
        from .plugins import load_plugins
        self.plugins = load_plugins(self)

    # ---- hooks for teammates -------------------------------------------------------------------
    def record(self, event: Event, produced: Optional[list[Artefact]] = None) -> Event:
        """Register produced artefacts, then append the event that produced them."""
        for a in produced or []:
            self.registry.register(a)
        event.artefact_ids = list(dict.fromkeys(event.artefact_ids + [a.artefact_id for a in produced or []]))
        return self.ledger.append(event)

    def record_train(self, client_id: str, zone_id: int, time_bucket: Optional[int], produced: list[Artefact],
                     touched_shared_encoder: bool, model_version: Optional[str] = None,
                     parent_version: Optional[str] = None, payload: Optional[dict] = None) -> Event:
        return self.record(Event("train", keys=[Key(client_id, zone_id, time_bucket)], client_id=client_id,
                                 model_version=model_version, parent_version=parent_version,
                                 touched_shared_encoder=touched_shared_encoder, payload=payload or {}), produced)

    def record_aggregate(self, round_id: str, contributions: list[Key], new_encoder: Artefact,
                         parent_version: Optional[str], payload: Optional[dict] = None) -> Event:
        return self.record(Event("aggregate", keys=contributions, model_version=new_encoder.version,
                                 parent_version=parent_version, touched_shared_encoder=True,
                                 payload={"round_id": round_id, **(payload or {})}), [new_encoder])

    def record_memory_write(self, client_id: str, zone_id: int, time_bucket: Optional[int],
                            memory: Artefact, payload: Optional[dict] = None) -> Event:
        return self.record(Event("memory_write", keys=[Key(client_id, zone_id, time_bucket)], client_id=client_id,
                                 payload=payload or {}), [memory])

    # ---- metrics ---------------------------------------------------------------------------------
    def save_metrics(self, key: str, data: dict) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO metrics VALUES (?,?,?)", (key, time.time(), json.dumps(data)))

    def get_metrics(self, key: str) -> Optional[dict]:
        r = self.conn.execute("SELECT data FROM metrics WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else None

    # ---- clients ---------------------------------------------------------------------------------
    def clients(self) -> list[dict]:
        """Clients and their zone/time partitions: from Shreyas's manifest if present, else from config."""
        mpath = self.cfg.path(self.cfg.dataset.manifest_path)
        parts: dict[str, dict] = {}
        if mpath.exists():
            with open(mpath, newline="") as f:
                for r in csv.DictReader(f):
                    c = parts.setdefault(r["client_id"], {"zones": {}, "n_samples": 0})
                    z = c["zones"].setdefault(int(r["zone_id"]), {})
                    tb = int(r["time_bucket"])
                    z[tb] = z.get(tb, 0) + 1
                    c["n_samples"] += 1
            source = "manifest"
        else:
            for z in range(self.cfg.n_zones):
                c = parts.setdefault(f"uav{z % self.cfg.n_clients}", {"zones": {}, "n_samples": 0})
                c["zones"][z] = {tb: 0 for tb in range(self.cfg.n_time_buckets)}
            source = "config"
        out = []
        for cid in sorted(parts):
            c = parts[cid]
            out.append({"client_id": cid, "n_samples": c["n_samples"], "source": source,
                        "zones_are_synthetic": self.cfg.dataset.zones_are_synthetic,
                        "partitions": [{"zone_id": z, "time_bucket": tb, "n_samples": n}
                                       for z in sorted(c["zones"]) for tb, n in sorted(c["zones"][z].items())]})
        return out
