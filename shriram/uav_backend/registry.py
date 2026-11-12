"""B5: versioned state registry.

Indexes every stored piece of model state (shared-encoder versions, per-zone adapters, memory partitions,
checkpoints, derived artefacts) with its parents, forming a DAG. Erasure uses the DAG to find everything
derived from erased state. Status changes are the only mutation; files are deleted only by purge().
"""
from __future__ import annotations

import json, sqlite3, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

KINDS = {"shared_encoder", "adapter", "memory", "checkpoint", "derived"}
STATUSES = {"active", "stopped", "purged", "invalidated", "superseded"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS artefacts (
  artefact_id TEXT PRIMARY KEY, kind TEXT NOT NULL, client_id TEXT, zone_id INTEGER, time_bucket INTEGER,
  version TEXT, path TEXT, status TEXT NOT NULL, created_ts REAL NOT NULL, meta TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS artefact_parents (child TEXT NOT NULL, parent TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_parents_parent ON artefact_parents(parent);
"""


@dataclass
class Artefact:
    artefact_id: str
    kind: str
    client_id: Optional[str] = None
    zone_id: Optional[int] = None
    time_bucket: Optional[int] = None
    version: Optional[str] = None
    path: Optional[str] = None
    status: str = "active"
    parents: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    created_ts: Optional[float] = None


class Registry:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        conn.executescript(SCHEMA)

    def register(self, a: Artefact) -> Artefact:
        if a.kind not in KINDS:
            raise ValueError(f"unknown kind {a.kind!r}")
        for p in a.parents:
            if self.get(p) is None:
                raise KeyError(f"parent {p!r} is not registered")
        a.created_ts = a.created_ts or time.time()
        with self.conn:
            self.conn.execute("INSERT INTO artefacts VALUES (?,?,?,?,?,?,?,?,?,?)",
                              (a.artefact_id, a.kind, a.client_id, a.zone_id, a.time_bucket, a.version,
                               a.path, a.status, a.created_ts, json.dumps(a.meta)))
            self.conn.executemany("INSERT INTO artefact_parents VALUES (?,?)", [(a.artefact_id, p) for p in a.parents])
        return a

    def _row(self, r) -> Artefact:
        parents = [p for (p,) in self.conn.execute("SELECT parent FROM artefact_parents WHERE child=?", (r[0],))]
        return Artefact(artefact_id=r[0], kind=r[1], client_id=r[2], zone_id=r[3], time_bucket=r[4], version=r[5],
                        path=r[6], status=r[7], created_ts=r[8], meta=json.loads(r[9]), parents=parents)

    def get(self, artefact_id: str) -> Optional[Artefact]:
        r = self.conn.execute("SELECT * FROM artefacts WHERE artefact_id=?", (artefact_id,)).fetchone()
        return self._row(r) if r else None

    def list(self, kind: Optional[str] = None, status: Optional[str] = None,
             client_id: Optional[str] = None) -> list[Artefact]:
        q, args = "SELECT * FROM artefacts WHERE 1=1", []
        for col, val in (("kind", kind), ("status", status), ("client_id", client_id)):
            if val is not None:
                q += f" AND {col}=?"; args.append(val)
        return [self._row(r) for r in self.conn.execute(q + " ORDER BY created_ts, rowid", args)]

    def descendants(self, ids: Iterable[str]) -> set[str]:
        """Every artefact derived (transitively) from any of `ids`, excluding the ids themselves."""
        seen, frontier = set(), list(ids)
        start = set(frontier)
        while frontier:
            cur = frontier.pop()
            for (child,) in self.conn.execute("SELECT child FROM artefact_parents WHERE parent=?", (cur,)):
                if child not in seen:
                    seen.add(child); frontier.append(child)
        return seen - start

    def set_status(self, ids: Iterable[str], status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        with self.conn:
            self.conn.executemany("UPDATE artefacts SET status=? WHERE artefact_id=?", [(status, i) for i in ids])

    def purge(self, ids: Iterable[str]) -> list[str]:
        """Delete the stored files and mark purged. Returns paths actually deleted."""
        deleted = []
        for i in ids:
            a = self.get(i)
            if a and a.path and Path(a.path).exists():
                Path(a.path).unlink()
                deleted.append(a.path)
        self.set_status(ids, "purged")
        return deleted

    def latest_shared_encoder(self) -> Optional[Artefact]:
        r = self.conn.execute("SELECT * FROM artefacts WHERE kind='shared_encoder' AND status='active' "
                              "ORDER BY created_ts DESC, rowid DESC LIMIT 1").fetchone()
        return self._row(r) if r else None

    def active_model(self, client_id: str) -> dict:
        """What inference for a client is allowed to use: latest active encoder + its active zone state."""
        enc = self.latest_shared_encoder()
        ads = [a for a in self.list(kind="adapter", status="active", client_id=client_id)]
        mem = [a for a in self.list(kind="memory", status="active", client_id=client_id)]
        return {"client_id": client_id,
                "shared_encoder": enc.artefact_id if enc else None,
                "adapters": {a.zone_id: a.artefact_id for a in ads},
                "memory": [a.artefact_id for a in mem]}
