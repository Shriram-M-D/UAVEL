"""B4: append-only provenance ledger (SQLite).

Every training step, aggregation, memory write, checkpoint, erasure step and verification is an event.
An event can carry several (client, zone, time_bucket) keys: a FedAvg round is influenced by many zones.
UPDATE and DELETE on events are blocked by triggers: deleting a ledger row would not undo its
influence on parameters, so the ledger only ever grows.
"""
from __future__ import annotations

import json, sqlite3, time, uuid
from dataclasses import dataclass, field
from typing import Iterable, Optional

EVENT_TYPES = {"train", "aggregate", "memory_write", "checkpoint", "erase", "verify"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY, seq INTEGER NOT NULL, ts REAL NOT NULL, run_id TEXT,
  event_type TEXT NOT NULL, client_id TEXT, model_version TEXT, parent_version TEXT,
  artefact_ids TEXT NOT NULL, touched_shared_encoder INTEGER NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS event_keys (
  event_id TEXT NOT NULL REFERENCES events(event_id), client_id TEXT, zone_id INTEGER, time_bucket INTEGER);
CREATE INDEX IF NOT EXISTS ix_keys_zone ON event_keys(zone_id, time_bucket);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS keys_no_update BEFORE UPDATE ON event_keys
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS keys_no_delete BEFORE DELETE ON event_keys
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
"""


@dataclass
class Key:
    """One (client, zone, time_bucket) unit of data that influenced an event."""
    client_id: Optional[str] = None
    zone_id: Optional[int] = None
    time_bucket: Optional[int] = None


@dataclass
class Event:
    event_type: str
    keys: list[Key] = field(default_factory=list)
    client_id: Optional[str] = None
    model_version: Optional[str] = None
    parent_version: Optional[str] = None
    artefact_ids: list[str] = field(default_factory=list)
    touched_shared_encoder: bool = False
    payload: dict = field(default_factory=dict)
    run_id: Optional[str] = None
    event_id: Optional[str] = None
    ts: Optional[float] = None
    seq: Optional[int] = None


class Ledger:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        conn.executescript(SCHEMA)

    def append(self, ev: Event) -> Event:
        if ev.event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event_type {ev.event_type!r}; allowed: {sorted(EVENT_TYPES)}")
        ev.event_id = ev.event_id or uuid.uuid4().hex[:12]
        ev.ts = ev.ts or time.time()
        with self.conn:
            ev.seq = self.conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM events").fetchone()[0]
            self.conn.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (ev.event_id, ev.seq, ev.ts, ev.run_id, ev.event_type, ev.client_id, ev.model_version,
                 ev.parent_version, json.dumps(ev.artefact_ids), int(ev.touched_shared_encoder), json.dumps(ev.payload)))
            self.conn.executemany(
                "INSERT INTO event_keys VALUES (?,?,?,?)",
                [(ev.event_id, k.client_id, k.zone_id, k.time_bucket) for k in ev.keys])
        return ev

    def _rows_to_events(self, rows) -> list[Event]:
        out = []
        for r in rows:
            keys = [Key(*k) for k in self.conn.execute(
                "SELECT client_id, zone_id, time_bucket FROM event_keys WHERE event_id=?", (r[0],))]
            out.append(Event(event_id=r[0], seq=r[1], ts=r[2], run_id=r[3], event_type=r[4], client_id=r[5],
                             model_version=r[6], parent_version=r[7], artefact_ids=json.loads(r[8]),
                             touched_shared_encoder=bool(r[9]), payload=json.loads(r[10]), keys=keys))
        return out

    def events(self, event_type: Optional[str] = None, zone_id: Optional[int] = None,
               limit: int = 500, offset: int = 0) -> list[Event]:
        q, args = "SELECT DISTINCT e.* FROM events e LEFT JOIN event_keys k ON k.event_id=e.event_id WHERE 1=1", []
        if event_type:
            q += " AND e.event_type=?"; args.append(event_type)
        if zone_id is not None:
            q += " AND k.zone_id=?"; args.append(zone_id)
        q += " ORDER BY e.seq LIMIT ? OFFSET ?"; args += [limit, offset]
        return self._rows_to_events(self.conn.execute(q, args).fetchall())

    def influenced_by(self, zone_id: int, time_buckets: Optional[Iterable[int]] = None,
                      client_ids: Optional[Iterable[str]] = None) -> list[Event]:
        """All learning events whose data included (zone, time) - the starting point of an erasure."""
        q = ("SELECT DISTINCT e.* FROM events e JOIN event_keys k ON k.event_id=e.event_id "
             "WHERE k.zone_id=? AND e.event_type IN ('train','aggregate','memory_write','checkpoint')")
        args: list = [zone_id]
        if time_buckets is not None:
            tb = list(time_buckets)
            q += f" AND k.time_bucket IN ({','.join('?' * len(tb))})"; args += tb
        if client_ids is not None:
            cl = list(client_ids)
            q += f" AND k.client_id IN ({','.join('?' * len(cl))})"; args += cl
        return self._rows_to_events(self.conn.execute(q + " ORDER BY e.seq", args).fetchall())

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
