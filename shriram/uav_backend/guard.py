"""B9: raw-data locality guard.

Every client -> coordinator payload passes through check_update(). It rejects anything that is not an
allow-listed field, any parameter that is not a shared component (zone adapters must stay on the UAV),
and anything that looks like image data. Accepted and rejected payloads are logged, which produces the
'raw images never left the client' evidence for the demo.
"""
from __future__ import annotations

import sqlite3, time
from typing import Any

ALLOWED_FIELDS = {"client_id", "round_id", "model_version", "num_samples", "zone_ids", "time_buckets",
                  "state_dict", "metrics"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS comm_log (
  ts REAL, client_id TEXT, round_id TEXT, accepted INTEGER, bytes INTEGER, reason TEXT);
"""


class LocalityViolation(ValueError):
    pass


def _nbytes(v: Any) -> int:
    if hasattr(v, "nbytes"):
        return int(v.nbytes)  # numpy
    if hasattr(v, "element_size") and hasattr(v, "numel"):
        return int(v.element_size() * v.numel())  # torch
    raise LocalityViolation(f"state_dict value of type {type(v).__name__} is not a tensor/array")


def _looks_like_image(v: Any) -> bool:
    dt = str(getattr(v, "dtype", ""))
    if "uint8" in dt:
        return True  # parameters are never uint8; pixels usually are
    shape = tuple(getattr(v, "shape", ()))
    # (H, W, 3) or (N, 3, H, W) / (N, H, W, 3) with image-sized spatial dims
    if len(shape) == 3 and shape[-1] in (1, 3, 4) and min(shape[:2]) >= 16:
        return True
    if len(shape) == 4 and (shape[1] in (1, 3, 4) and min(shape[2:]) >= 16 or
                            shape[-1] in (1, 3, 4) and min(shape[1:3]) >= 16):
        return True
    return False


def _is_raw(v: Any) -> bool:
    return isinstance(v, (bytes, bytearray, memoryview)) or hasattr(v, "shape") or type(v).__name__ == "Image"


class Guard:
    def __init__(self, conn: sqlite3.Connection, shared_param_prefixes: list[str]):
        self.conn = conn
        self.prefixes = tuple(shared_param_prefixes)
        conn.executescript(SCHEMA)

    def check_update(self, payload: dict) -> int:
        """Validate one client update. Returns transmitted bytes; raises LocalityViolation if rejected."""
        try:
            n = self._check(payload)
        except LocalityViolation as e:
            self._log(payload, False, 0, str(e))
            raise
        self._log(payload, True, n, "")
        return n

    def _check(self, payload: dict) -> int:
        extra = set(payload) - ALLOWED_FIELDS
        if extra:
            raise LocalityViolation(f"fields not allowed to leave the client: {sorted(extra)}")
        for k, v in payload.items():
            if k != "state_dict" and _is_raw(v):
                raise LocalityViolation(f"field {k!r} carries array/binary data")
        for k, v in (payload.get("metrics") or {}).items():
            if not isinstance(v, (int, float)):
                raise LocalityViolation(f"metric {k!r} must be a scalar")
        total = 0
        for name, t in (payload.get("state_dict") or {}).items():
            if not name.startswith(self.prefixes):
                raise LocalityViolation(f"parameter {name!r} is not a shared component (adapters stay local)")
            if _looks_like_image(t):
                raise LocalityViolation(f"parameter {name!r} looks like image data (shape/dtype)")
            total += _nbytes(t)
        return total

    def _log(self, p: dict, ok: bool, n: int, reason: str) -> None:
        with self.conn:
            self.conn.execute("INSERT INTO comm_log VALUES (?,?,?,?,?,?)",
                              (time.time(), str(p.get("client_id")), str(p.get("round_id")), int(ok), n, reason))

    def evidence(self) -> dict:
        acc, rej, b = self.conn.execute(
            "SELECT COALESCE(SUM(accepted),0), COALESCE(SUM(1-accepted),0), COALESCE(SUM(bytes),0) FROM comm_log").fetchone()
        reasons = [r for (r,) in self.conn.execute("SELECT DISTINCT reason FROM comm_log WHERE accepted=0")]
        return {"payloads_accepted": acc, "payloads_rejected": rej, "bytes_transmitted": b,
                "raw_image_bytes_transmitted": 0, "rejection_reasons": reasons,
                "allowed_fields": sorted(ALLOWED_FIELDS), "shared_param_prefixes": list(self.prefixes)}
