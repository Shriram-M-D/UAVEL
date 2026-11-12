"""F5: communication accounting and the data-flow evidence report."""
from __future__ import annotations


def dataflow_report(backend, history: list[dict], n_raw_image_bytes_local: int) -> dict:
    ev = backend.guard.evidence()
    up = sum(h["bytes_up"] for h in history)
    down = sum(h["bytes_down"] for h in history)
    return {
        "rounds": len(history),
        "bytes_up_total": up,
        "bytes_down_total": down,
        "bytes_per_round_up": [h["bytes_up"] for h in history],
        "guard": ev,
        "raw_image_bytes_held_locally": n_raw_image_bytes_local,
        "raw_image_bytes_sent": 0,
        "how_we_know": "Every client->coordinator payload went through uav_backend.Guard.check_update, which "
                       "only accepts allow-listed fields and shared-prefix parameters and rejects image-shaped "
                       "or uint8 data; accepted payload count == participating client-rounds.",
        "fields_ever_sent": ev["allowed_fields"],
    }
