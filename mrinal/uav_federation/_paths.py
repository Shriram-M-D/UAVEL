"""Make teammates' packages importable without copying their code (docs/INTERFACES.md section 0)."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "shriram", REPO / "shreyas"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
