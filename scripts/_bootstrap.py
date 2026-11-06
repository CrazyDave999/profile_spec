"""Repository paths and sibling imports for directly executable research scripts."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for category in ("analysis", "experiments", "reports", "setup"):
    path = str(ROOT / "scripts" / category)
    if path not in sys.path:
        sys.path.append(path)
