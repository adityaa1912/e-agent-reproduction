from __future__ import annotations

import sys
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parents[1]
REPO_SRC = BASELINE_DIR.parent / "src"

for _path in (BASELINE_DIR, REPO_SRC):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
