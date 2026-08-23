from __future__ import annotations

import sys
from pathlib import Path


def repo_src_path() -> Path:
    return Path(__file__).resolve().parents[2] / "src"


def ensure_eagent_importable() -> None:
    src = repo_src_path()
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


ensure_eagent_importable()
