"""
Small CSV/JSON cache helpers for data providers.

Default cache directory:
    .cache/data_providers/

Override with:
    FIN_DATA_CACHE=/path/to/cache
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CACHE_DIR = Path(".cache") / "data_providers"


def cache_dir() -> Path:
    root = Path(os.environ.get("FIN_DATA_CACHE", DEFAULT_CACHE_DIR))
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_key(*parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def cache_path(namespace: str, key: str, suffix: str) -> Path:
    folder = cache_dir() / namespace
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{key}.{suffix}"


def is_fresh(path: Path, ttl_seconds: int | None) -> bool:
    if ttl_seconds is None:
        return path.exists()
    if ttl_seconds <= 0 or not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) <= ttl_seconds


def read_frame(namespace: str, key: str, ttl_seconds: int | None = None) -> pd.DataFrame | None:
    path = cache_path(namespace, key, "csv")
    if not is_fresh(path, ttl_seconds):
        return None
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None


def write_frame(namespace: str, key: str, frame: pd.DataFrame) -> None:
    path = cache_path(namespace, key, "csv")
    frame.to_csv(path)


def read_json(namespace: str, key: str, ttl_seconds: int | None = None) -> Any | None:
    path = cache_path(namespace, key, "json")
    if not is_fresh(path, ttl_seconds):
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(namespace: str, key: str, value: Any) -> None:
    path = cache_path(namespace, key, "json")
    path.write_text(json.dumps(value, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
