from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike | None = None, **overrides: Any) -> dict[str, Any]:
    p = Path(path or os.environ.get("GAPAGENT_CONFIG", ROOT / "config.yaml"))
    cfg = yaml.safe_load(p.read_text())
    for key in ("backend", "model", "mode", "db_path", "data_dir"):
        env = os.environ.get(f"GAPAGENT_{key.upper()}")
        if env:
            cfg[key] = env
    for k, v in overrides.items():
        if v is None:
            continue
        if k in cfg.get("retrieval", {}):
            cfg["retrieval"][k] = v
        else:
            cfg[k] = v
    for key in ("data_dir", "db_path"):
        if not Path(cfg[key]).is_absolute():
            cfg[key] = str(ROOT / cfg[key])
    return cfg


def config_hash(cfg: dict) -> str:
    keep = {k: cfg[k] for k in ("backend", "model", "temperature", "retrieval", "mode") if k in cfg}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:12]
