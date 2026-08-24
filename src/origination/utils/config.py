from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment / pipeline config."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    logger.debug("Loaded config from {}", path)
    return cfg


def project_root() -> Path:
    """Repository root (parent of src/)."""
    # .../src/origination/utils/config.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3]


def resolve_data_dir(cfg: dict[str, Any]) -> Path:
    root = project_root()
    rel = cfg.get("project", {}).get("data_dir", "data")
    return (root / rel).resolve()


def resolve_experiments_dir(cfg: dict[str, Any]) -> Path:
    root = project_root()
    rel = cfg.get("project", {}).get("experiments_dir", "experiments")
    return (root / rel).resolve()
