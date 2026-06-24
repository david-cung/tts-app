"""Persist last successful model load settings for auto-restore on startup."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from vieneu_utils.core_utils import env_bool


def _session_dir(project_root: str) -> str:
    return os.path.join(project_root, ".vieneu")


def session_path(project_root: str) -> str:
    return os.path.join(_session_dir(project_root), "model_session.json")


def load_model_session(project_root: str) -> Optional[dict[str, Any]]:
    path = session_path(project_root)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def save_model_session(project_root: str, payload: dict[str, Any]) -> str:
    directory = _session_dir(project_root)
    os.makedirs(directory, exist_ok=True)
    path = session_path(project_root)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def should_auto_load_model(project_root: str) -> bool:
    if env_bool("VIENEU_SKIP_AUTO_LOAD", default=False):
        return False
    return load_model_session(project_root) is not None
