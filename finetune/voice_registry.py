"""Persistent registry for user-trained custom voices."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

USER_VOICE_PREFIX = "user_voice:"
REGISTRY_DIR_NAME = "user_voices"
REGISTRY_FILE_NAME = "registry.json"


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def registry_dir() -> str:
    return os.path.join(_project_root(), "finetune", REGISTRY_DIR_NAME)


def registry_path() -> str:
    return os.path.join(registry_dir(), REGISTRY_FILE_NAME)


def slugify_voice_name(name: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", (name or "").strip(), flags=re.UNICODE)
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_").lower()
    return slug or f"voice_{uuid.uuid4().hex[:8]}"


def _empty_registry() -> dict[str, Any]:
    return {"voices": {}}


def load_registry() -> dict[str, Any]:
    path = registry_path()
    if not os.path.isfile(path):
        return _empty_registry()
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "voices" not in data or not isinstance(data["voices"], dict):
        return _empty_registry()
    return data


def save_registry(data: dict[str, Any]) -> str:
    os.makedirs(registry_dir(), exist_ok=True)
    path = registry_path()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return path


def list_registered_voices() -> list[dict[str, Any]]:
    registry = load_registry()
    voices = []
    for voice_id, entry in registry.get("voices", {}).items():
        item = dict(entry)
        item["voice_id"] = voice_id
        voices.append(item)
    voices.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return voices


def get_registered_voice(voice_id: str) -> Optional[dict[str, Any]]:
    entry = load_registry().get("voices", {}).get(voice_id)
    if not entry:
        return None
    result = dict(entry)
    result["voice_id"] = voice_id
    return result


def list_user_voice_dropdown_choices() -> list[tuple[str, str]]:
    choices = []
    for entry in list_registered_voices():
        display_name = entry.get("display_name") or entry["voice_id"]
        choices.append((f"🎤 {display_name}", f"{USER_VOICE_PREFIX}{entry['voice_id']}"))
    return choices


def is_user_voice_choice(voice_choice: Optional[str]) -> bool:
    return bool(voice_choice and str(voice_choice).startswith(USER_VOICE_PREFIX))


def parse_user_voice_choice(voice_choice: str) -> str:
    return str(voice_choice).removeprefix(USER_VOICE_PREFIX)


def register_voice(
    *,
    display_name: str,
    lora_path: str,
    voice_preset_id: str,
    description: str = "",
    base_model: str = "pnnbao-ump/VieNeu-TTS-0.3B",
    voice_id: Optional[str] = None,
) -> dict[str, Any]:
    display_name = (display_name or "").strip()
    if not display_name:
        raise ValueError("Tên giọng đọc không được để trống.")

    voice_id = voice_id or slugify_voice_name(display_name)
    lora_path = os.path.abspath(lora_path)
    if not os.path.isdir(lora_path):
        raise FileNotFoundError(f"Không tìm thấy thư mục LoRA: {lora_path}")

    registry = load_registry()
    voices = registry.setdefault("voices", {})
    if voice_id in voices:
        voice_id = f"{voice_id}_{uuid.uuid4().hex[:6]}"

    entry = {
        "display_name": display_name,
        "description": description or f"Giọng tự train: {display_name}",
        "lora_path": os.path.relpath(lora_path, _project_root()),
        "base_model": base_model,
        "voice_preset_id": voice_preset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    voices[voice_id] = entry
    save_registry(registry)
    entry["voice_id"] = voice_id
    return entry


def unregister_voice(voice_id: str) -> bool:
    """Remove a voice from the registry. Returns True if it existed."""
    registry = load_registry()
    voices = registry.setdefault("voices", {})
    if voice_id in voices:
        del voices[voice_id]
        save_registry(registry)
        return True
    return False
