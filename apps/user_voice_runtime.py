"""Lazy-loaded inference runtimes for user-trained custom voices."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import soundfile as sf

from finetune.voice_registry import (
    get_registered_voice,
    is_user_voice_choice,
    list_user_voice_dropdown_choices,
    parse_user_voice_choice,
)

_USER_TTS_CACHE: dict[str, object] = {}
_USER_TTS_LOCK = threading.Lock()


def merge_voice_dropdown_choices(model_choices: Optional[list] = None) -> list:
    user_choices = list_user_voice_dropdown_choices()
    if not model_choices:
        return user_choices
    if not user_choices:
        return list(model_choices)

    existing_values = set()
    merged = []
    for item in user_choices:
        merged.append(item)
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            existing_values.add(item[1])
        else:
            existing_values.add(item)

    for item in model_choices:
        value = item[1] if isinstance(item, (list, tuple)) and len(item) >= 2 else item
        if value not in existing_values:
            merged.append(item)
    return merged


def get_user_voice_entry(voice_choice: Optional[str]) -> Optional[dict]:
    if not is_user_voice_choice(voice_choice):
        return None
    voice_id = parse_user_voice_choice(voice_choice)
    return get_registered_voice(voice_id)


def _resolve_lora_path(entry: dict) -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lora_path = entry.get("lora_path") or ""
    if os.path.isabs(lora_path):
        return os.path.abspath(lora_path)
    return os.path.abspath(os.path.join(project_root, lora_path))


def _load_user_voice_tts(entry: dict):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Cần PyTorch để dùng giọng tự train. Hãy chạy `uv sync --group gpu`."
        ) from exc

    from vieneu.standard import VieNeuTTS

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    lora_path = _resolve_lora_path(entry)
    base_model = entry.get("base_model") or "pnnbao-ump/VieNeu-TTS-0.3B"

    tts = VieNeuTTS(
        backbone_repo=base_model,
        backbone_device=device,
        codec_repo="neuphonic/distill-neucodec",
        codec_device=device,
    )
    tts.load_lora_adapter(lora_path)
    voices_json = os.path.join(lora_path, "voices.json")
    if os.path.isfile(voices_json):
        tts._load_voices_from_file(Path(voices_json))
    return tts


def get_user_voice_tts(voice_choice: str):
    entry = get_user_voice_entry(voice_choice)
    if not entry:
        raise ValueError("Giọng tự train không tồn tại trong registry.")

    voice_id = entry["voice_id"]
    with _USER_TTS_LOCK:
        cached = _USER_TTS_CACHE.get(voice_id)
        if cached is not None:
            return cached
        tts = _load_user_voice_tts(entry)
        _USER_TTS_CACHE[voice_id] = tts
        return tts


def get_user_voice_preset_id(voice_choice: str) -> str:
    entry = get_user_voice_entry(voice_choice)
    if not entry:
        raise ValueError("Giọng tự train không tồn tại.")
    return entry.get("voice_preset_id") or parse_user_voice_choice(voice_choice)


def clear_user_voice_cache() -> None:
    with _USER_TTS_LOCK:
        for tts in _USER_TTS_CACHE.values():
            close_fn = getattr(tts, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        _USER_TTS_CACHE.clear()


def synthesize_registered_voice(text: str, voice_choice: str, temperature: float = 1.0) -> tuple[str, str]:
    entry = get_user_voice_entry(voice_choice)
    if not entry:
        raise ValueError("Giọng tự train không tồn tại.")

    if not text or not text.strip():
        raise ValueError("Vui lòng nhập văn bản.")

    started = time.time()
    infer_tts = get_user_voice_tts(voice_choice)
    preset_id = get_user_voice_preset_id(voice_choice)
    voice_data = infer_tts.get_preset_voice(preset_id)
    ref_codes = voice_data["codes"]
    ref_text = voice_data["text"]

    if "torch" in sys.modules:
        import torch

        if isinstance(ref_codes, torch.Tensor):
            ref_codes = ref_codes.cpu().numpy()

    wav = infer_tts.infer(
        text.strip(),
        ref_codes=ref_codes,
        ref_text=ref_text,
        temperature=temperature,
    )
    sample_rate = getattr(infer_tts, "sample_rate", 24000)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, wav, sample_rate)
        output_path = tmp.name

    elapsed = time.time() - started
    return output_path, (
        f"✅ Hoàn tất với giọng **{entry['display_name']}** "
        f"(Thời gian: {elapsed:.1f}s)"
    )
