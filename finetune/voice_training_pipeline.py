"""End-to-end pipeline: dataset -> filter -> encode -> train -> voices.json -> registry."""

from __future__ import annotations

import os
import re
import sys
from typing import Callable, Optional

from finetune.voice_registry import register_voice, slugify_voice_name

ProgressCallback = Callable[[str], None]


def _prepare_memory_for_training(notify: Optional[ProgressCallback] = None) -> None:
    """Free unified memory on Mac by unloading the main Gradio TTS model."""
    def log(message: str) -> None:
        if notify:
            notify(message)

    import gc

    from apps.ui_utils import cleanup_gpu_memory

    log("🧹 Giải phóng bộ nhớ GPU (gỡ model TTS đang tải trong app)...")
    try:
        import apps.gradio_main as gradio_app

        if getattr(gradio_app, "tts", None) is not None:
            gradio_app.tts = None
            gradio_app.model_loaded = False
    except Exception:
        pass

    gc.collect()
    cleanup_gpu_memory()

    try:
        import torch

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
            torch.mps.empty_cache()
            log("🍎 Đã dọn cache MPS. Sau khi train xong, bấm **Tải Model** lại để dùng TTS.")
    except Exception:
        pass


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dataset_dir() -> str:
    return os.path.join(_project_root(), "finetune", "dataset")


def _count_metadata_lines(metadata_path: str) -> int:
    if not os.path.isfile(metadata_path):
        return 0
    count = 0
    with open(metadata_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if "|" in line.strip():
                count += 1
    return count


def validate_training_environment() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, (
            "Chưa cài PyTorch/PEFT. Hãy chạy `uv sync --group gpu` trên máy có GPU, "
            "rồi khởi động lại app."
        )

    has_accel = torch.cuda.is_available() or (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    )
    if not has_accel:
        return False, (
            "Không phát hiện GPU (CUDA/MPS). Train giọng tự động cần GPU; "
            "Docker web (CPU) không hỗ trợ bước này."
        )
    return True, ""


def validate_dataset_ready(
    min_samples: int = 5,
    dataset_dir: Optional[str] = None,
) -> tuple[int, str]:
    dataset_dir = dataset_dir or _dataset_dir()
    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    sample_count = _count_metadata_lines(metadata_path)
    if sample_count < min_samples:
        raise ValueError(
            f"Dataset chưa đủ dữ liệu ({sample_count} mẫu). "
            f"Hãy lưu ít nhất {min_samples} file WAV + transcript trước khi train LoRA."
        )
    return sample_count, metadata_path


def _pick_reference_sample(dataset_dir: str) -> tuple[str, str]:
    """Pick the longest valid clip as the cloning reference.

    A longer, clean reference gives the model far more voice context at inference
    time than the very first line in metadata — this materially improves clone
    similarity on small/medium datasets.
    """
    import soundfile as sf

    best: tuple[float, str, str] | None = None
    for metadata_name in ("metadata_cleaned.csv", "metadata.csv"):
        metadata_path = os.path.join(dataset_dir, metadata_name)
        if not os.path.isfile(metadata_path):
            continue
        with open(metadata_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if "|" not in line:
                    continue
                filename, text = line.split("|", 1)
                filename = os.path.basename(filename.strip())
                text = text.strip()
                if not filename or not text:
                    continue
                audio_path = os.path.join(dataset_dir, "raw_audio", filename)
                if not os.path.isfile(audio_path):
                    continue
                try:
                    duration = float(sf.info(audio_path).duration)
                except Exception:
                    continue
                # Sweet spot for reference: 4–13s — too short lacks voice context,
                # too long stretches the prompt and slows generation.
                if 4.0 <= duration <= 13.0 and (best is None or duration > best[0]):
                    best = (duration, audio_path, text)
        if best is not None:
            return best[1], best[2]

    # Fallback: take whatever the first usable line was, regardless of duration.
    for metadata_name in ("metadata_cleaned.csv", "metadata.csv"):
        metadata_path = os.path.join(dataset_dir, metadata_name)
        if not os.path.isfile(metadata_path):
            continue
        with open(metadata_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if "|" not in line:
                    continue
                filename, text = line.split("|", 1)
                filename = os.path.basename(filename.strip())
                text = text.strip()
                if not filename or not text:
                    continue
                audio_path = os.path.join(dataset_dir, "raw_audio", filename)
                if os.path.isfile(audio_path):
                    return audio_path, text
    raise FileNotFoundError(
        "Không tìm thấy file audio mẫu trong dataset để tạo preset giọng đọc."
    )


def _safe_run_name(display_name: str) -> str:
    slug = slugify_voice_name(display_name)
    return re.sub(r"[^A-Za-z0-9._-]", "_", slug) or "custom_voice"


def _recommended_max_steps(sample_count: int, requested: Optional[int] = None) -> int:
    if requested is not None:
        return int(requested)
    return min(500, max(150, sample_count * 60))


def run_voice_training_pipeline(
    display_name: str,
    *,
    progress: Optional[ProgressCallback] = None,
    max_steps: Optional[int] = None,
    dataset_dir: Optional[str] = None,
    min_samples: int = 5,
) -> dict:
    def notify(message: str) -> None:
        if progress:
            progress(message)

    ok, reason = validate_training_environment()
    if not ok:
        raise RuntimeError(reason)

    dataset_dir = dataset_dir or _dataset_dir()
    sample_count, _metadata_path = validate_dataset_ready(
        min_samples=min_samples,
        dataset_dir=dataset_dir,
    )

    from finetune.configs.lora_config import training_config

    run_name = _safe_run_name(display_name)
    output_dir = os.path.join(_project_root(), training_config["output_dir"])

    project_root = _project_root()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    src_root = os.path.join(project_root, "src")
    if src_root not in sys.path:
        sys.path.insert(0, src_root)

    notify(f"📦 Bắt đầu thêm giọng **{display_name}** ({sample_count} mẫu trong dataset)...")

    notify("🧹 Bước 1/5 — Lọc dataset...")
    from finetune.data_scripts.filter_data import filter_and_process_dataset

    filter_and_process_dataset(dataset_dir)
    cleaned_count = _count_metadata_lines(os.path.join(dataset_dir, "metadata_cleaned.csv"))
    if cleaned_count < 5:
        raise RuntimeError(
            f"Sau khi lọc chỉ còn {cleaned_count} mẫu (cần ít nhất 5). "
            "Hãy upload file dài hơn với nhiều câu nói rõ, có dấu câu (. , ? !), "
            "mỗi đoạn 3–15 giây."
        )

    notify(f"🔐 Bước 2/5 — Mã hóa audio ({cleaned_count} mẫu)...")
    import gc

    import torch
    from finetune.data_scripts.encode_data import encode_dataset

    encode_dataset(dataset_dir)
    gc.collect()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    encoded_count = _count_metadata_lines(os.path.join(dataset_dir, "metadata_encoded.csv"))
    if encoded_count == 0:
        raise RuntimeError("Mã hóa dataset thất bại — không có mẫu encoded.")

    _prepare_memory_for_training(notify)

    train_eta = (
        "Trên Mac có thể mất 30–90 phút; đang dùng batch=1 để tránh hết RAM."
        if os.uname().sysname == "Darwin"
        else "Có thể mất 30–90 phút tùy GPU."
    )
    notify(f"🏋️ Bước 3/5 — Train LoRA ({encoded_count} mẫu). {train_eta}")
    from finetune.train import run_training

    effective_max_steps = _recommended_max_steps(encoded_count, max_steps)
    if max_steps is None and effective_max_steps < training_config.get("max_steps", 5000):
        notify(f"ℹ️ Dataset nhỏ — giới hạn train **{effective_max_steps} steps** (tránh overfit/NaN trên Mac).")

    lora_dir = run_training(
        run_name=run_name,
        max_steps=effective_max_steps,
        dataset_dir=dataset_dir,
        progress=notify,
    )

    voice_preset_id = slugify_voice_name(display_name)
    ref_audio, ref_text = _pick_reference_sample(dataset_dir)

    notify("📝 Bước 4/5 — Tạo preset giọng đọc (voices.json)...")
    voices_json_path = os.path.join(lora_dir, "voices.json")
    from finetune.create_voices_json import create_voices_json

    create_voices_json(
        audio_path=ref_audio,
        text=ref_text,
        voice_name=voice_preset_id,
        output_path=voices_json_path,
        description=f"Giọng tự train: {display_name}",
        append=False,
        set_default=True,
    )

    notify("🎧 Bước 5/5 — Kiểm tra giọng (tạo file nghe thử)...")
    try:
        from finetune.preview_trained_voice import validate_checkpoint_generates_audio

        validate_checkpoint_generates_audio(lora_dir, dataset_dir=dataset_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Train xong nhưng giọng chưa tạo được audio hợp lệ: {exc} "
            "Hãy train lại với tên giọng mới và nhiều đoạn mẫu hơn (8–10 đoạn)."
        ) from exc

    entry = register_voice(
        display_name=display_name,
        lora_path=lora_dir,
        voice_preset_id=voice_preset_id,
        description=f"Giọng tự train: {display_name}",
        base_model=training_config["model"],
        voice_id=voice_preset_id,
    )

    notify(
        f"✅ Đã thêm giọng **{display_name}**. "
        "Chuyển sang tab **Preset** và chọn giọng có biểu tượng 🎤 để dùng."
    )
    return entry


def _voice_run_dataset_dir(display_name: str) -> str:
    slug = slugify_voice_name(display_name or "custom_voice")
    return os.path.join(_project_root(), "finetune", "dataset", "runs", slug)


def run_voice_training_from_media_files(
    display_name: str,
    media_files,
    *,
    progress: Optional[ProgressCallback] = None,
    max_steps: Optional[int] = None,
    min_duration: float = 3,
    max_duration: float = 15,
    silence_ms: int = 600,
    whisper_model: str = "small",
    min_clips: int = 5,
) -> dict:
    """Train a LoRA voice from one or more uploaded audio/video files.

    Each file is VAD-split into clips when long; short clean clips (3–15s) are
    used as-is. All clips are merged into a single dataset before training.
    """
    def notify(message: str) -> None:
        if progress:
            progress(message)

    if not (display_name or "").strip():
        raise ValueError("Vui lòng nhập tên giọng trước khi train LoRA.")

    if isinstance(media_files, str):
        media_files = [media_files]
    elif media_files is None:
        media_files = []
    media_files = [m for m in media_files if m]
    if not media_files:
        raise ValueError("Vui lòng upload ít nhất một file audio/video làm mẫu.")

    ok, reason = validate_training_environment()
    if not ok:
        raise RuntimeError(reason)

    from apps.ui_utils import (
        _collect_training_clips_from_files,
        _transcribe_clip_paths,
        persist_training_dataset_entries,
    )

    notify(f"✂️ Đang chuẩn bị {len(media_files)} file mẫu...")
    clips, warnings = _collect_training_clips_from_files(
        media_files,
        min_duration=min_duration,
        max_duration=max_duration,
        silence_ms=silence_ms,
        progress=notify,
    )
    for warning in warnings:
        notify(f"⚠️ {warning}")

    if len(clips) < min_clips:
        raise ValueError(
            f"Chỉ thu được {len(clips)} đoạn hợp lệ (cần ít nhất {min_clips}). "
            "Hãy upload thêm file mẫu hoặc dùng file có nhiều câu nói rõ ràng."
        )

    notify(f"📝 Đang tạo transcript Whisper cho {len(clips)} đoạn...")
    transcripts = _transcribe_clip_paths(clips, whisper_model_size=whisper_model)

    dataset_dir = _voice_run_dataset_dir(display_name)
    notify("💾 Đang lưu dataset riêng cho lần train này...")
    saved = persist_training_dataset_entries(
        clips,
        transcripts,
        dataset_dir,
        clear_existing=True,
    )
    notify(f"📦 Dataset sẵn sàng ({saved} mẫu). Bắt đầu pipeline LoRA...")

    return run_voice_training_pipeline(
        display_name,
        progress=progress,
        max_steps=max_steps,
        dataset_dir=dataset_dir,
        min_samples=min_clips,
    )


def run_voice_training_from_media_file(
    display_name: str,
    media_file: str,
    **kwargs,
) -> dict:
    """Backward-compatible single-file wrapper."""
    return run_voice_training_from_media_files(display_name, [media_file], **kwargs)
