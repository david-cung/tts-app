"""End-to-end pipeline: dataset -> filter -> encode -> train -> voices.json -> registry."""

from __future__ import annotations

import os
import re
import sys
from typing import Callable, Optional

from finetune.voice_registry import register_voice, slugify_voice_name

ProgressCallback = Callable[[str], None]


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


def validate_dataset_ready(min_samples: int = 3) -> tuple[int, str]:
    metadata_path = os.path.join(_dataset_dir(), "metadata.csv")
    sample_count = _count_metadata_lines(metadata_path)
    if sample_count < min_samples:
        raise ValueError(
            f"Dataset chưa đủ dữ liệu ({sample_count} mẫu). "
            f"Hãy lưu ít nhất {min_samples} file WAV + transcript bằng **Lưu dataset** trước."
        )
    return sample_count, metadata_path


def _pick_reference_sample(dataset_dir: str) -> tuple[str, str]:
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


def run_voice_training_pipeline(
    display_name: str,
    *,
    progress: Optional[ProgressCallback] = None,
    max_steps: Optional[int] = None,
) -> dict:
    def notify(message: str) -> None:
        if progress:
            progress(message)

    ok, reason = validate_training_environment()
    if not ok:
        raise RuntimeError(reason)

    sample_count, _metadata_path = validate_dataset_ready()
    dataset_dir = _dataset_dir()

    from finetune.configs.lora_config import training_config

    run_name = _safe_run_name(display_name)
    output_dir = os.path.join(_project_root(), training_config["output_dir"])
    lora_dir = os.path.join(output_dir, run_name)

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
    if cleaned_count == 0:
        raise RuntimeError(
            "Không còn mẫu hợp lệ sau khi lọc. Kiểm tra transcript (có dấu câu . , ? !) "
            "và độ dài audio 3–15 giây."
        )

    notify(f"🔐 Bước 2/5 — Mã hóa audio ({cleaned_count} mẫu)...")
    from finetune.data_scripts.encode_data import encode_dataset

    encode_dataset(dataset_dir)
    encoded_count = _count_metadata_lines(os.path.join(dataset_dir, "metadata_encoded.csv"))
    if encoded_count == 0:
        raise RuntimeError("Mã hóa dataset thất bại — không có mẫu encoded.")

    notify(
        f"🏋️ Bước 3/5 — Train LoRA ({encoded_count} mẫu). "
        "Có thể mất 30–90 phút tùy GPU..."
    )
    from finetune.train import run_training

    run_training(run_name=run_name, max_steps=max_steps)

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

    notify("🎧 Bước 5/5 — Tạo file nghe thử...")
    try:
        from finetune.preview_trained_voice import generate_training_preview

        generate_training_preview(lora_dir)
    except Exception as exc:
        notify(f"⚠️ Train xong nhưng không tạo được preview: {exc}")

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
