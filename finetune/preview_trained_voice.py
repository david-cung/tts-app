"""Generate preview audio for fine-tuned LoRA checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional

DEFAULT_PREVIEW_TEXT = (
    "Xin chào, đây là giọng nói sau khi fine-tune với VieNeu-TTS. "
    "Bạn có thể nghe thử để kiểm tra độ giống với dữ liệu huấn luyện."
)
DEFAULT_BASE_MODEL = "pnnbao-ump/VieNeu-TTS-0.3B"
DEFAULT_CODEC_REPO = "neuphonic/distill-neucodec"
PREVIEW_SAMPLE_NAME = "preview_sample.wav"
PREVIEW_META_NAME = "preview_meta.json"


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_checkpoint_path(checkpoint: str) -> str:
    checkpoint = (checkpoint or "").strip()
    if not checkpoint:
        raise ValueError("Vui lòng chọn một checkpoint LoRA.")

    if os.path.isabs(checkpoint):
        resolved = os.path.abspath(checkpoint)
    else:
        resolved = os.path.abspath(os.path.join(_project_root(), checkpoint))

    if not _is_lora_checkpoint_dir(resolved):
        raise FileNotFoundError(f"Không tìm thấy checkpoint LoRA hợp lệ: {checkpoint}")
    return resolved


def _is_lora_checkpoint_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    adapter_markers = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
    )
    return any(os.path.isfile(os.path.join(path, name)) for name in adapter_markers)


def find_dataset_reference(dataset_dir: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    dataset_dir = dataset_dir or os.path.join(_project_root(), "finetune", "dataset")
    metadata_candidates = (
        os.path.join(dataset_dir, "metadata_cleaned.csv"),
        os.path.join(dataset_dir, "metadata.csv"),
    )

    for metadata_path in metadata_candidates:
        if not os.path.isfile(metadata_path):
            continue
        with open(metadata_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or "|" not in line:
                    continue
                filename, text = line.split("|", 1)
                filename = os.path.basename(filename.strip())
                text = text.strip()
                if not filename or not text:
                    continue
                audio_path = os.path.join(dataset_dir, "raw_audio", filename)
                if os.path.isfile(audio_path):
                    return audio_path, text
    return None, None


def list_training_checkpoints(output_dir: Optional[str] = None) -> list[str]:
    output_dir = output_dir or os.path.join(_project_root(), "finetune", "output")
    if not os.path.isdir(output_dir):
        return []

    checkpoints: list[str] = []
    for root, dirs, _files in os.walk(output_dir):
        dirs.sort(reverse=True)
        if _is_lora_checkpoint_dir(root):
            rel_path = os.path.relpath(root, _project_root())
            checkpoints.append(rel_path)

    def sort_key(path: str) -> tuple[int, str]:
        name = os.path.basename(path)
        if name.startswith("checkpoint-"):
            try:
                return (1, f"{int(name.split('-', 1)[1]):09d}")
            except ValueError:
                pass
        return (0, path)

    checkpoints.sort(key=sort_key, reverse=True)
    return checkpoints


def get_preview_sample_path(checkpoint: str) -> Optional[str]:
    checkpoint_dir = _resolve_checkpoint_path(checkpoint)
    sample_path = os.path.join(checkpoint_dir, PREVIEW_SAMPLE_NAME)
    return sample_path if os.path.isfile(sample_path) else None


def _write_preview_meta(
    checkpoint_dir: str,
    *,
    text: str,
    output_wav: str,
    ref_audio: Optional[str],
    ref_text: Optional[str],
) -> None:
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "output_wav": os.path.basename(output_wav),
        "ref_audio": os.path.basename(ref_audio) if ref_audio else None,
        "ref_text": ref_text,
        "base_model": DEFAULT_BASE_MODEL,
    }
    meta_path = os.path.join(checkpoint_dir, PREVIEW_META_NAME)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)


def generate_training_preview(
    checkpoint: str,
    text: Optional[str] = None,
    *,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    output_wav: Optional[str] = None,
    base_model: str = DEFAULT_BASE_MODEL,
    codec_repo: str = DEFAULT_CODEC_REPO,
    device: Optional[str] = None,
) -> str:
    """Synthesize preview audio for a LoRA checkpoint and return the WAV path."""
    checkpoint_dir = _resolve_checkpoint_path(checkpoint)
    preview_text = (text or DEFAULT_PREVIEW_TEXT).strip()
    if not preview_text:
        raise ValueError("Văn bản preview không được để trống.")

    ref_audio_path = ref_audio
    ref_text_value = ref_text
    if not ref_audio_path or not ref_text_value:
        dataset_ref_audio, dataset_ref_text = find_dataset_reference()
        ref_audio_path = ref_audio_path or dataset_ref_audio
        ref_text_value = ref_text_value or dataset_ref_text

    if not ref_audio_path or not os.path.isfile(ref_audio_path):
        raise FileNotFoundError(
            "Không tìm thấy audio mẫu để tham chiếu. Hãy lưu dataset có file trong "
            "`finetune/dataset/raw_audio/` hoặc truyền `--ref-audio` khi tạo preview."
        )
    if not ref_text_value:
        raise ValueError("Thiếu transcript cho audio mẫu tham chiếu.")

    if output_wav:
        output_path = os.path.abspath(output_wav)
    elif text is None or text.strip() == DEFAULT_PREVIEW_TEXT:
        output_path = os.path.join(checkpoint_dir, PREVIEW_SAMPLE_NAME)
    else:
        output_path = os.path.join(
            checkpoint_dir,
            f"preview_custom_{uuid.uuid4().hex[:8]}.wav",
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    project_root = _project_root()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    src_root = os.path.join(project_root, "src")
    if src_root not in sys.path:
        sys.path.insert(0, src_root)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Cần PyTorch để nghe thử LoRA. Hãy chạy `uv sync --group gpu` rồi thử lại."
        ) from exc

    from vieneu.standard import VieNeuTTS

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    tts = VieNeuTTS(
        backbone_repo=base_model,
        backbone_device=device,
        codec_repo=codec_repo,
        codec_device=device,
        hf_token=None,
    )
    try:
        tts.load_lora_adapter(checkpoint_dir)
        audio = tts.infer(
            preview_text,
            ref_audio=ref_audio_path,
            ref_text=ref_text_value,
        )
        tts.save(audio, output_path)
    finally:
        close_fn = getattr(tts, "close", None)
        if callable(close_fn):
            close_fn()

    _write_preview_meta(
        checkpoint_dir,
        text=preview_text,
        output_wav=output_path,
        ref_audio=ref_audio_path,
        ref_text=ref_text_value,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate preview audio for a trained LoRA checkpoint")
    parser.add_argument("checkpoint", help="Checkpoint path (absolute or relative to repo root)")
    parser.add_argument("--text", default=DEFAULT_PREVIEW_TEXT, help="Text to synthesize")
    parser.add_argument("--ref-audio", default=None, help="Reference audio path")
    parser.add_argument("--ref-text", default=None, help="Reference transcript")
    parser.add_argument("--output", default=None, help="Output WAV path")
    args = parser.parse_args()

    output_path = generate_training_preview(
        args.checkpoint,
        text=args.text,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        output_wav=args.output,
    )
    print(f"✅ Preview saved to: {output_path}")


if __name__ == "__main__":
    main()
