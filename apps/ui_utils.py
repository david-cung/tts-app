import time
import gc
import math
import sys
import os
import shutil
import subprocess
import tempfile
import uuid
import gradio as gr
import numpy as np
import soundfile as sf
from functools import lru_cache

_REFERENCE_AUDIO_CACHE = {}
_WHISPER_MODEL_CACHE = {}

def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"

def _split_estimate_status(status: str) -> tuple[str, str]:
    if not isinstance(status, str):
        return status, ""

    estimate_marker = " | Ước tính còn lại: "
    if estimate_marker in status:
        status_text, estimate_text = status.split(" | ", 1)
        if status.endswith("...") and not status_text.endswith("..."):
            status_text += "..."
        return status_text, estimate_text.rstrip(". ")

    if ("batch mẫu:" in status or "trung bình batch:" in status) and "ước tính còn lại:" in status:
        start = status.find("(")
        end = status.rfind(")")
        if start != -1 and end != -1 and end > start:
            status_text = status[:start].strip()
            estimate_text = status[start + 1:end].replace(", ", "\n")
            return status_text, estimate_text

    return status, ""

def _extract_progress(status: str) -> tuple[str, int, int] | None:
    if not isinstance(status, str):
        return None

    for marker, label in (("Đang xử lý batch ", "batch"), ("Đang xử lý đoạn ", "đoạn")):
        if marker not in status:
            continue

        progress_text = status.split(marker, 1)[1].split(" ", 1)[0].strip(".")
        if "/" not in progress_text:
            return None

        current_text, total_text = progress_text.split("/", 1)
        try:
            current = int(current_text)
            total = int(total_text)
        except ValueError:
            return None

        if current > 0 and total > 0:
            return label, current, total

    return None

def wrap_with_estimate(synthesize_fn):
    def wrapper(*args):
        previous_progress_time = None
        total_unit_duration = 0.0
        completed_units = 0

        for audio_path, status in synthesize_fn(*args):
            status_text, estimate_text = _split_estimate_status(status)

            if not estimate_text:
                progress = _extract_progress(status_text)
                if progress:
                    unit_label, current, total = progress
                    now = time.time()
                    if previous_progress_time is not None:
                        total_unit_duration += now - previous_progress_time
                        completed_units += 1
                    previous_progress_time = now

                    if completed_units == 0:
                        estimate_text = f"Đang đo thời gian {unit_label} đầu tiên..."
                    else:
                        average_unit_duration = total_unit_duration / completed_units
                        estimated_total = average_unit_duration * total
                        estimated_remaining = average_unit_duration * max(0, total - current + 1)
                        estimate_text = (
                            f"Ước tính còn lại: {_format_duration(estimated_remaining)}\n"
                            f"Tổng: {_format_duration(estimated_total)}"
                        )

            yield audio_path, status_text, estimate_text
    return wrapper

def cleanup_gpu_memory():
    """Aggressively cleanup GPU memory (CUDA, MPS, XPU)"""
    if 'torch' in sys.modules:
        import torch
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
    gc.collect()

@lru_cache(maxsize=32)
def get_ref_text_cached(text_path: str) -> str:
    """Cache reference text loading"""
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read()

def on_codec_change(codec: str, current_mode: str):
    is_onnx = "onnx" in codec.lower()
    if is_onnx and current_mode == "custom_mode":
        return gr.update(visible=False), gr.update(selected="preset_mode"), "preset_mode"
    return gr.update(visible=not is_onnx), gr.update(), current_mode

def validate_audio_duration(audio_path):
    if not audio_path:
        return gr.update(visible=False)
    try:
        info = sf.info(audio_path)
        if info.duration > 5.1:
            return gr.update(
                value=f"⚠️ **Cảnh báo:** Audio mẫu hiện tại dài {info.duration:.1f} giây. Để có kết quả clone giọng tối ưu, bạn nên sử dụng đoạn audio có độ dài lý tưởng từ **3 đến 5 giây**.",
                visible=True
            )
    except Exception:
        pass
    return gr.update(visible=False)

def _coerce_audio_path(audio_path):
    if isinstance(audio_path, dict):
        return audio_path.get("path") or audio_path.get("name")
    if isinstance(audio_path, (list, tuple)) and audio_path:
        return audio_path[0]
    if hasattr(audio_path, "name"):
        return audio_path.name
    return audio_path

def prepare_reference_audio(audio_path):
    """Copy/convert uploaded reference audio to a stable WAV file.

    Gradio stores uploads in temporary folders that can disappear before the
    generation callback reads them. Keeping our own normalized copy avoids
    "File does not exist" errors and lets MP3/MP4 inputs work through ffmpeg.
    """
    source = _coerce_audio_path(audio_path)
    if not source:
        raise ValueError("Vui lòng upload file Audio mẫu (Reference Audio)!")

    source = os.path.abspath(os.fspath(source))
    cached_path = _REFERENCE_AUDIO_CACHE.get(source)
    if cached_path and os.path.isfile(cached_path):
        return cached_path

    if not os.path.isfile(source):
        raise FileNotFoundError(
            f"Không tìm thấy file audio mẫu: {source}. Hãy upload lại file và bấm Generate ngay sau khi upload xong."
        )

    cache_dir = os.path.join(tempfile.gettempdir(), "vieneu_tts_reference_audio")
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.abspath(source).startswith(os.path.abspath(cache_dir) + os.sep):
        return source

    ext = os.path.splitext(source)[1].lower()
    stable_path = os.path.join(cache_dir, f"reference_{uuid.uuid4().hex}.wav")

    if ext == ".wav":
        shutil.copy2(source, stable_path)
        _REFERENCE_AUDIO_CACHE[source] = stable_path
        return stable_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        source,
        "-vn",
        "-ac",
        "2",
        "-ar",
        "48000",
        stable_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Không tìm thấy ffmpeg để chuyển audio/video sang WAV.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip().splitlines()
        detail_text = detail[-1] if detail else str(exc)
        raise RuntimeError(f"Không chuyển được file audio/video sang WAV: {detail_text}") from exc

    _REFERENCE_AUDIO_CACHE[source] = stable_path
    return stable_path

def validate_and_cache_reference_audio(audio_path):
    if not audio_path:
        return gr.update(visible=False)

    try:
        stable_path = prepare_reference_audio(audio_path)
        info = sf.info(stable_path)
        if info.duration > 5.1:
            return gr.update(
                value=f"⚠️ **Cảnh báo:** Audio mẫu hiện tại dài {info.duration:.1f} giây. Để có kết quả clone giọng tối ưu, bạn nên sử dụng đoạn audio có độ dài lý tưởng từ **3 đến 5 giây**.",
                visible=True
            )
        return gr.update(visible=False)
    except Exception as exc:
        return gr.update(
            value=f"⚠️ **Không xử lý được audio mẫu:** {exc}",
            visible=True,
        )

def _coerce_file_list(files):
    if not files:
        return []
    if not isinstance(files, (list, tuple)):
        files = [files]

    paths = []
    for item in files:
        path = _coerce_audio_path(item)
        if path:
            paths.append(os.path.abspath(os.fspath(path)))
    return paths

def _build_vad_training_segments(speech_chunks, total_samples, min_samples, max_samples):
    """Combine VAD speech regions into bounded training clips."""
    if total_samples <= 0 or min_samples <= 0 or max_samples < min_samples:
        return []

    regions = []
    for chunk in speech_chunks or []:
        start = max(0, min(total_samples, int(chunk.get("start", 0))))
        end = max(start, min(total_samples, int(chunk.get("end", start))))
        duration = end - start
        part_count = max(1, math.ceil(duration / max_samples))
        part_size = math.ceil(duration / part_count)
        for part_index in range(part_count):
            part_start = start + part_index * part_size
            part_end = min(end, part_start + part_size)
            if part_end > part_start:
                regions.append([part_start, part_end])

    combined = []
    for start, end in regions:
        if combined and end - combined[-1][0] <= max_samples:
            combined[-1][1] = end
        else:
            combined.append([start, end])

    index = 0
    while index < len(combined):
        start, end = combined[index]
        if end - start >= min_samples:
            index += 1
            continue

        if index + 1 < len(combined) and combined[index + 1][1] - start <= max_samples:
            combined[index][1] = combined[index + 1][1]
            del combined[index + 1]
            continue

        if index > 0 and end - combined[index - 1][0] <= max_samples:
            combined[index - 1][1] = end
            del combined[index]
            index -= 1
            continue

        missing = min_samples - (end - start)
        grow_left = min(start, missing // 2)
        grow_right = min(total_samples - end, missing - grow_left)
        grow_left += min(start - grow_left, missing - grow_left - grow_right)
        combined[index] = [start - grow_left, end + grow_right]
        index += 1

    return [(start, end) for start, end in combined if end > start]

def _score_voice_reference_segment(audio, start, end, sample_rate, target_duration):
    """Score a VAD segment for reusable voice reference quality."""
    clip = np.asarray(audio[start:end], dtype=np.float32)
    if clip.size == 0:
        return float("-inf")

    duration = clip.size / float(sample_rate)
    abs_clip = np.abs(clip)
    rms = float(np.sqrt(np.mean(np.square(clip)) + 1e-12))
    peak = float(np.max(abs_clip)) if abs_clip.size else 0.0
    silence_ratio = float(np.mean(abs_clip < 0.005))
    clipped_ratio = float(np.mean(abs_clip > 0.98))
    dynamic_range = float(np.percentile(abs_clip, 95) - np.percentile(abs_clip, 20))

    duration_score = max(0.0, 1.0 - abs(duration - target_duration) / max(target_duration, 1.0))
    loudness_score = min(rms / 0.06, 1.0)
    peak_penalty = max(0.0, peak - 0.95) * 6.0

    return (
        duration_score * 4.0
        + loudness_score * 2.0
        + dynamic_range * 6.0
        - silence_ratio * 1.5
        - clipped_ratio * 12.0
        - peak_penalty
    )

def prepare_best_voice_training_reference(
    media_file,
    min_duration=3.0,
    target_duration=10.0,
    max_duration=15.0,
):
    """Extract the best short speech-only WAV from an uploaded audio/video file.

    The Voice Training button uses this before encoding a reusable voice preset.
    It intentionally avoids encoding a full long recording because silence,
    music, room noise, or speaker changes make the cloned voice less stable.
    """
    source = _coerce_audio_path(media_file)
    if not source:
        raise ValueError("Vui lòng upload một file giọng đọc.")

    source = os.path.abspath(os.fspath(source))
    if not os.path.isfile(source):
        raise FileNotFoundError(f"Không tìm thấy file nguồn: {source}")

    try:
        from faster_whisper.audio import decode_audio
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError as exc:
        raise RuntimeError(
            "Chưa cài faster-whisper nên không thể tự cắt đoạn giọng tốt nhất."
        ) from exc

    sample_rate = 16_000
    audio = np.asarray(decode_audio(source, sampling_rate=sample_rate), dtype=np.float32)
    if audio.size == 0:
        raise ValueError("File nguồn không có dữ liệu audio.")

    min_duration = float(min_duration)
    target_duration = float(target_duration)
    max_duration = float(max_duration)
    if min_duration <= 0 or target_duration < min_duration or max_duration < target_duration:
        raise ValueError("Cấu hình thời lượng cắt giọng không hợp lệ.")

    vad_options = VadOptions(
        min_speech_duration_ms=300,
        max_speech_duration_s=max_duration,
        min_silence_duration_ms=350,
        speech_pad_ms=180,
    )
    speech_chunks = get_speech_timestamps(audio, vad_options, sampling_rate=sample_rate)
    total_duration = audio.size / float(sample_rate)

    if not speech_chunks:
        if total_duration <= max_duration:
            segments = [(0, audio.size)]
            vad_note = "Không tách được VAD rõ ràng, dùng toàn bộ file vì audio đang ngắn."
        else:
            raise ValueError(
                "Không phát hiện được đoạn có giọng nói rõ trong file. "
                "Hãy thử upload file ít nhạc nền/ít nhiễu hơn."
            )
    else:
        segments = _build_vad_training_segments(
            speech_chunks,
            total_samples=len(audio),
            min_samples=int(min_duration * sample_rate),
            max_samples=int(max_duration * sample_rate),
        )
        vad_note = ""

    if not segments:
        raise ValueError("Không tìm được đoạn giọng đủ dài để training.")

    best_start, best_end = max(
        segments,
        key=lambda item: _score_voice_reference_segment(
            audio,
            item[0],
            item[1],
            sample_rate,
            target_duration,
        ),
    )

    output_dir = os.path.join(
        tempfile.gettempdir(),
        "vieneu_tts_best_voice_reference",
        uuid.uuid4().hex,
    )
    os.makedirs(output_dir, exist_ok=True)

    source_stem = _safe_training_filename(os.path.basename(source)).rsplit(".", 1)[0]
    output_path = os.path.join(output_dir, f"{source_stem}_best.wav")
    sf.write(output_path, audio[best_start:best_end], sample_rate, subtype="PCM_16")

    selected_duration = (best_end - best_start) / float(sample_rate)
    metadata = {
        "source": source,
        "source_name": os.path.basename(source),
        "original_duration": total_duration,
        "selected_duration": selected_duration,
        "start_sec": best_start / float(sample_rate),
        "end_sec": best_end / float(sample_rate),
        "candidate_count": len(segments),
        "was_trimmed": abs(selected_duration - total_duration) > 0.25,
        "note": vad_note,
    }
    return output_path, metadata

def split_voice_training_media(media_file, min_duration=3, max_duration=15, silence_ms=600):
    """Split one long audio/video file into training WAV clips using Silero VAD."""
    try:
        source = _coerce_audio_path(media_file)
        if not source:
            raise ValueError("Vui lòng upload một file audio hoặc video dài.")

        source = os.path.abspath(os.fspath(source))
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Không tìm thấy file nguồn: {source}")

        min_duration = float(min_duration)
        max_duration = float(max_duration)
        silence_ms = int(silence_ms)
        if min_duration < 1 or max_duration <= min_duration:
            raise ValueError("Thời lượng tối đa phải lớn hơn thời lượng tối thiểu.")

        try:
            from faster_whisper.audio import decode_audio
            from faster_whisper.vad import VadOptions, get_speech_timestamps
        except ImportError as exc:
            raise RuntimeError(
                "Chưa cài faster-whisper. Hãy cài dependency ASR rồi khởi động lại app."
            ) from exc

        sample_rate = 16_000
        audio = np.asarray(decode_audio(source, sampling_rate=sample_rate), dtype=np.float32)
        if audio.size == 0:
            raise ValueError("File nguồn không có dữ liệu audio.")

        vad_options = VadOptions(
            min_speech_duration_ms=250,
            max_speech_duration_s=max_duration,
            min_silence_duration_ms=max(100, silence_ms),
            speech_pad_ms=150,
        )
        speech_chunks = get_speech_timestamps(audio, vad_options, sampling_rate=sample_rate)
        segments = _build_vad_training_segments(
            speech_chunks,
            total_samples=len(audio),
            min_samples=int(min_duration * sample_rate),
            max_samples=int(max_duration * sample_rate),
        )
        if not segments:
            raise ValueError("Không phát hiện được đoạn có giọng nói trong file.")

        source_stem = _safe_training_filename(os.path.basename(source)).rsplit(".", 1)[0]
        output_dir = os.path.join(
            tempfile.gettempdir(),
            "vieneu_tts_training_split",
            uuid.uuid4().hex,
        )
        os.makedirs(output_dir, exist_ok=True)

        output_paths = []
        for index, (start, end) in enumerate(segments, start=1):
            output_path = os.path.join(output_dir, f"{source_stem}_{index:04d}.wav")
            sf.write(output_path, audio[start:end], sample_rate, subtype="PCM_16")
            output_paths.append(output_path)

        rows, warnings = _training_audio_rows(output_paths)
        speech_seconds = sum((end - start) / sample_rate for start, end in segments)
        status = (
            f"✅ Đã cắt {len(output_paths)} đoạn WAV từ `{os.path.basename(source)}` "
            f"({_format_duration(speech_seconds)} có tiếng nói)."
        )
        if warnings:
            status += "\n\n⚠️ " + "\n⚠️ ".join(warnings[:8])

        return output_paths, status, rows, gr.update(value=""), []
    except Exception as exc:
        return gr.update(), f"❌ Lỗi cắt audio/video: {exc}", [], gr.update(), []

def _read_training_script(script_file, script_text):
    if script_text and script_text.strip():
        return script_text.strip()

    parts = []
    script_path = _coerce_audio_path(script_file)
    if script_path:
        script_path = os.path.abspath(os.fspath(script_path))
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"Không tìm thấy file script: {script_path}")

        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "cp1258"):
            try:
                with open(script_path, "r", encoding=encoding) as f:
                    parts.append(f.read())
                last_error = None
                break
            except UnicodeDecodeError as exc:
                last_error = exc

        if last_error is not None:
            raise UnicodeDecodeError(
                last_error.encoding,
                last_error.object,
                last_error.start,
                last_error.end,
                "Không đọc được script. Hãy lưu file ở UTF-8.",
            )

    return "\n".join(parts).strip()

def _safe_training_filename(filename):
    stem, ext = os.path.splitext(os.path.basename(filename))
    safe_stem = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")
    safe_ext = ext.lower() or ".wav"
    return f"{safe_stem or 'audio'}{safe_ext}"

def _unique_training_filename(raw_audio_dir, filename, used_names):
    safe_name = _safe_training_filename(filename)
    stem, ext = os.path.splitext(safe_name)
    candidate = safe_name
    counter = 2

    while candidate in used_names or os.path.exists(os.path.join(raw_audio_dir, candidate)):
        candidate = f"{stem}_{counter}{ext}"
        counter += 1

    used_names.add(candidate)
    return candidate

def _validate_training_audio_paths(audio_files):
    audio_paths = _coerce_file_list(audio_files)
    if not audio_paths:
        raise ValueError("Vui lòng upload ít nhất một file WAV.")

    missing_files = [path for path in audio_paths if not os.path.isfile(path)]
    if missing_files:
        raise FileNotFoundError(f"Không tìm thấy file WAV: {missing_files[0]}")

    invalid_files = [
        os.path.basename(path)
        for path in audio_paths
        if os.path.splitext(path)[1].lower() != ".wav"
    ]
    if invalid_files:
        raise ValueError("Voice Training hiện chỉ nhận file .wav: " + ", ".join(invalid_files[:5]))

    return audio_paths

def _training_audio_duration(path):
    try:
        info = sf.info(path)
        warnings = []
        if info.duration < 3:
            warnings.append(f"{os.path.basename(path)}: ngắn hơn 3 giây")
        elif info.duration > 15:
            warnings.append(f"{os.path.basename(path)}: dài hơn 15 giây")
        return f"{info.duration:.1f}s", warnings
    except Exception as exc:
        return "?", [f"{os.path.basename(path)}: không đọc được duration ({exc})"]

def _training_audio_rows(audio_paths):
    rows = []
    warnings = []
    for index, path in enumerate(audio_paths, start=1):
        duration_text, duration_warnings = _training_audio_duration(path)
        rows.append([str(index), os.path.basename(path), duration_text])
        warnings.extend(duration_warnings)
    return rows, warnings

def inspect_voice_training_audio(audio_files):
    try:
        audio_paths = _validate_training_audio_paths(audio_files)
        rows, warnings = _training_audio_rows(audio_paths)
        status = f"✅ Đã nhận {len(rows)} file WAV. Transcript bên phải sẽ ghép theo cột #."
        if warnings:
            status += "\n\n⚠️ " + "\n⚠️ ".join(warnings[:8])
            if len(warnings) > 8:
                status += f"\n⚠️ Còn {len(warnings) - 8} cảnh báo khác."
        return status, rows
    except Exception as exc:
        return f"❌ Lỗi đọc WAV: {exc}", []

def _build_training_preview(audio_paths, script_content, require_complete):
    lines = [line.strip() for line in script_content.splitlines() if line.strip()]
    if not lines and require_complete:
        raise ValueError("Vui lòng nhập hoặc upload transcript/script cho audio training.")

    has_metadata_lines = any("|" in line for line in lines)
    transcripts = []
    if has_metadata_lines:
        text_by_name = {}
        for line in lines:
            if "|" not in line:
                continue
            file_name, text = line.split("|", 1)
            file_name = os.path.basename(file_name.strip())
            text = text.strip()
            if file_name and text:
                text_by_name[file_name] = text

        for audio_path in audio_paths:
            base_name = os.path.basename(audio_path)
            transcripts.append(text_by_name.get(base_name, ""))

        missing = [
            os.path.basename(audio_path)
            for audio_path, transcript in zip(audio_paths, transcripts)
            if not transcript
        ]
        if missing and require_complete:
            raise ValueError(
                "Script dạng metadata phải dùng đúng tên file audio. Thiếu transcript cho: "
                + ", ".join(missing[:5])
                + ("..." if len(missing) > 5 else "")
            )
    else:
        transcripts = lines[:len(audio_paths)]
        if len(transcripts) < len(audio_paths):
            transcripts.extend([""] * (len(audio_paths) - len(transcripts)))

    missing_indexes = [
        str(index)
        for index, transcript in enumerate(transcripts, start=1)
        if not transcript
    ]
    extra_count = max(0, len(lines) - len(audio_paths)) if not has_metadata_lines else 0

    if require_complete and missing_indexes:
        raise ValueError(
            "Thiếu transcript cho file số: "
            + ", ".join(missing_indexes[:8])
            + ("..." if len(missing_indexes) > 8 else "")
        )

    if require_complete and extra_count:
        raise ValueError(f"Script đang dư {extra_count} dòng so với số file WAV.")

    entries = [(audio_path, transcript) for audio_path, transcript in zip(audio_paths, transcripts)]
    preview_rows = []
    warnings = []
    for index, (audio_path, transcript) in enumerate(entries, start=1):
        duration_text, duration_warnings = _training_audio_duration(audio_path)
        preview_rows.append([str(index), os.path.basename(audio_path), duration_text, transcript])
        warnings.extend(duration_warnings)

    return entries, preview_rows, warnings, missing_indexes, extra_count

def preview_voice_training_dataset(audio_files, script_file=None, script_text=""):
    try:
        audio_paths = _validate_training_audio_paths(audio_files)
        script_content = _read_training_script(script_file, script_text)
        _, preview_rows, warnings, missing_indexes, extra_count = _build_training_preview(
            audio_paths,
            script_content,
            require_complete=False,
        )

        status = f"✅ Preview {len(preview_rows)} file WAV."
        if missing_indexes:
            status += "\n\n⚠️ Chưa có transcript cho file số: " + ", ".join(missing_indexes[:8])
            if len(missing_indexes) > 8:
                status += f"... và {len(missing_indexes) - 8} file khác"
        if extra_count:
            status += f"\n\n⚠️ Script đang dư {extra_count} dòng so với số file WAV."
        if warnings:
            status += "\n\n⚠️ " + "\n⚠️ ".join(warnings[:8])
            if len(warnings) > 8:
                status += f"\n⚠️ Còn {len(warnings) - 8} cảnh báo khác."

        return status, preview_rows
    except Exception as exc:
        return f"❌ Lỗi preview dataset: {exc}", []

def _get_whisper_model(model_size):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Chưa cài faster-whisper. Hãy chạy `uv sync --extra asr` hoặc `uv pip install faster-whisper` "
            "rồi khởi động lại app để dùng Auto Transcript."
        ) from exc

    model_size = (model_size or "small").strip()
    cache_key = (model_size, "cpu", "int8")
    if cache_key not in _WHISPER_MODEL_CACHE:
        _WHISPER_MODEL_CACHE[cache_key] = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )
    return _WHISPER_MODEL_CACHE[cache_key]

def transcribe_voice_training_audio(audio_files, whisper_model_size="small"):
    try:
        audio_paths = _validate_training_audio_paths(audio_files)
        model_size = (whisper_model_size or "small").strip()
        model = _get_whisper_model(model_size)

        transcripts = []
        warnings = []
        for index, audio_path in enumerate(audio_paths, start=1):
            segments, _ = model.transcribe(
                audio_path,
                language="vi",
                beam_size=5,
                condition_on_previous_text=False,
            )
            text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
            text = " ".join(text.split())
            transcripts.append(text)
            if not text:
                warnings.append(f"File số {index} ({os.path.basename(audio_path)}) chưa nhận được transcript.")

        script_text = "\n".join(transcripts)
        _, preview_rows, duration_warnings, missing_indexes, _ = _build_training_preview(
            audio_paths,
            script_text,
            require_complete=False,
        )
        warnings.extend(duration_warnings)

        status = (
            f"✅ Whisper đã tạo transcript cho {len(audio_paths)} file WAV.\n\n"
            "Bạn hãy kiểm tra/sửa lại transcript bên phải, sau đó bấm **Lưu dataset**."
        )
        if missing_indexes:
            status += "\n\n⚠️ Còn thiếu transcript cho file số: " + ", ".join(missing_indexes[:8])
        if warnings:
            status += "\n\n⚠️ " + "\n⚠️ ".join(warnings[:8])
            if len(warnings) > 8:
                status += f"\n⚠️ Còn {len(warnings) - 8} cảnh báo khác."

        return status, gr.update(value=script_text), preview_rows
    except Exception as exc:
        return f"❌ Lỗi Auto Transcript: {exc}", gr.update(), []

def save_voice_training_dataset(audio_files, script_file=None, script_text=""):
    """Save uploaded WAV files and transcripts into finetune/dataset."""
    try:
        audio_paths = _validate_training_audio_paths(audio_files)
        script_content = _read_training_script(script_file, script_text)
        entries, _, _, _, _ = _build_training_preview(
            audio_paths,
            script_content,
            require_complete=True,
        )

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        dataset_dir = os.path.join(repo_root, "finetune", "dataset")
        raw_audio_dir = os.path.join(dataset_dir, "raw_audio")
        metadata_path = os.path.join(dataset_dir, "metadata.csv")
        os.makedirs(raw_audio_dir, exist_ok=True)

        used_names = set()
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "|" in line:
                        used_names.add(line.split("|", 1)[0].strip())

        saved_rows = []
        warnings = []
        for index, (source_path, transcript) in enumerate(entries, start=1):
            dest_name = _unique_training_filename(raw_audio_dir, os.path.basename(source_path), used_names)
            dest_path = os.path.join(raw_audio_dir, dest_name)
            shutil.copy2(source_path, dest_path)

            duration_text, duration_warnings = _training_audio_duration(dest_path)
            warnings.extend(duration_warnings)

            saved_rows.append([str(index), dest_name, duration_text, transcript])

        needs_leading_newline = False
        if os.path.exists(metadata_path) and os.path.getsize(metadata_path) > 0:
            with open(metadata_path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                needs_leading_newline = f.read(1) != b"\n"

        with open(metadata_path, "a", encoding="utf-8") as f:
            if needs_leading_newline:
                f.write("\n")
            for _, file_name, _, transcript in saved_rows:
                f.write(f"{file_name}|{transcript}\n")

        with open(metadata_path, "r", encoding="utf-8") as f:
            total_samples = sum(1 for line in f if "|" in line)

        status = (
            f"## ✅ Lưu dataset thành công\n\n"
            f"Đã thêm **{len(saved_rows)} mẫu mới**. Dataset hiện có **{total_samples} mẫu**.\n\n"
            f"Thư mục dataset: `{dataset_dir}`\n\n"
            f"Metadata: `{metadata_path}`\n\n"
            "Bước tiếp theo: chạy filter/encode rồi train LoRA."
        )
        if warnings:
            status += "\n\n⚠️ " + "\n⚠️ ".join(warnings[:8])
            if len(warnings) > 8:
                status += f"\n⚠️ Còn {len(warnings) - 8} cảnh báo khác."

        gr.Info(
            f"Đã lưu {len(saved_rows)} mẫu mới. Dataset hiện có {total_samples} mẫu.",
            title="Lưu dataset thành công",
        )
        return status, saved_rows
    except Exception as exc:
        error_message = f"Lỗi lưu dataset: {exc}"
        gr.Warning(error_message, title="Không thể lưu dataset")
        return f"## ❌ Không thể lưu dataset\n\n{exc}", []

def on_custom_id_change(model_id):
    # Auto detect LoRA and base model
    if model_id and "lora" in model_id.lower():
        # Detect base model
        if "0.3" in model_id:
            base_model = "VieNeu-TTS-0.3B (GPU)"
        else:
            base_model = "VieNeu-TTS (GPU)"

        return (
            gr.update(visible=True, value=base_model),
            gr.update(), gr.update()
        )

    return (
        gr.update(visible=False),
        gr.update(),
        gr.update()
    )
