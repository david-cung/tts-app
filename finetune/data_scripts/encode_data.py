import os
import random

import json
import librosa
import torch
from neucodec import DistillNeuCodec
from tqdm import tqdm


def _codec_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode_dataset(dataset_dir="finetune/dataset", max_samples=2000):
    metadata_path = os.path.join(dataset_dir, "metadata_cleaned.csv")
    if not os.path.exists(metadata_path):
        print("🦜 Không tìm thấy metadata_cleaned.csv, thử dùng metadata.csv gốc...")
        metadata_path = os.path.join(dataset_dir, "metadata.csv")

    output_path = os.path.join(dataset_dir, "metadata_encoded.csv")
    raw_audio_dir = os.path.join(dataset_dir, "raw_audio")

    if not os.path.exists(metadata_path):
        print("🦜 Không tìm thấy file metadata nào!")
        return

    print("🦜 Đang tải DistillNeuCodec (khớp với inference)...")
    device = _codec_device()
    codec = DistillNeuCodec.from_pretrained("neuphonic/distill-neucodec").to(device)
    codec.eval()

    print(f"🦜 Bắt đầu encode metadata: {metadata_path}")
    print(f"🦜 Device codec: {device}")

    lines_to_write = []
    skipped_count = 0

    with open(metadata_path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    random.shuffle(lines)
    if len(lines) > max_samples:
        lines = lines[:max_samples]

    for line in tqdm(lines):
        parts = line.strip().split("|")
        if len(parts) < 2:
            continue

        filename = parts[0]
        text = parts[1]
        audio_path = os.path.join(raw_audio_dir, filename)

        if not os.path.exists(audio_path):
            skipped_count += 1
            continue

        try:
            wav, _ = librosa.load(audio_path, sr=16000, mono=True)
            wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                codes = codec.encode_code(audio_or_path=wav_tensor)
                codes = codes.squeeze(0).squeeze(0).cpu().numpy().flatten().tolist()
                codes = [int(x) for x in codes]

            if not codes:
                print(f"🦜 Empty codes cho file: {filename}")
                skipped_count += 1
                continue

            if not all(0 <= c < 65536 for c in codes):
                print(f"🦜 Invalid code range cho file: {filename}")
                skipped_count += 1
                continue

            lines_to_write.append(f"{filename}|{text}|{json.dumps(codes)}\n")
        except Exception as exc:
            print(f"🦜 Lỗi xử lý file {filename}: {exc}")
            skipped_count += 1

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.writelines(lines_to_write)

    print(f"\n🦜 Hoàn tất! Đã lưu file mã hóa tại: {output_path}")
    print(f"   - Tổng file xử lý thành công: {len(lines_to_write)}")
    print(f"   - Số file lỗi/bỏ qua: {skipped_count}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    target_dir = os.path.join(project_root, "finetune", "dataset")

    encode_dataset(dataset_dir=target_dir)
