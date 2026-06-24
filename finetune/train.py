import os
import sys
import json
import torch
import random
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainerCallback,
    default_data_collator
)
from peft import get_peft_model

# Thêm thư mục gốc và src vào path để import các module nội bộ
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, project_root)

from vieneu_utils.phonemize_text import phonemize_with_dict
from finetune.configs.lora_config import (
    build_lora_config,
    get_training_args,
    lora_config,
    training_config,
)

def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _is_lora_checkpoint_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    markers = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
    return any(os.path.isfile(os.path.join(path, name)) for name in markers)


def _resolve_lora_checkpoint_dir(output_dir: str, run_name: str) -> str:
    """Return the directory containing LoRA adapter weights for a training run."""
    run_dir = os.path.join(output_dir, run_name)
    if _is_lora_checkpoint_dir(run_dir):
        return run_dir

    if not os.path.isdir(run_dir):
        raise FileNotFoundError(
            f"Không tìm thấy thư mục checkpoint LoRA: `{run_dir}`."
        )

    checkpoint_dirs = []
    for name in os.listdir(run_dir):
        if not name.startswith("checkpoint-"):
            continue
        path = os.path.join(run_dir, name)
        if _is_lora_checkpoint_dir(path):
            try:
                step = int(name.split("-", 1)[1])
            except (IndexError, ValueError):
                step = 0
            checkpoint_dirs.append((step, path))

    if not checkpoint_dirs:
        raise FileNotFoundError(
            f"Train LoRA chưa tạo adapter hợp lệ trong `{run_dir}`."
        )

    checkpoint_dirs.sort(key=lambda item: item[0], reverse=True)
    return checkpoint_dirs[0][1]

def _use_mps_training() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def _prepare_mps_runtime() -> None:
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    if _use_mps_training():
        torch.mps.empty_cache()


def apply_device_training_config(config: dict) -> dict:
    """Tune batch/sequence settings for memory-constrained Apple MPS training."""
    tuned = dict(config)
    if _use_mps_training():
        print("🍎 Apple MPS — dùng cấu hình train tiết kiệm bộ nhớ.")
        tuned["per_device_train_batch_size"] = 1
        tuned["gradient_accumulation_steps"] = max(
            int(tuned.get("gradient_accumulation_steps", 1)),
            4,
        )
        tuned["bf16"] = False
        tuned["fp16"] = False
        tuned["dataloader_num_workers"] = 0
        tuned["max_seq_len"] = min(int(tuned.get("max_seq_len", 2048)), 1024)
        tuned["gradient_checkpointing"] = True
    else:
        tuned.setdefault("max_seq_len", 2048)
        tuned.setdefault("dataloader_num_workers", 4)
        tuned.setdefault("gradient_checkpointing", False)
    return tuned


def _load_backbone_for_training(model_name: str):
    if _use_mps_training():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        return model.to("mps")
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )


class _TrainingProgressCallback(TrainerCallback):
    """Push LoRA step progress back to the Gradio status panel."""

    def __init__(self, progress=None, max_steps: int | None = None, report_every: int = 25):
        self.progress = progress
        self.max_steps = max_steps
        self.report_every = max(1, int(report_every))
        self._last_reported = -1

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.progress:
            return
        step = int(state.global_step or 0)
        max_steps = int(self.max_steps or state.max_steps or args.max_steps or 0)
        if step <= 0:
            return
        if step != max_steps and step % self.report_every != 0:
            return
        if step == self._last_reported:
            return
        self._last_reported = step

        pct = int((step / max_steps) * 100) if max_steps else 0
        loss = logs.get("loss") if logs else None
        loss_text = f", loss={loss:.4f}" if isinstance(loss, (int, float)) else ""
        self.progress(
            f"🏋️ Bước 3/5 — Train LoRA: **{step}/{max_steps}** steps (~{pct}%){loss_text}"
        )

class _NanStopCallback(TrainerCallback):
    """Stop training early when loss/grad_norm becomes NaN (avoids corrupt checkpoints)."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        grad_norm = logs.get("grad_norm")
        loss = logs.get("loss")
        bad_grad = isinstance(grad_norm, (int, float)) and (
            grad_norm != grad_norm or grad_norm > 1e6
        )
        bad_loss = isinstance(loss, (int, float)) and (loss != loss or loss > 100)
        if bad_grad or bad_loss:
            print(
                "⚠️ Phát hiện grad_norm/loss bất thường — dừng train sớm để tránh checkpoint hỏng."
            )
            control.should_training_stop = True


def preprocess_sample(
    sample,
    tokenizer,
    max_len=2048,
    ref_phones=None,
    ref_codes=None,
    emotion_tag="<|emotion_0|>",
):
    """Build a single-utterance training sample: emotion + phonemes -> target speech codes.

    The target codes are always prioritized so that at least some tokens are supervised;
    otherwise a fully-masked sample produces a NaN loss that corrupts the whole adapter.
    `ref_phones`/`ref_codes` are accepted for API compatibility but intentionally unused —
    including the full reference codes per sample blows past `max_len` on MPS.
    """
    speech_gen_start = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
    text_prompt_start = tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
    text_prompt_end = tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")
    speech_end = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
    ignore_index = -100
    pad_id = tokenizer.pad_token_id

    target_phones = sample["phones"]
    target_codes = sample["codes"]

    emotion_ids = (
        tokenizer.encode(emotion_tag, add_special_tokens=False) if emotion_tag else []
    )
    phone_ids = tokenizer.encode(target_phones, add_special_tokens=False)

    target_codes_str = "".join([f"<|speech_{i}|>" for i in target_codes])
    target_code_ids = tokenizer.encode(target_codes_str, add_special_tokens=False)

    prefix_ids = (
        [text_prompt_start]
        + emotion_ids
        + phone_ids
        + [text_prompt_end, speech_gen_start]
    )

    # Reserve room so target codes (the supervised part) always fit, even if we must
    # trim the phoneme prefix. Keep at least 1 target token + the speech-end token.
    reserved_for_target = 8  # minimum target codes we never drop
    max_prefix = max_len - reserved_for_target - 1
    if len(prefix_ids) > max_prefix:
        # Trim phonemes from the prefix (keep the structural special tokens).
        overflow = len(prefix_ids) - max_prefix
        keep_phones = max(0, len(phone_ids) - overflow)
        phone_ids = phone_ids[:keep_phones]
        prefix_ids = (
            [text_prompt_start]
            + emotion_ids
            + phone_ids
            + [text_prompt_end, speech_gen_start]
        )

    target_budget = max_len - len(prefix_ids) - 1  # -1 for speech_end
    if target_budget < 1:
        target_budget = 1
    target_code_ids = target_code_ids[:target_budget]

    full_ids = prefix_ids + target_code_ids + [speech_end]
    full_ids = full_ids[:max_len]
    gen_start = len(prefix_ids)

    if len(full_ids) < max_len:
        full_ids = full_ids + [pad_id] * (max_len - len(full_ids))

    input_ids = torch.tensor(full_ids, dtype=torch.long)
    labels = torch.full_like(input_ids, ignore_index)

    supervised = 0
    for idx in range(gen_start, len(full_ids)):
        token_id = int(input_ids[idx].item())
        if token_id == pad_id:
            break
        labels[idx] = token_id
        supervised += 1

    if supervised == 0:
        # Last-resort guard: never emit a fully-masked sample (would yield NaN loss).
        labels[gen_start] = input_ids[gen_start]

    attention_mask = (input_ids != pad_id).long()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }

class VieNeuDataset(Dataset):
    def __init__(self, metadata_path, tokenizer, max_len=2048):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        if not os.path.exists(metadata_path):
             raise FileNotFoundError(f"Missing dataset file: {metadata_path}")
             
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    # filename|text|codes
                    self.samples.append({
                        "filename": parts[0],
                        "text": parts[1],
                        "codes": json.loads(parts[2])
                    })
        print(f"🦜 Đã tải {len(self.samples)} mẫu dữ liệu từ {metadata_path}")

        self.ref_phones = ""
        self.ref_codes: list[int] = []
        if self.samples:
            ref_sample = self.samples[0]
            try:
                self.ref_phones = phonemize_with_dict(ref_sample["text"])
            except Exception as exc:
                print(f"⚠️ Lỗi phonemize ref sample: {exc}")
                self.ref_phones = ref_sample["text"]
            self.ref_codes = ref_sample["codes"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        
        try:
            phones = phonemize_with_dict(text)
        except Exception as e:
            print(f"⚠️ Lỗi khi xử lý text: {e}")
            phones = text 
            
        data_item = {
            "phones": phones,
            "codes": sample["codes"]
        }
        
        return preprocess_sample(
            data_item,
            self.tokenizer,
            self.max_len,
            ref_phones=self.ref_phones,
            ref_codes=self.ref_codes,
        )

def apply_small_dataset_training_config(config: dict, sample_count: int) -> dict:
    """Tune steps/LR by dataset size to balance quality vs overfitting/NaN."""
    tuned = dict(config)
    steps = int(tuned.get("max_steps", 800))
    lr = float(tuned.get("learning_rate", 1e-4))

    # Heuristic: ~80 steps per sample is a good rule of thumb for LoRA on small
    # speech datasets; gives enough updates for the model to lock in voice
    # characteristics without overshooting.
    if sample_count >= 30:
        steps = min(steps, max(600, sample_count * 50))
        lr = min(lr, 1.5e-4)
    elif sample_count >= 15:
        steps = min(steps, max(500, sample_count * 60))
        lr = min(lr, 1e-4)
    elif sample_count >= 10:
        steps = min(steps, max(400, sample_count * 70))
        lr = min(lr, 8e-5)
    elif sample_count >= 6:
        steps = min(steps, max(250, sample_count * 60))
        lr = min(lr, 5e-5)
    else:
        steps = min(steps, max(150, sample_count * 50))
        lr = min(lr, 3e-5)

    tuned["max_steps"] = steps
    tuned["learning_rate"] = lr
    tuned.setdefault("max_grad_norm", 1.0)
    return tuned

def run_training(run_name=None, max_steps=None, dataset_dir=None, progress=None):
    project_root = _project_root()
    dataset_dir = os.path.abspath(dataset_dir or os.path.join(project_root, "finetune", "dataset"))

    config = dict(training_config)
    if run_name:
        config["run_name"] = run_name
    if max_steps is not None:
        config["max_steps"] = int(max_steps)
    config["output_dir"] = os.path.join(project_root, training_config["output_dir"])
    _prepare_mps_runtime()
    config = apply_device_training_config(config)
    max_seq_len = int(config.get("max_seq_len", 2048))

    model_name = config['model']
    print(f"🦜 Đang tải model gốc: {model_name}")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load Model
    model = _load_backbone_for_training(model_name)
    
    # Load Dataset
    dataset_path = os.path.join(dataset_dir, "metadata_encoded.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Không tìm thấy `{dataset_path}`. Hãy chạy bước mã hóa dataset trước khi train."
        )

    full_dataset = VieNeuDataset(dataset_path, tokenizer, max_len=max_seq_len)
    if len(full_dataset) == 0:
        raise RuntimeError(
            f"Dataset encoded trống tại `{dataset_path}`. Không có mẫu hợp lệ để train."
        )

    config = apply_small_dataset_training_config(config, len(full_dataset))
    print(
        f"🦜 Training config: {len(full_dataset)} samples, "
        f"max_steps={config['max_steps']}, lr={config['learning_rate']}"
    )
    
    print(f"🦜 Total samples: {len(full_dataset)} (eval disabled, training only)")
    
    # Apply LoRA — scale rank by dataset size for better voice quality on
    # larger datasets while keeping tiny datasets compact.
    lora_cfg = build_lora_config(len(full_dataset))
    print(
        f"🦜 Đang áp dụng LoRA adapters (r={lora_cfg.r}, alpha={lora_cfg.lora_alpha}) "
        f"cho {len(full_dataset)} mẫu..."
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    if config.get("gradient_checkpointing") and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    
    # Trainer Setup
    args = get_training_args(config)
    callbacks = [_NanStopCallback()]
    if progress:
        callbacks.append(
            _TrainingProgressCallback(
                progress=progress,
                max_steps=int(config["max_steps"]),
                report_every=max(10, int(config.get("logging_steps", 50) // 2)),
            )
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=full_dataset,
        eval_dataset=None,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )
    
    print("🦜 Bắt đầu quá trình huấn luyện! (Chúc may mắn)")
    if progress:
        progress(
            f"🏋️ Bước 3/5 — Train LoRA: **0/{config['max_steps']}** steps (0%)"
        )
    trainer.train()
    if progress:
        progress(
            f"🏋️ Bước 3/5 — Train LoRA hoàn tất (**{config['max_steps']}/{config['max_steps']}** steps)."
        )

    # Guard: a single NaN gradient (e.g. an all-masked sample) corrupts every LoRA
    # weight. Detect it before saving so we never register an unusable voice.
    nan_params = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and (torch.isnan(param).any() or torch.isinf(param).any())
    ]
    if nan_params:
        raise RuntimeError(
            "Trọng số LoRA bị NaN/Inf sau khi train (mô hình phân kỳ). "
            "Thường do learning rate quá cao hoặc dữ liệu quá ít/nhiễu trên Mac. "
            "Hãy thử lại với nhiều đoạn mẫu rõ hơn; app đã hạ learning rate và giới hạn steps."
        )

    # Save Final Model
    save_path = os.path.join(config['output_dir'], config['run_name'])
    os.makedirs(save_path, exist_ok=True)
    print(f"🦜 Đang lưu model LoRA tại: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    checkpoint_dir = _resolve_lora_checkpoint_dir(config["output_dir"], config["run_name"])

    try:
        from finetune.preview_trained_voice import generate_training_preview

        print("🎧 Đang tạo file nghe thử preview_sample.wav...")
        preview_path = generate_training_preview(checkpoint_dir, dataset_dir=dataset_dir)
        print(f"✅ Đã lưu mẫu nghe thử tại: {preview_path}")
    except Exception as exc:
        print(f"⚠️ Không tạo được preview mẫu sau train: {exc}")
        print("   Bạn vẫn có thể tạo preview trong tab Voice Training hoặc chạy:")
        print(f"   uv run python finetune/preview_trained_voice.py {checkpoint_dir}")

    return checkpoint_dir

if __name__ == "__main__":
    run_training()
