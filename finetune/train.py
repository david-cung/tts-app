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
    default_data_collator
)
from peft import get_peft_model

# Thêm thư mục gốc và src vào path để import các module nội bộ
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, project_root)

from vieneu_utils.phonemize_text import phonemize_with_dict
from finetune.configs.lora_config import lora_config, training_config, get_training_args

def preprocess_sample(sample, tokenizer, max_len=2048):
    speech_gen_start = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
    ignore_index = -100
    
    phones = sample["phones"]
    vq_codes = sample["codes"]
    
    codes_str = "".join([f"<|speech_{i}|>" for i in vq_codes])
    chat = f"""<|TEXT_PROMPT_START|>{phones}<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>{codes_str}<|SPEECH_GENERATION_END|>"""
    
    ids = tokenizer.encode(chat)
    
    # Pad nếu ngắn
    if len(ids) < max_len:
        ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
    elif len(ids) > max_len:
        ids = ids[:max_len]
    
    input_ids = torch.tensor(ids, dtype=torch.long)
    labels = torch.full_like(input_ids, ignore_index)
    
    speech_gen_start_idx = (input_ids == speech_gen_start).nonzero(as_tuple=True)[0]
    if len(speech_gen_start_idx) > 0:
        speech_gen_start_idx = speech_gen_start_idx[0]
        labels[speech_gen_start_idx:] = input_ids[speech_gen_start_idx:]
    
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
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
        
        return preprocess_sample(data_item, self.tokenizer, self.max_len)

def run_training(run_name=None, max_steps=None):
    config = dict(training_config)
    if run_name:
        config["run_name"] = run_name
    if max_steps is not None:
        config["max_steps"] = int(max_steps)

    model_name = config['model']
    print(f"🦜 Đang tải model gốc: {model_name}")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load Dataset
    dataset_path = os.path.join("finetune", "dataset", "metadata_encoded.csv")
    if not os.path.exists(dataset_path):
        print(f"⚠️ Không tìm thấy {dataset_path}. Vui lòng chạy prepare data trước.")
        return

    full_dataset = VieNeuDataset(dataset_path, tokenizer)
    
    print(f"🦜 Total samples: {len(full_dataset)} (eval disabled, training only)")
    
    # Apply LoRA
    print("🦜 Đang áp dụng LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Trainer Setup
    args = get_training_args(config)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=full_dataset,
        eval_dataset=None,
        data_collator=default_data_collator,
    )
    
    print("🦜 Bắt đầu quá trình huấn luyện! (Chúc may mắn)")
    trainer.train()
    
    # Save Final Model
    save_path = os.path.join(config['output_dir'], config['run_name'])
    print(f"🦜 Đang lưu model LoRA tại: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    try:
        from finetune.preview_trained_voice import generate_training_preview

        print("🎧 Đang tạo file nghe thử preview_sample.wav...")
        preview_path = generate_training_preview(save_path)
        print(f"✅ Đã lưu mẫu nghe thử tại: {preview_path}")
    except Exception as exc:
        print(f"⚠️ Không tạo được preview mẫu sau train: {exc}")
        print("   Bạn vẫn có thể tạo preview trong tab Voice Training hoặc chạy:")
        print(f"   uv run python finetune/preview_trained_voice.py {save_path}")

if __name__ == "__main__":
    run_training()
