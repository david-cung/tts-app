import os
from transformers import TrainingArguments
from peft import LoraConfig, TaskType

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Fallback config used when dataset size is not known. Real training picks
# capacity dynamically via `build_lora_config(sample_count)`.
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)


def build_lora_config(sample_count: int) -> LoraConfig:
    """Scale LoRA capacity with dataset size.

    Tiny datasets must stay compact to avoid overfitting/NaN; sufficient datasets
    benefit a lot from higher rank because LoRA needs to learn fine speaker
    characteristics across all attention and MLP projections.
    """
    if sample_count >= 30:
        r, alpha, dropout = 32, 64, 0.05
    elif sample_count >= 15:
        r, alpha, dropout = 16, 32, 0.05
    elif sample_count >= 8:
        r, alpha, dropout = 12, 24, 0.05
    else:
        r, alpha, dropout = 8, 16, 0.05
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


training_config = {
    'model': "pnnbao-ump/VieNeu-TTS-0.3B",
    'run_name': "VieNeu-TTS-0.3B-LoRA",
    'output_dir': os.path.join("finetune", "output"),
    
    'per_device_train_batch_size': 2, 
    'gradient_accumulation_steps': 1, 
    'max_seq_len': 2048,
    'dataloader_num_workers': 4,
    
    'learning_rate': 1e-4,
    'max_steps': 800,
    'logging_steps': 25,
    'save_steps': 100,
    'eval_steps': 500,

    'warmup_ratio': 0.1,
    'bf16': True,

    'use_4bit': False, 
}

def get_training_args(config):
    kwargs = dict(
        output_dir=os.path.join(config['output_dir'], config['run_name']),
        do_train=True,
        do_eval=False,
        max_steps=config['max_steps'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        warmup_ratio=config['warmup_ratio'],
        bf16=config.get('bf16', True),
        logging_steps=config['logging_steps'],
        save_steps=config['save_steps'],
        eval_strategy="no",
        save_strategy="steps",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=config.get('dataloader_num_workers', 4),
        ddp_find_unused_parameters=False,
        max_grad_norm=config.get('max_grad_norm', 1.0),
    )
    if config.get("fp16"):
        kwargs["fp16"] = True
    return TrainingArguments(**kwargs)
