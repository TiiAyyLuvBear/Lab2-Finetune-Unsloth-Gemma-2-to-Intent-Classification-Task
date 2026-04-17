"""
train.py
Fine-tune Llama-3.1-8B (QLoRA / Unsloth) cho bài toán Banking Intent Classification.
Mỗi utterance được format theo template Alpaca Instruction-Tuning.
"""

import argparse
import logging
import os
import shutil
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    logging.warning("Unsloth không được cài đặt.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Prompt template (giống hệt notebook) ──────────────────────────────────────
ALPACA_PROMPT = """Below is an utterance from a customer.
Your task is to classify this utterance into one of the 77 banking intents.

### Instruction:
Identify the intent of the following message.

### Input:
{}

### Response:
{}"""


def load_id2label(mapping_path: str) -> dict:
    """Load id → label_text mapping từ file CSV."""
    df = pd.read_csv(mapping_path)
    return dict(zip(df["label"].astype(str), df["label_text"]))


def formatting_prompts_func(examples: dict, id2label: dict, eos_token: str) -> dict:
    """Format batch theo Alpaca prompt; output = intent TEXT (không phải số)."""
    inputs = examples["text"]
    outputs = [id2label[str(label)] for label in examples["label"]]
    texts = []
    for inp, out in zip(inputs, outputs):
        text = ALPACA_PROMPT.format(inp, out) + eos_token
        texts.append(text)
    return {"text": texts}


def setup_model(cfg: dict):
    """Load base model + attach LoRA adapters qua Unsloth."""
    model_name = cfg["model"]["name"]
    max_seq_length = cfg["model"].get("max_seq_length", 2048)
    load_in_4bit = cfg["model"].get("load_in_4bit", True)
    dtype = cfg["model"].get("dtype", None)  # None = auto

    logger.info("Loading model: %s  |  4bit=%s  |  max_seq=%s", model_name, load_in_4bit, max_seq_length)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    logger.info("Gắn LoRA adapters (r=%s, alpha=%s)", cfg["training"]["lora_r"], cfg["training"]["lora_alpha"])

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["training"].get("lora_r", 16),
        target_modules=cfg["training"].get("lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"]),
        lora_alpha=cfg["training"].get("lora_alpha", 16),
        lora_dropout=cfg["training"].get("lora_dropout", 0),
        bias=cfg["training"].get("bias", "none"),
        use_gradient_checkpointing=cfg["training"].get("gradient_checkpointing", "unsloth"),
        random_state=cfg["training"].get("seed", 3407),
    )
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Banking Intent Classifier với Unsloth.")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Đường dẫn file config YAML")
    parser.add_argument("--data_dir", type=str, default="sample_data", help="Thư mục chứa train.csv & id2label.csv")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Thư mục lưu checkpoint")
    parser.add_argument("--id2label_path", type=str, default=None,
                        help="Đường dẫn file id2label.csv (mặc định: {data_dir}/id2label.csv)")
    args = parser.parse_args()

    # ── Load config ────────────────────────────────────────────────────────────
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ── Load id2label ──────────────────────────────────────────────────────────
    id2label_path = args.id2label_path or str(Path(args.data_dir) / "id2label.csv")
    id2label = load_id2label(id2label_path)
    logger.info("Đã load %d intent labels từ %s", len(id2label), id2label_path)

    # ── Setup model ────────────────────────────────────────────────────────────
    if not UNSLOTH_AVAILABLE:
        raise RuntimeError("Unsloth is required for this script. Cài đặt: pip install unsloth")
    model, tokenizer = setup_model(cfg)

    # ── Load & format dataset ──────────────────────────────────────────────────
    train_csv = Path(args.data_dir) / "train.csv"
    logger.info("Tải train dataset từ %s", train_csv)
    dataset = load_dataset("csv", data_files={"train": str(train_csv)})["train"]

    # Format theo Alpaca prompt
    EOS_TOKEN = tokenizer.eos_token
    dataset = dataset.map(
        lambda ex: formatting_prompts_func(ex, id2label, EOS_TOKEN),
        batched=True,
        remove_columns=["text", "label"]  # chỉ giữ cột 'text' đã format
    )
    logger.info("Dataset sau format: %d examples", len(dataset))

    # ── SFT Trainer ─────────────────────────────────────────────────────────────
    t_cfg = cfg["training"]
    sft_cfg = SFTConfig(
        per_device_train_batch_size=t_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 4),
        warmup_steps=t_cfg.get("warmup_steps", 5),
        num_train_epochs=t_cfg.get("num_train_epochs", 3),
        max_steps=t_cfg.get("max_steps", -1),
        learning_rate=t_cfg.get("learning_rate", 2e-4),
        logging_steps=t_cfg.get("logging_steps", 20),
        optim=t_cfg.get("optim", "adamw_8bit"),
        weight_decay=t_cfg.get("weight_decay", 0.001),
        lr_scheduler_type=t_cfg.get("lr_scheduler_type", "linear"),
        seed=t_cfg.get("seed", 3407),
        output_dir=args.output_dir,
        report_to=t_cfg.get("report_to", "none"),
        packing=t_cfg.get("packing", False),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg["model"].get("max_seq_length", 2048),
        args=sft_cfg,
    )
    logger.info("Bắt đầu training ... (epochs=%d, batch=%d, grad_accum=%d)",
                 t_cfg.get("num_train_epochs"), t_cfg.get("per_device_train_batch_size"),
                 t_cfg.get("gradient_accumulation_steps"))
    trainer.train()

    # ── Save adapter ───────────────────────────────────────────────────────────
    model_dir = Path(args.output_dir) / "banking_intent_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    logger.info("Model (adapter) đã lưu tại: %s", model_dir)

    # ── Merge & save merged model (tùy chọn) ──────────────────────────────────
    if t_cfg.get("save_merged", False):
        merged_dir = Path(args.output_dir) / "merged"
        logger.info("Merge LoRA vào base model, lưu tại: %s", merged_dir)
        model.merge_and_unload().save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        logger.info("Merged model đã lưu.")


if __name__ == "__main__":
    main()