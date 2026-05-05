"""LoRA fine-tuning script for SQLCoder-7B-2 on Spider train.

Slide 7 setup: rank-16 adapters on attention + FFN, 4-bit base model, ~2h on A100.
Slide 10 result: +13pp over the un-tuned base on Spider dev exec-acc.

Run as:
    python -m visql.lora_train --spider /path/to/spider/train_spider.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from . import config as cfg

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]   # attention + FFN
LR = 2e-4
EPOCHS = 2
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_LEN = 1024


class SpiderSQLDataset(Dataset):
    """Spider train as (prompt, completion) pairs for supervised fine-tuning."""

    def __init__(self, spider_path: str | Path, tokenizer, max_len: int = MAX_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(spider_path) as f:
            self.data = json.load(f)
        # Filter to entries with question + query
        self.data = [d for d in self.data if d.get("question") and d.get("query")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        prompt = (
            "### Task\nWrite a SQL query for the question.\n\n"
            f"### Question\n{ex['question']}\n\n### SQL\n"
        )
        target = ex["query"].rstrip(";") + ";"
        full = prompt + target
        enc = self.tokenizer(full, truncation=True, max_length=self.max_len,
                              padding="max_length", return_tensors="pt")
        ids = enc["input_ids"][0]
        # Mask prompt tokens from loss (only learn the SQL part)
        prompt_len = len(self.tokenizer(prompt, truncation=True, max_length=self.max_len)["input_ids"])
        labels = ids.clone()
        labels[:prompt_len] = -100
        labels[ids == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": ids,
            "attention_mask": enc["attention_mask"][0],
            "labels": labels,
        }


def train_lora(spider_path: str, output_dir: str | Path = None) -> str:
    """Run the LoRA fine-tune. Returns the output directory."""
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    output_dir = Path(output_dir or (cfg.LORA_DIR / "sqlcoder-spider-r16"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) load 4-bit base
    print(f"[lora] loading base model {cfg.SQL_BASE_MODEL} in 4-bit")
    tokenizer = AutoTokenizer.from_pretrained(cfg.SQL_BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.SQL_BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # 2) attach LoRA
    lora_cfg = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 3) dataset + trainer
    ds = SpiderSQLDataset(spider_path, tokenizer, max_len=MAX_LEN)
    print(f"[lora] dataset size: {len(ds)}")

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=lambda batch: {
            k: torch.stack([b[k] for b in batch]) for k in batch[0]
        },
    )

    print("[lora] training begins")
    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[lora] adapter saved to {output_dir}")
    return str(output_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--spider", required=True, help="Path to Spider train_spider.json")
    p.add_argument("--out", default=None, help="Output directory for LoRA adapter")
    a = p.parse_args()
    train_lora(a.spider, a.out)
