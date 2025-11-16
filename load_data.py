from datasets import load_dataset
from torch.utils.data import DataLoader
import os, torch

def load_prompts(tokenizer, args):
    # Load once; keep raw text to avoid huge pre-tokenized tensors in RAM
    ds = load_dataset("tatsu-lab/alpaca", split="train", cache_dir=args["data_dir"])

    def collate(batch):
        texts = [ex["instruction"] for ex in batch]
        enc = tokenizer(
            texts,
            padding=True,                  # dynamic padding to batch max
            truncation=True,
            max_length=args["max_length"],
            return_tensors="pt"
        )
        return enc["input_ids"], enc["attention_mask"]

    num_workers = max(2, os.cpu_count() // 2)
    return DataLoader(
        ds,
        batch_size=args["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
        collate_fn=collate
    )