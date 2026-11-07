"""Precompute and cache reference model log probabilities.

Computes ref_chosen_logps and ref_rejected_logps for every example,
caches to disk keyed by model name + subset. Reusable across all experiments
with the same base model and dataset subset.
"""

import os
import re

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.dpo_trainer import DataCollatorForPreference
from trl.trainer.utils import selective_log_softmax


def _model_short_name(model_name: str) -> str:
    """Extract short name from HF model path."""
    short = model_name.split("/")[-1]
    return re.sub(r"[^\w\-.]", "_", short)


def _cache_path(model_name: str, subset: str, split: str, cache_dir: str, data_mode: str = "") -> str:
    short = _model_short_name(model_name)
    suffix = f"_{data_mode}" if data_mode else ""
    return os.path.join(cache_dir, f"ref_logps_{short}_{subset}_{split}{suffix}.npz")


def compute_ref_logps_for_dataset(
    model_name: str,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    split: str,
    subset: str = "alpaca2-7b",
    max_length: int = 512,
    batch_size: int = 4,
    cache_dir: str = "outputs/ref_logps",
    device: str = "cuda",
    force_recompute: bool = False,
    data_mode: str = "",
) -> Dataset:
    """Compute ref logps and add as columns to dataset. Caches to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = _cache_path(model_name, subset, split, cache_dir, data_mode)

    if os.path.exists(cache_file) and not force_recompute:
        print(f"Loading cached ref logps from {cache_file}")
        loaded = np.load(cache_file)
        ref_chosen_logps = loaded["ref_chosen_logps"].tolist()
        ref_rejected_logps = loaded["ref_rejected_logps"].tolist()

        if len(ref_chosen_logps) != len(dataset):
            print(f"Cache size mismatch ({len(ref_chosen_logps)} vs {len(dataset)}), recomputing...")
        else:
            dataset = dataset.add_column("ref_chosen_logps", ref_chosen_logps)
            dataset = dataset.add_column("ref_rejected_logps", ref_rejected_logps)
            print(f"  Loaded {len(ref_chosen_logps)} ref logps pairs")
            return dataset

    print(f"Computing ref logps for {split} split ({len(dataset)} examples)...")
    print(f"  Loading ref model: {model_name}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    ref_model.eval()

    if "prompt_ids" not in dataset.column_names:
        first_example = dataset[0]
        is_conversational = isinstance(first_example["prompt"], list)

        if is_conversational:
            print("  Tokenizing dataset with chat template...")

            def tokenize_fn(example):
                prompt_ids = tokenizer.apply_chat_template(
                    example["prompt"], add_generation_prompt=True, tokenize=True, return_dict=False,
                )
                pc_ids = tokenizer.apply_chat_template(
                    example["prompt"] + example["chosen"], tokenize=True, return_dict=False,
                )
                pr_ids = tokenizer.apply_chat_template(
                    example["prompt"] + example["rejected"], tokenize=True, return_dict=False,
                )
                chosen_ids = pc_ids[len(prompt_ids):]
                rejected_ids = pr_ids[len(prompt_ids):]
                if len(prompt_ids) + len(chosen_ids) > max_length:
                    chosen_ids = chosen_ids[:max_length - len(prompt_ids)]
                if len(prompt_ids) + len(rejected_ids) > max_length:
                    rejected_ids = rejected_ids[:max_length - len(prompt_ids)]
                return {"prompt_ids": prompt_ids, "chosen_ids": chosen_ids, "rejected_ids": rejected_ids}
        else:
            print("  Tokenizing dataset with plain text...")

            def tokenize_fn(example):
                eos = tokenizer.eos_token or ""
                chosen = example["chosen"]
                rejected = example["rejected"]
                if eos and not chosen.endswith(eos):
                    chosen = chosen + eos
                if eos and not rejected.endswith(eos):
                    rejected = rejected + eos

                prompt_ids = tokenizer(example["prompt"], add_special_tokens=True)["input_ids"]
                pc_ids = tokenizer(example["prompt"] + chosen, add_special_tokens=True)["input_ids"]
                pr_ids = tokenizer(example["prompt"] + rejected, add_special_tokens=True)["input_ids"]
                chosen_ids = pc_ids[len(prompt_ids):]
                rejected_ids = pr_ids[len(prompt_ids):]
                if len(prompt_ids) + len(chosen_ids) > max_length:
                    chosen_ids = chosen_ids[:max_length - len(prompt_ids)]
                if len(prompt_ids) + len(rejected_ids) > max_length:
                    rejected_ids = rejected_ids[:max_length - len(prompt_ids)]
                return {"prompt_ids": prompt_ids, "chosen_ids": chosen_ids, "rejected_ids": rejected_ids}

        dataset_tokenized = dataset.map(tokenize_fn, num_proc=4)
    else:
        dataset_tokenized = dataset

    collator = DataCollatorForPreference(pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset_tokenized,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=0,
    )

    all_chosen_logps = []
    all_rejected_logps = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Ref logps ({split})"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            completion_mask = batch["completion_mask"].to(device)

            outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_cmask = completion_mask[..., 1:].contiguous()

            per_token_logps = selective_log_softmax(shift_logits, shift_labels)
            per_token_logps[shift_cmask == 0] = 0.0
            logps = per_token_logps.sum(dim=1)

            chosen_logps, rejected_logps = logps.chunk(2, dim=0)
            all_chosen_logps.append(chosen_logps.float().cpu())
            all_rejected_logps.append(rejected_logps.float().cpu())

    all_chosen_logps = torch.cat(all_chosen_logps).numpy()
    all_rejected_logps = torch.cat(all_rejected_logps).numpy()

    print(f"  Saving ref logps to {cache_file}")
    np.savez_compressed(
        cache_file,
        ref_chosen_logps=all_chosen_logps,
        ref_rejected_logps=all_rejected_logps,
        model_name=np.array(model_name),
        subset=np.array(subset),
    )

    del ref_model
    torch.cuda.empty_cache()

    dataset = dataset.add_column("ref_chosen_logps", all_chosen_logps.tolist())
    dataset = dataset.add_column("ref_rejected_logps", all_rejected_logps.tolist())
    print(f"  Computed {len(all_chosen_logps)} ref logps pairs")

    return dataset
