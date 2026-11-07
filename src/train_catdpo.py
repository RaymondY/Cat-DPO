"""Cat-DPO: Category-Adaptive DPO via Dual-Inspired Per-Category Margin Updates.

Per-category adaptive margins motivated by constrained optimization:
    max E[r(x,y)] - beta*KL  s.t.  E_{D_k}[h(x,y)] <= epsilon  for all k

The dual variables lambda_k serve as per-category margins in a surrogate DPO loss:
    L = -log sigma(beta*delta - lambda_k)

lambda_k is updated via a dual-inspired per-sample rule:
    For each sample i in category k:
        lambda_k <- max(0, lambda_k + eta * (1 - sigma(beta*delta_i) - epsilon))
"""

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from trl.trainer.dpo_trainer import DataCollatorForPreference
from trl.trainer.utils import selective_log_softmax

from data import load_pairs, format_for_dpo, HARM_CATEGORIES
from category_logger import CategoryLogger, CAT_SHORT_NAMES
from ref_logps import compute_ref_logps_for_dataset

NUM_CATEGORIES = len(HARM_CATEGORIES)


@dataclass
class DataCollatorForPreferenceWithCategories(DataCollatorForPreference):
    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        output = super().torch_call(examples)
        if "category_indices" in examples[0]:
            output["category_indices"] = [ex["category_indices"] for ex in examples]
        return output


class CatDPOTrainer(DPOTrainer):
    """Category-Adaptive DPO with per-sample dual-inspired margin updates.

    Each sample directly updates lambda_k for its harm categories.
    lambda_k itself provides all the memory/smoothing needed.
    """

    def __init__(
        self, *args,
        eta_dual: float = 0.1,
        epsilon: float = 0.05,
        category_logger: CategoryLogger = None,
        margin_agg: str = "max",
        lambda_cap: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # margin_agg: "max" (the paper's aggregation) or "sum" (the strict Lagrangian)
        self.margin_agg = margin_agg
        # lambda_cap: 0 disables; otherwise clip every lambda_k to this ceiling
        self.lambda_cap = lambda_cap

        self.data_collator = DataCollatorForPreferenceWithCategories(
            pad_token_id=self.processing_class.pad_token_id,
        )

        if (self.precompute_ref_logps and self.ref_model is not None
                and "ref_chosen_logps" in self.train_dataset.column_names):
            del self.ref_model
            self.ref_model = None
            torch.cuda.empty_cache()

        self.eta_dual = eta_dual
        self.epsilon = epsilon
        self.category_logger = category_logger

        # Per-category Lagrange multipliers (dual variables)
        self.lambdas = torch.zeros(NUM_CATEGORIES)

        # Track number of updates per category
        self.cat_update_count = torch.zeros(NUM_CATEGORIES, dtype=torch.long)

        self.catdpo_step = 0

        # CSV logger for lambda trajectories
        os.makedirs(self.args.output_dir, exist_ok=True)
        self._csv = open(os.path.join(self.args.output_dir, "catdpo_lambdas.csv"), "w", newline="")
        header = ["step"]
        header += [f"lambda/{name}" for name in CAT_SHORT_NAMES]
        header += [f"updates/{name}" for name in CAT_SHORT_NAMES]
        self._csv_writer = csv.writer(self._csv)
        self._csv_writer.writerow(header)

    def _precompute_ref_logps(self, dataset, name, batch_size):
        if "ref_chosen_logps" in dataset.column_names:
            return dataset
        return super()._precompute_ref_logps(dataset, name, batch_size)

    def _compute_loss(self, model, inputs, return_outputs):
        mode = "train" if self.model.training else "eval"
        device = self.accelerator.device

        # === Forward pass ===
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        completion_mask = inputs["completion_mask"]
        input_ids, attention_mask, completion_mask = self._truncate_inputs(
            input_ids, attention_mask, completion_mask
        )

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()
        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[shift_completion_mask == 0] = 0.0
        logps = per_token_logps.sum(dim=1)
        chosen_logps, rejected_logps = logps.chunk(2, dim=0)

        # === Ref logps ===
        if self.precompute_ref_logps:
            ref_chosen_logps = inputs["ref_chosen_logps"]
            ref_rejected_logps = inputs["ref_rejected_logps"]
        else:
            from trl.trainer.utils import disable_gradient_checkpointing
            with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
            ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
            ref_per_token_logps[shift_completion_mask == 0] = 0.0
            ref_logps = ref_per_token_logps.sum(dim=1)
            ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

        # === Delta score and margins ===
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        delta_score = chosen_logratios - rejected_logratios
        margins = self.beta * delta_score

        # Standard DPO loss (for tracking)
        per_sequence_loss = -F.logsigmoid(self.beta * delta_score)

        batch_size = per_sequence_loss.size(0)

        if mode == "train" and "category_indices" in inputs:
            cat_indices_batch = inputs["category_indices"]

            # --- Apply lambda_k margin per example ---
            lam = self.lambdas.to(device)
            shifted_margins = torch.zeros(batch_size, device=device)
            for i in range(batch_size):
                cats = cat_indices_batch[i]
                if isinstance(cats, torch.Tensor):
                    cats = cats.tolist()
                if cats:  # has categories = safe-unsafe pair
                    vals = [lam[c].item() for c in cats]
                    shifted_margins[i] = sum(vals) if self.margin_agg == "sum" else max(vals)

            # Margin-shifted DPO loss
            shifted_loss = -F.logsigmoid(self.beta * delta_score - shifted_margins)
            loss = shifted_loss.mean()

            # --- Per-sample dual-inspired update (detached) ---
            with torch.no_grad():
                pref_probs = torch.sigmoid(self.beta * delta_score).detach()

                for i in range(batch_size):
                    cats = cat_indices_batch[i]
                    if isinstance(cats, torch.Tensor):
                        cats = cats.tolist()
                    if not cats:
                        continue

                    # Per-sample violation: V_i = 1 - sigma(beta * delta_i)
                    V_i = 1.0 - pref_probs[i].item()

                    # Update lambda_k for each category this sample belongs to
                    for c in cats:
                        upd = max(0.0, self.lambdas[c] + self.eta_dual * (V_i - self.epsilon))
                        if self.lambda_cap > 0:
                            upd = min(upd, self.lambda_cap)
                        self.lambdas[c] = upd
                        self.cat_update_count[c] += 1

                self.catdpo_step += 1

                # Category logger
                det_margins = margins.detach()
                det_losses = per_sequence_loss.detach()
                prefprobs = torch.sigmoid(det_margins)
                if self.category_logger is not None:
                    self.category_logger.update(cat_indices_batch, det_margins, det_losses, prefprobs)

                # Logging every 10 steps
                if self.catdpo_step % 10 == 0:
                    lam_np = self.lambdas.numpy()
                    cnt_np = self.cat_update_count.numpy()

                    log_dict = {
                        "catdpo/lambda_max": float(lam_np.max()),
                        "catdpo/lambda_min": float(lam_np.min()),
                        "catdpo/lambda_mean": float(lam_np.mean()),
                    }
                    for c in range(NUM_CATEGORIES):
                        name = CAT_SHORT_NAMES[c]
                        log_dict[f"catdpo/lambda/{name}"] = float(lam_np[c])
                    if wandb.run is not None:
                        wandb.log(log_dict, step=self.state.global_step)

                    # CSV
                    row = [self.catdpo_step]
                    for c in range(NUM_CATEGORIES):
                        row.append(f"{lam_np[c]:.6f}")
                    for c in range(NUM_CATEGORIES):
                        row.append(str(int(cnt_np[c])))
                    self._csv_writer.writerow(row)
                    self._csv.flush()
        else:
            loss = per_sequence_loss.mean()

        # === Standard TRL metrics ===
        from trl.trainer.utils import entropy_from_logits
        per_token_entropy = entropy_from_logits(shift_logits.detach())
        entropy = per_token_entropy[shift_completion_mask.bool()].mean()
        entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
        self._metrics[mode]["entropy"].append(entropy)

        if mode == "train":
            num_tokens = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            self._total_train_tokens += num_tokens
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        self._metrics[mode]["rewards/chosen"].append(self.accelerator.gather(chosen_rewards).mean().item())
        self._metrics[mode]["rewards/rejected"].append(self.accelerator.gather(rejected_rewards).mean().item())
        self._metrics[mode]["rewards/accuracies"].append(
            self.accelerator.gather((chosen_rewards > rejected_rewards).float()).mean().item())
        self._metrics[mode]["rewards/margins"].append(self.accelerator.gather(margins.detach()).mean().item())
        self._metrics[mode]["logps/chosen"].append(self.accelerator.gather(chosen_logps.detach()).mean().item())
        self._metrics[mode]["logps/rejected"].append(self.accelerator.gather(rejected_logps.detach()).mean().item())

        return (loss, outputs) if return_outputs else loss


def build_dataset(split: str, subset: str = "alpaca2-7b", data_mode: str = "bettersafe",
                   conversational: bool = False, category_map: str = None) -> Dataset:
    examples = load_pairs(data_mode, split, subset=subset)
    if category_map:
        # Replace the dataset's annotated harm categories with externally supplied ones
        # (e.g. the unsupervised clusters from make_cluster_categories.py), keeping
        # safe-safe pairs category-free so they stay margin-free.
        with open(category_map) as f:
            cmap = json.load(f)
        hit = 0
        for ex in examples:
            if not ex["category_indices"]:
                continue
            h = hashlib.md5(ex["prompt"].encode()).hexdigest()[:12]
            if h in cmap:
                ex["category_indices"] = list(cmap[h])
                hit += 1
            else:
                ex["category_indices"] = []
        print(f"  category_map applied to {hit}/{len(examples)} examples ({split})")
    data_dict = format_for_dpo(examples, conversational=conversational)
    return Dataset.from_dict(data_dict)


def main():
    parser = argparse.ArgumentParser(description="Cat-DPO: Category-Adaptive Safety Alignment")
    parser.add_argument("--model_name", type=str, default="PKU-Alignment/alpaca2-7b-reproduced-llama-2")
    parser.add_argument("--output_dir", type=str, default="outputs/catdpo")
    parser.add_argument("--subset", type=str, default="alpaca2-7b")
    parser.add_argument("--data_mode", type=str, default="bettersafe",
                        choices=["helpful", "harmful", "bettersafe"])
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--eta_dual", type=float, default=0.1,
                        help="Dual ascent step size for per-sample lambda_k update")
    parser.add_argument("--margin_agg", type=str, default="max", choices=["max", "sum"],
                        help="Aggregate lambda over a sample's categories: 'max' (the paper's "
                             "choice, see App. A.5) or 'sum' (the strict Lagrangian aggregation)")
    parser.add_argument("--lambda_cap", type=float, default=0.0,
                        help="Clip every lambda_k to this ceiling (0 = uncapped). "
                             "--lambda_cap 10 reproduces the fixed-margin comparison")
    parser.add_argument("--category_map", type=str, default=None,
                        help="JSON {prompt_md5_12: [category_ids]} replacing the dataset's harm "
                             "labels, e.g. the clusters from make_cluster_categories.py")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="Per-category safety tolerance")
    parser.add_argument("--wandb_project", type=str, default="catdpo")
    parser.add_argument("--run_name", type=str, default="catdpo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_recompute_ref", action="store_true", default=False)
    parser.add_argument("--ref_cache_dir", type=str, default="outputs/ref_logps")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "BEGINNING OF CONVERSATION: USER: {{ message['content'] }} ASSISTANT:"
            "{% elif message['role'] == 'assistant' %}"
            " {{ message['content'] }}{{ eos_token }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{% endif %}"
        )
    conversational = True

    print(f"Loading datasets (subset={args.subset}, mode={args.data_mode})...")
    train_ds = build_dataset("train", subset=args.subset, data_mode=args.data_mode,
                             conversational=conversational, category_map=args.category_map)
    eval_ds = build_dataset("test", subset=args.subset, data_mode=args.data_mode,
                            conversational=conversational, category_map=args.category_map)
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    print("\nPreparing reference log probs...")
    train_ds = compute_ref_logps_for_dataset(
        model_name=args.model_name, dataset=train_ds, tokenizer=tokenizer,
        split="train", subset=args.subset, max_length=args.max_length,
        batch_size=args.batch_size, cache_dir=args.ref_cache_dir,
        force_recompute=args.force_recompute_ref, data_mode=args.data_mode,
    )
    eval_ds = compute_ref_logps_for_dataset(
        model_name=args.model_name, dataset=eval_ds, tokenizer=tokenizer,
        split="test", subset=args.subset, max_length=args.max_length,
        batch_size=args.batch_size, cache_dir=args.ref_cache_dir,
        force_recompute=args.force_recompute_ref, data_mode=args.data_mode,
    )

    print(f"\nLoading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16, attn_implementation="sdpa")

    training_args = DPOConfig(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, gradient_checkpointing=True,
        learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay,
        bf16=True, tf32=True, beta=args.beta, max_length=args.max_length,
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
        seed=args.seed, report_to="wandb", run_name=args.run_name,
        remove_unused_columns=False, precompute_ref_log_probs=True,
    )

    wandb.init(project=args.wandb_project, name=args.run_name,
               config={**vars(args), "method": "catdpo"})

    cat_logger = CategoryLogger(output_dir=args.output_dir, log_every=10)

    trainer = CatDPOTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tokenizer,
        eta_dual=args.eta_dual, epsilon=args.epsilon, category_logger=cat_logger,
        margin_agg=args.margin_agg, lambda_cap=args.lambda_cap,
    )

    trainer.train()

    final_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    cat_logger.close()
    trainer._csv.close()

    final_lambdas = {HARM_CATEGORIES[i]: trainer.lambdas[i].item() for i in range(NUM_CATEGORIES)}
    with open(os.path.join(args.output_dir, "final_lambdas.json"), "w") as f:
        json.dump(final_lambdas, f, indent=2)

    wandb.finish()
    print(f"\nDone. Model saved to {final_path}")


if __name__ == "__main__":
    main()
