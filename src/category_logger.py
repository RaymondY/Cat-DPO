"""Per-category metrics logger for training monitoring.

Writes per-category margins, losses, and preference probabilities
to both wandb and local CSV.
"""

import csv
import os

import torch
import wandb

from data import HARM_CATEGORIES

NUM_CATEGORIES = len(HARM_CATEGORIES)

CAT_SHORT_NAMES = [c.replace(" ", "_")[:20] for c in HARM_CATEGORIES]


class CategoryLogger:
    """Accumulates per-category metrics per batch,
    then logs summary every N steps to wandb + local CSV."""

    def __init__(self, output_dir: str, log_every: int = 10):
        self.output_dir = output_dir
        self.log_every = log_every
        self.step = 0

        self._margin_sum = torch.zeros(NUM_CATEGORIES)
        self._loss_sum = torch.zeros(NUM_CATEGORIES)
        self._prefprob_sum = torch.zeros(NUM_CATEGORIES)
        self._count = torch.zeros(NUM_CATEGORIES)

        os.makedirs(output_dir, exist_ok=True)
        self.csv_path = os.path.join(output_dir, "category_metrics.csv")
        self._csv_file = None
        self._csv_writer = None

    def _ensure_csv(self):
        if self._csv_writer is not None:
            return
        self._csv_file = open(self.csv_path, "w", newline="")
        header = ["step"]
        for name in CAT_SHORT_NAMES:
            header.extend([
                f"margin/{name}",
                f"loss/{name}",
                f"prefprob/{name}",
                f"count/{name}",
            ])
        header.extend(["margin_std", "loss_std", "prefprob_std"])
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(header)

    def update(
        self,
        category_indices_batch: list[list[int]],
        margins: torch.Tensor,
        losses: torch.Tensor,
        prefprobs: torch.Tensor,
    ):
        """Accumulate per-category stats from one batch."""
        margins = margins.detach().cpu()
        losses = losses.detach().cpu()
        prefprobs = prefprobs.detach().cpu()

        for i, cats in enumerate(category_indices_batch):
            if isinstance(cats, torch.Tensor):
                cats = cats.tolist()
            for c in cats:
                self._margin_sum[c] += margins[i]
                self._loss_sum[c] += losses[i]
                self._prefprob_sum[c] += prefprobs[i]
                self._count[c] += 1

        self.step += 1
        if self.step % self.log_every == 0:
            self._flush()

    def _flush(self):
        """Compute averages and write to wandb + CSV."""
        self._ensure_csv()

        avg_margin = torch.zeros(NUM_CATEGORIES)
        avg_loss = torch.zeros(NUM_CATEGORIES)
        avg_prefprob = torch.zeros(NUM_CATEGORIES)

        for c in range(NUM_CATEGORIES):
            if self._count[c] > 0:
                avg_margin[c] = self._margin_sum[c] / self._count[c]
                avg_loss[c] = self._loss_sum[c] / self._count[c]
                avg_prefprob[c] = self._prefprob_sum[c] / self._count[c]

        active = self._count > 0
        active_margins = avg_margin[active].numpy()
        active_losses = avg_loss[active].numpy()
        active_prefprobs = avg_prefprob[active].numpy()

        margin_std = float(active_margins.std()) if len(active_margins) > 1 else 0.0
        loss_std = float(active_losses.std()) if len(active_losses) > 1 else 0.0
        prefprob_std = float(active_prefprobs.std()) if len(active_prefprobs) > 1 else 0.0

        log_dict = {
            "cat/margin_std": margin_std,
            "cat/loss_std": loss_std,
            "cat/prefprob_std": prefprob_std,
        }
        for c in range(NUM_CATEGORIES):
            if self._count[c] > 0:
                name = CAT_SHORT_NAMES[c]
                log_dict[f"cat/margin/{name}"] = float(avg_margin[c])
                log_dict[f"cat/loss/{name}"] = float(avg_loss[c])
                log_dict[f"cat/prefprob/{name}"] = float(avg_prefprob[c])

        if wandb.run is not None:
            wandb.log(log_dict)

        row = [self.step]
        for c in range(NUM_CATEGORIES):
            if self._count[c] > 0:
                row.extend([
                    f"{avg_margin[c]:.6f}",
                    f"{avg_loss[c]:.6f}",
                    f"{avg_prefprob[c]:.6f}",
                    f"{int(self._count[c])}",
                ])
            else:
                row.extend(["", "", "", "0"])
        row.extend([f"{margin_std:.6f}", f"{loss_std:.6f}", f"{prefprob_std:.6f}"])
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        self._margin_sum.zero_()
        self._loss_sum.zero_()
        self._prefprob_sum.zero_()
        self._count.zero_()

    def close(self):
        if self._csv_file is not None:
            self._csv_file.close()
