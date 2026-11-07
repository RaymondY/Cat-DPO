"""Data loading and filtering for PKU-SafeRLHF dataset.

Supports multiple data modes for dual-objective (helpfulness + safety) training:
  - helpful: chosen = more helpful (better_response_id), excludes unsafe-unsafe
  - harmful: chosen = safer (safer_response_id), excludes unsafe-unsafe
  - bettersafe: chosen = response that is BOTH more helpful AND safer
                (better_response_id == safer_response_id), excludes unsafe-unsafe
"""

from datasets import load_dataset

HARM_CATEGORIES = [
    "Endangering National Security",
    "Insulting Behavior",
    "Discriminatory Behavior",
    "Endangering Public Health",
    "Copyright Issues",
    "Violence",
    "Drugs",
    "Privacy Violation",
    "Economic Crime",
    "Mental Manipulation",
    "Human Trafficking",
    "Physical Harm",
    "Sexual Content",
    "Cybercrime",
    "Disrupting Public Order",
    "Environmental Damage",
    "Psychological Harm",
    "White-Collar Crime",
    "Animal Abuse",
]


def _extract_categories(row, chosen_id):
    """Extract harm categories from the rejected (non-chosen) response."""
    rejected_id = 1 - chosen_id
    harm_dict = row[f"response_{rejected_id}_harm_category"]

    categories = []
    category_indices = []
    for idx, cat_name in enumerate(HARM_CATEGORIES):
        if harm_dict.get(cat_name, False):
            categories.append(cat_name)
            category_indices.append(idx)

    if not categories:
        harm_dict_chosen = row[f"response_{chosen_id}_harm_category"]
        for idx, cat_name in enumerate(HARM_CATEGORIES):
            if harm_dict_chosen.get(cat_name, False):
                categories.append(cat_name)
                category_indices.append(idx)

    return categories, category_indices


def _build_example(row, chosen_id, categories, category_indices):
    """Build a training example dict."""
    rejected_id = 1 - chosen_id
    return {
        "prompt": row["prompt"],
        "chosen": row[f"response_{chosen_id}"],
        "rejected": row[f"response_{rejected_id}"],
        "categories": categories,
        "category_indices": category_indices,
    }


def load_helpful_pairs(split: str = "train", subset: str = "alpaca2-7b") -> list[dict]:
    """Load pairs where chosen = more helpful response. Excludes unsafe-unsafe."""
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", subset, split=split)

    examples = []
    for row in ds:
        if not row["is_response_0_safe"] and not row["is_response_1_safe"]:
            continue
        chosen_id = row["better_response_id"]
        categories, category_indices = _extract_categories(row, chosen_id)
        examples.append(_build_example(row, chosen_id, categories, category_indices))

    return examples


def load_harmful_pairs(split: str = "train", subset: str = "alpaca2-7b") -> list[dict]:
    """Load pairs where chosen = safer response. Excludes unsafe-unsafe."""
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", subset, split=split)

    examples = []
    for row in ds:
        if not row["is_response_0_safe"] and not row["is_response_1_safe"]:
            continue
        chosen_id = row["safer_response_id"]
        categories, category_indices = _extract_categories(row, chosen_id)
        examples.append(_build_example(row, chosen_id, categories, category_indices))

    return examples


def load_bettersafe_pairs(split: str = "train", subset: str = "alpaca2-7b") -> list[dict]:
    """Load pairs where better_response_id == safer_response_id (no conflict).

    Excludes unsafe-unsafe pairs.
    """
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", subset, split=split)

    examples = []
    for row in ds:
        if not row["is_response_0_safe"] and not row["is_response_1_safe"]:
            continue
        if row["better_response_id"] != row["safer_response_id"]:
            continue
        chosen_id = row["better_response_id"]
        categories, category_indices = _extract_categories(row, chosen_id)
        examples.append(_build_example(row, chosen_id, categories, category_indices))

    return examples


# Data mode dispatcher
DATA_LOADERS = {
    "helpful": load_helpful_pairs,
    "harmful": load_harmful_pairs,
    "bettersafe": load_bettersafe_pairs,
}


def load_pairs(data_mode: str, split: str = "train", subset: str = "alpaca2-7b") -> list[dict]:
    """Load data pairs by mode. Unified entry point."""
    if data_mode not in DATA_LOADERS:
        raise ValueError(f"Unknown data_mode: {data_mode}. Choose from {list(DATA_LOADERS.keys())}")
    return DATA_LOADERS[data_mode](split=split, subset=subset)


def format_for_dpo(examples: list[dict], conversational: bool = False) -> dict:
    """Convert to HuggingFace DPO format."""
    if conversational:
        result = {
            "prompt": [[{"role": "user", "content": ex["prompt"]}] for ex in examples],
            "chosen": [[{"role": "assistant", "content": ex["chosen"]}] for ex in examples],
            "rejected": [[{"role": "assistant", "content": ex["rejected"]}] for ex in examples],
            "category_indices": [ex["category_indices"] for ex in examples],
        }
    else:
        result = {
            "prompt": [ex["prompt"] for ex in examples],
            "chosen": [ex["chosen"] for ex in examples],
            "rejected": [ex["rejected"] for ex in examples],
            "category_indices": [ex["category_indices"] for ex in examples],
        }
    return result


def get_category_stats(examples: list[dict]) -> dict:
    """Compute dataset statistics per category."""
    from collections import Counter
    cat_counts = Counter()
    for ex in examples:
        for cat in ex["categories"]:
            cat_counts[cat] += 1

    total = len(examples)
    stats = {}
    for cat_name in HARM_CATEGORIES:
        count = cat_counts.get(cat_name, 0)
        stats[cat_name] = {"count": count, "fraction": count / total if total > 0 else 0}

    return stats
