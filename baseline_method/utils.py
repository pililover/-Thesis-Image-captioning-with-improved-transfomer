import gc
import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Fix random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    """Read a JSONL file into a list of dictionaries."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

    return rows


def write_json(data, path) -> None:
    """Write a dictionary/list to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_text(text) -> str:
    """Convert text-like values to a stripped string."""
    if text is None:
        return ""
    return str(text).strip()


def cleanup_memory() -> None:
    """Release Python and CUDA cached memory."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def count_parameters(model):
    """Return total/trainable parameter statistics."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": int(total),
        "trainable": int(trainable),
        "trainable_pct": float(100 * trainable / max(1, total)),
    }
