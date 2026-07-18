import json
import shutil
from datetime import datetime
from pathlib import Path

import torch


def save_processor_and_tokenizer(checkpoint_dir, tokenizer, image_processor):
    checkpoint_dir = Path(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
    image_processor.save_pretrained(checkpoint_dir / "image_processor")


def save_checkpoint(
    checkpoint_dir,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    batch_idx,
    epoch_completed,
    global_step,
    best_score,
    training_log,
    training_config,
    tokenizer=None,
    image_processor=None,
    save_training_state=True,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), checkpoint_dir / "model_state.pt")

    if save_training_state:
        training_state = {
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": int(epoch),
            "batch_idx": None if batch_idx is None else int(batch_idx),
            "epoch_completed": bool(epoch_completed),
            "global_step": int(global_step),
            "best_score": float(best_score),
            "training_log": training_log,
            "training_config": training_config,
        }

        torch.save(training_state, checkpoint_dir / "training_state.pt")

    if tokenizer is not None and image_processor is not None:
        save_processor_and_tokenizer(checkpoint_dir, tokenizer, image_processor)

    with (checkpoint_dir / "checkpoint_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "epoch": int(epoch),
                "batch_idx": None if batch_idx is None else int(batch_idx),
                "epoch_completed": bool(epoch_completed),
                "global_step": int(global_step),
                "best_score": float(best_score),
                "saved_at": datetime.utcnow().isoformat(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if training_config is not None:
        with (checkpoint_dir / "model_config.json").open("w", encoding="utf-8") as f:
            json.dump(
                training_config,
                f,
                ensure_ascii=False,
                indent=2,
            )


def load_checkpoint(checkpoint_dir, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    checkpoint_dir = Path(checkpoint_dir)

    model_state_file = checkpoint_dir / "model_state.pt"
    training_state_file = checkpoint_dir / "training_state.pt"

    if not model_state_file.exists():
        raise FileNotFoundError(f"Missing model state: {model_state_file}")

    state_dict = torch.load(model_state_file, map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict, strict=True)

    if not training_state_file.exists():
        return {}

    training_state = torch.load(training_state_file, map_location=map_location, weights_only=False)

    if optimizer is not None and training_state.get("optimizer") is not None:
        optimizer.load_state_dict(training_state["optimizer"])

    if scheduler is not None and training_state.get("scheduler") is not None:
        scheduler.load_state_dict(training_state["scheduler"])

    if scaler is not None and training_state.get("scaler") is not None:
        scaler.load_state_dict(training_state["scaler"])

    return training_state


def push_folder_to_hub(local_folder, repo_id, commit_message, token=None, private=True):
    from huggingface_hub import HfApi, create_repo

    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=token)

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(local_folder),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
