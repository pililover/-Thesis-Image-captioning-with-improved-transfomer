import json
import random
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import sacrebleu
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from .data import BBoxAwareImageCaptioningCollator
from .utils import cleanup_memory


def move_batch_to_device(batch, device):
    output = {}

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.to(device, non_blocking=True)
        else:
            output[key] = value

    return output


def compute_loss_from_logits(logits, labels, label_smoothing=0.0):
    vocab_size = logits.size(-1)

    loss_fct = nn.CrossEntropyLoss(
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )

    return loss_fct(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
    )


def make_optimizer(model, learning_rate=3e-5, weight_decay=0.01):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (
            name.endswith(".bias")
            or "layernorm" in name.lower()
            or "layer_norm" in name.lower()
            or "norm" in name.lower()
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    use_fused = torch.cuda.is_available()

    try:
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=use_fused,
        )
    except TypeError:
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    return optimizer


def select_eval_indices(dataset_size, max_samples=None, seed=42):
    if max_samples is None or int(max_samples) >= dataset_size:
        return list(range(dataset_size))

    rng = random.Random(int(seed))
    return rng.sample(range(dataset_size), int(max_samples))


def pick_diverse_examples(rows, num_examples=5, seed=42):
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    selected = []
    seen_images = set()

    for row in shuffled:
        if row["image_id"] in seen_images:
            continue

        selected.append(row)
        seen_images.add(row["image_id"])

        if len(selected) >= num_examples:
            break

    if len(selected) < num_examples:
        for row in shuffled:
            if row not in selected:
                selected.append(row)

            if len(selected) >= num_examples:
                break

    return selected


@torch.no_grad()
def evaluate_loss(
    model,
    dataloader,
    device,
    label_smoothing=0.0,
    use_bf16=False,
    use_fp16=False,
):
    model.eval()

    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    total_loss = 0.0
    total_batches = 0

    for batch in tqdm(dataloader, desc="eval loss", leave=False):
        batch = move_batch_to_device(batch, device)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=(use_bf16 or use_fp16),
        ):
            outputs = model(
                pixel_values=batch["pixel_values"],
                bbox_features=batch["bbox_features"],
                labels=batch["labels"],
            )

            loss = compute_loss_from_logits(
                logits=outputs.logits,
                labels=batch["labels"],
                label_smoothing=label_smoothing,
            )

        total_loss += float(loss.detach().cpu())
        total_batches += 1

    cleanup_memory()

    return total_loss / max(1, total_batches)


@torch.no_grad()
def evaluate_generation(
    model,
    dataset,
    tokenizer,
    image_processor,
    device,
    tgt_lang_code="en_XX",
    max_target_length=64,
    eval_batch_size=128,
    num_workers=4,
    pin_memory=True,
    max_samples=None,
    num_examples=5,
    sample_seed=42,
    output_path=None,
    split_name="val",
    num_beams=4,
    max_new_tokens=30,
    repetition_penalty=1.1,
    no_repeat_ngram_size=3,
    length_penalty=1.0,
    use_bf16=False,
    use_fp16=False,
):
    model.eval()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    indices = select_eval_indices(
        dataset_size=len(dataset),
        max_samples=max_samples,
        seed=sample_seed,
    )

    eval_subset = torch.utils.data.Subset(dataset, indices)

    eval_collator = BBoxAwareImageCaptioningCollator(
        image_processor=image_processor,
        tokenizer=tokenizer,
        tgt_lang_code=tgt_lang_code,
        max_target_length=max_target_length,
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_subset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=eval_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    predictions = []
    references = []
    rows = []

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)

    for batch in tqdm(eval_loader, desc=f"generation eval ({split_name})", leave=False):
        batch = move_batch_to_device(batch, device)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=(use_bf16 or use_fp16),
        ):
            generated_ids = model.generate(
                pixel_values=batch["pixel_values"],
                bbox_features=batch["bbox_features"],
                tokenizer=tokenizer,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                forced_bos_token_id=forced_bos_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty,
            )

        batch_predictions = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        for image_id, region_id, raw_bbox, reference, prediction in zip(
            batch["image_id"],
            batch["region_id"],
            batch["raw_bbox"],
            batch["caption"],
            batch_predictions,
        ):
            pred = str(prediction).strip()
            ref = str(reference).strip()

            predictions.append(pred)
            references.append(ref)

            rows.append({
                "split": split_name,
                "image_id": image_id,
                "region_id": region_id,
                "raw_bbox": raw_bbox,
                "ground_truth": ref,
                "prediction": pred,
                "is_empty": pred == "",
                "same_as_ground_truth": pred.lower() == ref.lower(),
            })

    bleu = sacrebleu.corpus_bleu(predictions, [references]).score if predictions else float("nan")
    chrf = sacrebleu.corpus_chrf(predictions, [references]).score if predictions else float("nan")

    predictions_df = pd.DataFrame(rows)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Full {split_name} predictions saved to: {output_path}")
        print(f"Saved rows: {len(predictions_df):,}")

    example_rows = pick_diverse_examples(
        rows,
        num_examples=num_examples,
        seed=sample_seed,
    )

    cleanup_memory()

    return {
        "bleu": float(bleu),
        "chrf": float(chrf),
        "num_samples": len(predictions),
        "examples": rows,
        "qualitative_examples": example_rows,
        "predictions_path": str(output_path) if output_path is not None else None,
    }


def show_and_save_qualitative_examples(gen_metrics, epoch, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qualitative_rows = gen_metrics.get("qualitative_examples", [])
    qualitative_df = pd.DataFrame(qualitative_rows)

    qualitative_path = output_dir / f"epoch_{epoch + 1:03d}_qualitative_predictions.csv"
    qualitative_df.to_csv(qualitative_path, index=False, encoding="utf-8-sig")

    print("Qualitative predictions saved to:", qualitative_path)

    display_cols = [
        "image_id",
        "region_id",
        "raw_bbox",
        "ground_truth",
        "prediction",
        "is_empty",
    ]
    existing_cols = [col for col in display_cols if col in qualitative_df.columns]

    if len(qualitative_df) > 0:
        return qualitative_df[existing_cols]

    return qualitative_df


def summarize_prediction_diversity(predictions_df):
    return (
        predictions_df
        .groupby("image_id")
        .agg(
            num_regions=("region_id", "count"),
            unique_ground_truth=("ground_truth", "nunique"),
            unique_predictions=("prediction", "nunique"),
            first_prediction=("prediction", "first"),
        )
        .reset_index()
        .sort_values(["num_regions", "unique_ground_truth"], ascending=False)
    )


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch,
    global_step,
    grad_accum_steps=1,
    max_grad_norm=1.0,
    label_smoothing=0.0,
    use_bf16=False,
    use_fp16=False,
    start_batch_idx=0,
):
    model.train()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    running_loss = 0.0
    processed_batches = 0

    progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"epoch {epoch + 1}",
    )

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in progress_bar:
        if batch_idx < start_batch_idx:
            continue

        batch = move_batch_to_device(batch, device)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=(use_bf16 or use_fp16),
        ):
            outputs = model(
                pixel_values=batch["pixel_values"],
                bbox_features=batch["bbox_features"],
                labels=batch["labels"],
            )

            loss = compute_loss_from_logits(
                logits=outputs.logits,
                labels=batch["labels"],
                label_smoothing=label_smoothing,
            )

            loss_for_backward = loss / grad_accum_steps

        if use_fp16:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        running_loss += float(loss.detach().cpu())
        processed_batches += 1

        should_step = (
            ((batch_idx + 1) % grad_accum_steps == 0)
            or ((batch_idx + 1) == len(train_loader))
        )

        if should_step:
            if use_fp16:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm,
            )

            if use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        progress_bar.set_postfix(
            {
                "loss": f"{running_loss / max(1, processed_batches):.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "step": global_step,
            }
        )

    avg_loss = running_loss / max(1, processed_batches)
    return avg_loss, global_step
