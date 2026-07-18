import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoImageProcessor, MBart50TokenizerFast

from .data import xywh_to_bbox_indicator_features
from .model import BBoxAwareSigLIP2MBartCaptioner


def load_model(repo_id_or_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repo_path = Path(repo_id_or_path)

    if not repo_path.exists():
        repo_path = Path(
            snapshot_download(
                repo_id=repo_id_or_path,
                repo_type="model",
            )
        )

    with (repo_path / "model_config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)

    image_processor = AutoImageProcessor.from_pretrained(repo_path / "image_processor")
    tokenizer = MBart50TokenizerFast.from_pretrained(repo_path / "tokenizer")

    model = BBoxAwareSigLIP2MBartCaptioner(
        vision_model_name=config["vision_model_name"],
        mbart_name=config["text_model_name"],
        tgt_lang_code=config.get("tgt_lang_code", "en_XX"),
        freeze_vision=config.get("freeze_vision", True),
        use_lora=config.get("use_lora", True),
        lora_r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.1),
        lora_target_modules=config.get("lora_target_modules", ["q_proj", "v_proj"]),
        visual_projector_hidden_dim=config.get("visual_projector_hidden_dim", None),
        visual_projector_dropout=config.get("visual_projector_dropout", 0.1),
        bbox_indicator_dim=config.get("bbox_indicator_dim", 4),
        use_token_type_embeddings=config.get("use_token_type_embeddings", False),
        use_gradient_checkpointing=config.get("use_gradient_checkpointing", False),
    )

    state_dict = torch.load(
        repo_path / "model_state.pt",
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, tokenizer, image_processor, config, device


def make_bbox_features_from_xywh(bbox_xywh, image_width=None, image_height=None, device=None, ref=1000.0):
    """
    Build bbox-indicator features from raw [x, y, width, height].

    image_width and image_height are accepted for backward compatibility with
    existing call sites.
    """
    bbox_features = xywh_to_bbox_indicator_features(
        bbox_xywh=bbox_xywh,
        ref=ref,
    )

    return torch.tensor([bbox_features], dtype=torch.float32, device=device)


@torch.no_grad()
def caption_image_with_bbox(
    model,
    tokenizer,
    image_processor,
    image_path,
    bbox_xywh,
    config,
    device,
):
    image = Image.open(image_path).convert("RGB")

    pixel_values = image_processor(
        images=[image],
        return_tensors="pt",
    )["pixel_values"].to(device)

    bbox_features = make_bbox_features_from_xywh(
        bbox_xywh=bbox_xywh,
        device=device,
        ref=config.get("bbox_indicator_ref", 1000.0),
    )

    generated_ids = model.generate(
        pixel_values=pixel_values,
        bbox_features=bbox_features,
        tokenizer=tokenizer,
        num_beams=config.get("num_beams", 4),
        max_new_tokens=config.get("max_new_tokens", 30),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(config.get("tgt_lang_code", "en_XX")),
        repetition_penalty=config.get("repetition_penalty", 1.1),
        no_repeat_ngram_size=config.get("no_repeat_ngram_size", 3),
        length_penalty=config.get("length_penalty", 1.0),
    )

    caption = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return caption.strip()


@torch.no_grad()
def predict_rows(
    model,
    tokenizer,
    image_processor,
    rows,
    image_dir,
    image_ext,
    config,
    device,
    batch_size=64,
):
    """
    Predict one caption for each region-level row.
    """
    outputs = []
    bbox_indicator_ref = config.get("bbox_indicator_ref", 1000.0)

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start: start + batch_size]

        images = []
        bbox_features = []
        valid_rows = []

        for row in batch_rows:
            image_path = Path(image_dir) / f"{row['image_id']}{image_ext}"

            if not image_path.exists():
                outputs.append({
                    "image_id": row["image_id"],
                    "region_id": row["region_id"],
                    "ground_truth": row.get("caption", ""),
                    "prediction": "",
                    "is_empty": True,
                    "error": f"Image not found: {image_path}",
                })
                continue

            image = Image.open(image_path).convert("RGB")
            images.append(image)
            valid_rows.append(row)

            bbox_features.append(
                xywh_to_bbox_indicator_features(
                    bbox_xywh=row["bbox"],
                    ref=bbox_indicator_ref,
                )
            )

        if not images:
            continue

        pixel_values = image_processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        bbox_features = torch.tensor(bbox_features, dtype=torch.float32, device=device)

        generated_ids = model.generate(
            pixel_values=pixel_values,
            bbox_features=bbox_features,
            tokenizer=tokenizer,
            num_beams=config.get("num_beams", 4),
            max_new_tokens=config.get("max_new_tokens", 30),
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(config.get("tgt_lang_code", "en_XX")),
            repetition_penalty=config.get("repetition_penalty", 1.1),
            no_repeat_ngram_size=config.get("no_repeat_ngram_size", 3),
            length_penalty=config.get("length_penalty", 1.0),
        )

        predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for row, pred in zip(valid_rows, predictions):
            pred = str(pred).strip()
            gt = str(row.get("caption", "")).strip()

            outputs.append({
                "image_id": row["image_id"],
                "region_id": row["region_id"],
                "raw_bbox": row.get("bbox"),
                "ground_truth": gt,
                "prediction": pred,
                "is_empty": pred == "",
                "same_as_ground_truth": pred.lower() == gt.lower(),
                "error": "",
            })

    return pd.DataFrame(outputs)
