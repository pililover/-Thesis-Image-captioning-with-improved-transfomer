from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import json
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import clean_text


DEFAULT_BBOX_KEYS = (
    "region_bbox",
    "bbox",
    "box",
    "bounding_box",
    "region_box",
    "xywh",
    "bounds",
)


def read_json_or_jsonl(path):
    """
    Read a region-level annotation file.

    Supported formats:
    - .jsonl: one JSON object per line
    - .json: a list of row dictionaries, or a dictionary containing one of:
      data, rows, samples, annotations
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
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

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            for key in ["data", "rows", "samples", "annotations"]:
                value = data.get(key)
                if isinstance(value, list):
                    return value

        raise ValueError(
            f"Unsupported JSON structure in {path}. "
            "Expected a list of row dictionaries or a dictionary with a supported list field."
        )

    raise ValueError(f"Unsupported file extension: {path.suffix}")


def extract_bbox_xywh(
    row: Dict[str, Any],
    bbox_keys: Sequence[str] = DEFAULT_BBOX_KEYS,
) -> Optional[List[float]]:
    """
    Extract a bbox in [x, y, width, height] format.

    The input row may contain a list, tuple, string, or dictionary bbox.
    Dictionary bboxes may use either width/height or x2/y2 coordinates.
    """
    value = None
    for key in bbox_keys:
        if key in row and row[key] is not None:
            value = row[key]
            break

    if value is None:
        return None

    if isinstance(value, dict):
        x = value.get("x", value.get("left", value.get("x1")))
        y = value.get("y", value.get("top", value.get("y1")))
        w = value.get("width", value.get("w"))
        h = value.get("height", value.get("h"))

        if w is None and "x2" in value and x is not None:
            w = float(value["x2"]) - float(x)
        if h is None and "y2" in value and y is not None:
            h = float(value["y2"]) - float(y)

        if x is None or y is None or w is None or h is None:
            return None

        try:
            bbox = [float(x), float(y), float(w), float(h)]
        except (TypeError, ValueError):
            return None

        return bbox if bbox[2] > 0 and bbox[3] > 0 else None

    if isinstance(value, str):
        value = value.replace("[", "").replace("]", "").replace(",", " ")
        parts = [p for p in value.split() if p]
    else:
        parts = list(value) if isinstance(value, Iterable) else []

    if len(parts) < 4:
        return None

    try:
        bbox = [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]
    except (TypeError, ValueError):
        return None

    return bbox if bbox[2] > 0 and bbox[3] > 0 else None


def xywh_to_bbox_indicator_features(
    bbox_xywh: Sequence[float],
    ref: float = 1000.0,
) -> List[float]:
    """
    Convert a raw region bbox into normalized target-region indicator features.

    Input:
        bbox_xywh: [x, y, width, height]

    Output:
        [center_x / ref, center_y / ref, width / ref, height / ref]

    A fixed coordinate reference keeps bbox features in a stable numeric range.
    """
    if ref <= 0:
        raise ValueError(f"ref must be positive, got {ref}")

    x, y, w, h = [float(v) for v in bbox_xywh[:4]]
    w = max(0.0, w)
    h = max(0.0, h)

    cx = min(max((x + w / 2.0) / ref, 0.0), 1.0)
    cy = min(max((y + h / 2.0) / ref, 0.0), 1.0)
    nw = min(max(w / ref, 0.0), 1.0)
    nh = min(max(h / ref, 0.0), 1.0)

    return [cx, cy, nw, nh]


class BBoxAwareImageCaptioningDataset(Dataset):
    """
    Region-level image captioning dataset.

    Each sample contains a full image, a target-region bbox, a caption, and
    lightweight metadata for prediction export.
    """

    def __init__(
        self,
        jsonl_file,
        image_dir,
        image_ext=".jpg",
        target_lang="en",
        bbox_keys: Sequence[str] = DEFAULT_BBOX_KEYS,
        limit: Optional[int] = None,
        check_image_exists: bool = False,
    ):
        self.jsonl_file = Path(jsonl_file)
        self.image_dir = Path(image_dir)
        self.image_ext = image_ext
        self.target_lang = target_lang
        self.bbox_keys = tuple(bbox_keys)

        rows = read_json_or_jsonl(self.jsonl_file)

        clean_rows = []
        missing_bbox = 0
        missing_text = 0
        missing_id = 0
        missing_image = 0

        for row in rows:
            image_id = clean_text(row.get("image_id"))
            region_id = clean_text(row.get("region_id"))

            if not image_id or not region_id:
                missing_id += 1
                continue

            if target_lang == "en":
                caption = clean_text(
                    row.get("text_en")
                    or row.get("en")
                    or row.get("caption_en")
                    or row.get("caption")
                )
            elif target_lang == "vi":
                caption = clean_text(
                    row.get("text_vi")
                    or row.get("vi")
                    or row.get("caption_vi")
                    or row.get("caption")
                )
            else:
                caption = clean_text(row.get("caption"))

            if not caption:
                missing_text += 1
                continue

            bbox_xywh = extract_bbox_xywh(row, self.bbox_keys)
            if bbox_xywh is None:
                missing_bbox += 1
                continue

            image_path = self.image_dir / f"{image_id}{self.image_ext}"
            if check_image_exists and not image_path.exists():
                missing_image += 1
                continue

            clean_rows.append({
                "image_id": image_id,
                "region_id": region_id,
                "caption": caption,
                "en": clean_text(row.get("text_en") or row.get("en")),
                "vi": clean_text(row.get("text_vi") or row.get("vi")),
                "bbox": bbox_xywh,
            })

        if limit is not None:
            clean_rows = clean_rows[: int(limit)]

        self.rows = clean_rows

        print(
            f"Loaded {len(self.rows):,} bbox-indicator samples from {self.jsonl_file.name} "
            f"with target_lang={self.target_lang}. "
            f"Skipped: missing_id={missing_id:,}, missing_text={missing_text:,}, "
            f"missing_bbox={missing_bbox:,}, missing_image={missing_image:,}"
        )

        if len(self.rows) == 0:
            raise ValueError(
                f"No valid samples found in {self.jsonl_file}. "
                f"Check target_lang={self.target_lang}, bbox keys={self.bbox_keys}, and input format."
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = self.image_dir / f"{row['image_id']}{self.image_ext}"

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        return {
            "image": image,
            "image_id": row["image_id"],
            "region_id": row["region_id"],
            "caption": row["caption"],
            "en": row["en"],
            "vi": row["vi"],
            "bbox": row["bbox"],
            "image_width": image_width,
            "image_height": image_height,
        }


class BBoxAwareImageCaptioningCollator:
    """
    Build model-ready batches for region-level captioning.

    Outputs:
    - pixel_values: processed image tensor
    - bbox_features: normalized [center_x, center_y, width, height] indicator features
    - labels: mBART target token ids with padding masked as -100
    - metadata fields used for prediction export
    """

    def __init__(
        self,
        image_processor,
        tokenizer,
        tgt_lang_code="en_XX",
        max_target_length=64,
        bbox_indicator_ref=1000.0,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.tgt_lang_code = tgt_lang_code
        self.max_target_length = max_target_length
        self.bbox_indicator_ref = float(bbox_indicator_ref)

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        captions = [item["caption"] for item in batch]

        pixel_values = self.image_processor(
            images=images,
            return_tensors="pt",
        )["pixel_values"]

        bbox_features = []
        raw_bboxes = []

        for item in batch:
            raw_bbox = item["bbox"]
            raw_bboxes.append(raw_bbox)

            bbox_feature = xywh_to_bbox_indicator_features(
                bbox_xywh=raw_bbox,
                ref=self.bbox_indicator_ref,
            )
            bbox_features.append(bbox_feature)

        bbox_features = torch.tensor(bbox_features, dtype=torch.float32)

        self.tokenizer.tgt_lang = self.tgt_lang_code

        try:
            target = self.tokenizer(
                text_target=captions,
                padding=True,
                truncation=True,
                max_length=self.max_target_length,
                return_tensors="pt",
            )
        except TypeError:
            with self.tokenizer.as_target_tokenizer():
                target = self.tokenizer(
                    captions,
                    padding=True,
                    truncation=True,
                    max_length=self.max_target_length,
                    return_tensors="pt",
                )

        labels = target["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "bbox_features": bbox_features,
            "region_bbox": bbox_features,
            "labels": labels,
            "image_id": [item["image_id"] for item in batch],
            "region_id": [item["region_id"] for item in batch],
            "caption": captions,
            "raw_bbox": raw_bboxes,
            "image_width": [item["image_width"] for item in batch],
            "image_height": [item["image_height"] for item in batch],
        }
