import json
import torch
import sys
import os
from torch.utils.data import Dataset

sys.path.insert(0, '/content/drive/MyDrive/[Thesis] Improved_transformer/data/proposed_method')
from process_vivg import process_vivg_graph

class ViVGDataset(Dataset):
    def __init__(self, json_path, siglip_processor, mbart_tokenizer, image_dir="/content/drive/MyDrive/[Thesis] Improved_transformer/img", lang="vi", target_length=128, max_images=None, preloaded_data=None):
        if preloaded_data is not None:
            raw_data = preloaded_data
        else:
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

        # Flatten: one training sample per region (not per image).
        # max_images: if set, only use the first N images (but ALL their regions).
        if max_images is not None:
            raw_data = raw_data[:max_images]

        # Each (item, region) pair is independent: same image + global graph, different caption + region_bbox.
        self.samples = []
        for item in raw_data:
            regions = item.get('regions_mapping', {})
            region_list = list(regions.values()) if isinstance(regions, dict) else (regions if isinstance(regions, list) else [])
            for region in region_list:
                self.samples.append((item, region))

        self.siglip_processor = siglip_processor
        self.mbart_tokenizer = mbart_tokenizer
        self.image_dir = image_dir
        self.lang = lang
        self.target_length = target_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item, region = self.samples[idx]

        # Parse global scene graph (shared across all regions of this image)
        graph_data = process_vivg_graph(item)

        # Node tokens (SigLIP tokenization)
        siglip_inputs = self.siglip_processor(
            text=graph_data["node_texts"],
            padding="max_length",
            max_length=32,
            truncation=True,
            return_tensors="pt"
        )
        node_input_ids = siglip_inputs["input_ids"].squeeze(0) if siglip_inputs["input_ids"].dim() == 3 else siglip_inputs["input_ids"]

        # Caption from this specific region
        caption = region.get('text_vi', '') if self.lang == "vi" else region.get('text_en', '')

        # Region bbox (normalized) — prepended as indicator token in model
        region_bbox_raw = region.get('region_bbox', [0, 0, 0, 0])
        x, y, w, h = (list(region_bbox_raw) + [0, 0, 0, 0])[:4]
        _REF = 1000.0
        cx = min(max((x + w / 2) / _REF, 0.0), 1.0)
        cy = min(max((y + h / 2) / _REF, 0.0), 1.0)
        nw = min(max(w / _REF, 0.0), 1.0)
        nh = min(max(h / _REF, 0.0), 1.0)
        region_bbox_tensor = torch.tensor([cx, cy, nw, nh], dtype=torch.float32)

        # Region node mask — True for nodes belonging to this region's subgraph
        node_id_to_idx   = graph_data["node_id_to_idx"]
        num_nodes        = graph_data["num_nodes"]
        region_node_ids  = region.get("node_indices", [])
        region_node_mask = torch.zeros(num_nodes, dtype=torch.bool)
        for nid in region_node_ids:
            if nid in node_id_to_idx:
                region_node_mask[node_id_to_idx[nid]] = True
        # Fallback: if region has no node_indices, treat all nodes as in-region
        if not region_node_mask.any():
            region_node_mask = torch.ones(num_nodes, dtype=torch.bool)

        # Image (pixel values)
        try:
            from PIL import Image
            img_path = None
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = os.path.join(self.image_dir, f"{item['image_id']}{ext}")
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if img_path is None:
                raise FileNotFoundError(f"No image found for id {item['image_id']} in {self.image_dir}")
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.siglip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        except Exception:
            pixel_values = torch.zeros(3, 256, 256)

        # Labels (mBART tokenization of this region's caption)
        mbart_lang_code = "vi_VN" if self.lang == "vi" else "en_XX"
        self.mbart_tokenizer.tgt_lang = mbart_lang_code
        mbart_targets = self.mbart_tokenizer(
            caption,
            padding="max_length",
            max_length=self.target_length,
            truncation=True,
            return_tensors="pt"
        )
        labels = mbart_targets["input_ids"].squeeze(0)
        labels[labels == self.mbart_tokenizer.pad_token_id] = -100

        return {
            "image_id": graph_data["image_id"],
            "node_input_ids": node_input_ids,
            "node_bboxes": graph_data["node_bboxes"],
            "edge_index": graph_data["edge_index"],
            "region_bbox": region_bbox_tensor,
            "region_node_mask": region_node_mask,
            "labels": labels,
            "pixel_values": pixel_values
        }
