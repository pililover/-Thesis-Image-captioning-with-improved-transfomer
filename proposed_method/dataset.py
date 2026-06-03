import json
import torch
import sys
import os
from torch.utils.data import Dataset

# Add data/proposed_method directory to path for process_vivg import
sys.path.insert(0, '/content/drive/MyDrive/[Thesis] Improved_transformer/data/proposed_method')
from process_vivg import process_vivg_graph

class ViVGDataset(Dataset):
    def __init__(self, json_path, siglip_processor, mbart_tokenizer, image_dir="/content/drive/MyDrive/[Thesis] Improved_transformer/img", lang="vi", target_length=128):
        """
        json_path: Path to the train.json or val.json file
        siglip_processor: AutoProcessor from "google/siglip2-large-patch16-256"
        mbart_tokenizer: Tokenizer of MBart50 ("vi_VN" or "en_XX")
        image_dir: Directory containing original images on Drive
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        self.siglip_processor = siglip_processor
        self.mbart_tokenizer = mbart_tokenizer
        self.image_dir = image_dir
        self.lang = lang
        self.target_length = target_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        graph_data = process_vivg_graph(item)
        
        siglip_inputs = self.siglip_processor(
            text=graph_data["node_texts"], 
            padding="max_length", 
            max_length=32, # max length for phrase nodes
            truncation=True, 
            return_tensors="pt"
        )
        
        # Remove the extra batch dimension from the processor (1, N, L) -> (N, L)
        node_input_ids = siglip_inputs["input_ids"].squeeze(0) if siglip_inputs["input_ids"].dim() == 3 else siglip_inputs["input_ids"]
        
        # Target Text (mBART-50 Tokenization)
        caption = graph_data["caption_vi"] if self.lang == "vi" else graph_data["caption_en"]
        
        # Process image (Pixel Values) 
        try:
            from PIL import Image
            import os
            # Try multiple extensions (.jpg, .jpeg, .png) for robustness
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
        except Exception as e:
            # If error or folder does not exist (during offline test build), replace with dummy tensor
            pixel_values = torch.zeros(3, 256, 256)
        
        # Ensure mBART decodes in the correct target language
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
        # Replace pad token id with -100 for cross-entropy loss computation
        labels[labels == self.mbart_tokenizer.pad_token_id] = -100
        
        return {
            "image_id": graph_data["image_id"],
            "node_input_ids": node_input_ids,
            "node_bboxes": graph_data["node_bboxes"],
            "edge_index": graph_data["edge_index"],
            "labels": labels,
            "pixel_values": pixel_values
        }
