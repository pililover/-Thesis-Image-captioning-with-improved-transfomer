from typing import Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoModel, MBartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None


class SemanticProjector(nn.Module):
    """Project SigLIP-space visual tokens into the mBART hidden space."""

    def __init__(self, vision_dim=1024, text_dim=1024, hidden_dim=None, dropout=0.1):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = max(vision_dim, text_dim)

        self.net = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, text_dim),
        )

    def forward(self, visual_tokens):
        return self.net(visual_tokens)


class RegionIndicatorProjector(nn.Module):
    """
    Project target-region indicator features into the vision hidden space.

    Input:
        bbox_features: [B, 4] = [center_x, center_y, width, height]

    Output:
        region_indicator: [B, vision_dim]
    """

    def __init__(self, bbox_dim=4, vision_dim=1024):
        super().__init__()
        self.proj = nn.Linear(bbox_dim, vision_dim)

    def forward(self, bbox_features):
        return self.proj(bbox_features)


class BBoxAwareSigLIP2MBartCaptioner(nn.Module):
    """
    BBox-indicator visual baseline for region-level image captioning.

    The target region is represented as a learned indicator token:

        bbox_features [center_x, center_y, width, height]
            -> Linear(4, vision_dim)
            -> prepended before SigLIP image patch tokens

    Decoder memory:
        [region_indicator_token, patch_1, patch_2, ..., patch_N]
    """

    def __init__(
        self,
        vision_model_name,
        mbart_name,
        tgt_lang_code="en_XX",
        freeze_vision=True,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules: Optional[Sequence[str]] = None,
        visual_projector_hidden_dim=None,
        visual_projector_dropout=0.1,
        bbox_indicator_dim=4,
        use_token_type_embeddings=False,
        use_gradient_checkpointing=False,
    ):
        super().__init__()

        self.vision_model_name = vision_model_name
        self.mbart_name = mbart_name
        self.tgt_lang_code = tgt_lang_code
        self.freeze_vision = freeze_vision
        self.use_token_type_embeddings = bool(use_token_type_embeddings)
        self.bbox_indicator_dim = int(bbox_indicator_dim)

        siglip_backbone = AutoModel.from_pretrained(vision_model_name)

        if hasattr(siglip_backbone, "vision_model"):
            self.siglip = siglip_backbone.vision_model
        else:
            self.siglip = siglip_backbone

        if hasattr(self.siglip.config, "hidden_size"):
            vision_dim = self.siglip.config.hidden_size
        elif hasattr(self.siglip.config, "vision_config"):
            vision_dim = self.siglip.config.vision_config.hidden_size
        else:
            raise ValueError("Cannot infer SigLIP vision hidden size.")

        base_mbart = MBartForConditionalGeneration.from_pretrained(mbart_name)
        text_dim = base_mbart.config.d_model

        self.vision_dim = vision_dim
        self.text_dim = text_dim

        self.region_spatial_proj = RegionIndicatorProjector(
            bbox_dim=bbox_indicator_dim,
            vision_dim=vision_dim,
        )

        self.visual_projector = SemanticProjector(
            vision_dim=vision_dim,
            text_dim=text_dim,
            hidden_dim=visual_projector_hidden_dim,
            dropout=visual_projector_dropout,
        )

        if use_lora:
            if get_peft_model is None:
                raise ImportError("peft is required for use_lora=True")

            lora_target_modules = list(lora_target_modules or ["q_proj", "v_proj"])
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
            )
            self.mbart = get_peft_model(base_mbart, lora_config)
        else:
            self.mbart = base_mbart

        self.mbart.config.use_cache = not use_gradient_checkpointing

        if use_gradient_checkpointing:
            if hasattr(self.mbart, "gradient_checkpointing_enable"):
                self.mbart.gradient_checkpointing_enable()
            if hasattr(self.mbart, "enable_input_require_grads"):
                self.mbart.enable_input_require_grads()
        else:
            if hasattr(self.mbart, "gradient_checkpointing_disable"):
                self.mbart.gradient_checkpointing_disable()

        if self.use_token_type_embeddings:
            self.token_type_embeddings = nn.Embedding(2, text_dim)
        else:
            self.token_type_embeddings = None

        self.memory_layernorm = nn.LayerNorm(text_dim)

        if freeze_vision:
            for param in self.siglip.parameters():
                param.requires_grad = False

        if hasattr(self.siglip, "gradient_checkpointing_disable"):
            self.siglip.gradient_checkpointing_disable()

    def _add_token_type_embeddings(self, memory_tokens):
        if self.token_type_embeddings is None:
            return memory_tokens

        batch_size, seq_len, _ = memory_tokens.shape
        device = memory_tokens.device

        type_ids = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                torch.ones(seq_len - 1, dtype=torch.long, device=device),
            ],
            dim=0,
        )
        type_embeds = self.token_type_embeddings(type_ids).unsqueeze(0)
        return memory_tokens + type_embeds.expand(batch_size, -1, -1)

    def encode_image_with_bbox(self, pixel_values, bbox_features):
        """
        Encode the full image and the target-region indicator.

        Args:
            pixel_values: [B, 3, H, W]
            bbox_features: [B, 4] = [center_x, center_y, width, height]

        Returns:
            memory_tokens: [B, 1 + num_patches, text_dim]
            memory_attention: [B, 1 + num_patches]
            region_weights: None, kept for a stable return signature
        """
        if bbox_features is None:
            raise ValueError("bbox_features is required for bbox-indicator captioning.")

        if bbox_features.ndim != 2 or bbox_features.size(-1) != self.bbox_indicator_dim:
            raise ValueError(
                f"bbox_features must have shape [B, {self.bbox_indicator_dim}], "
                f"got {tuple(bbox_features.shape)}"
            )

        if self.freeze_vision:
            with torch.no_grad():
                vision_outputs = self.siglip(
                    pixel_values=pixel_values,
                    return_dict=True,
                )
        else:
            vision_outputs = self.siglip(
                pixel_values=pixel_values,
                return_dict=True,
            )

        patch_tokens = vision_outputs.last_hidden_state

        bbox_features = bbox_features.to(
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        )

        region_indicator = self.region_spatial_proj(bbox_features).unsqueeze(1)
        image_tokens = torch.cat([region_indicator, patch_tokens], dim=1)

        memory_tokens = self.visual_projector(image_tokens)
        memory_tokens = self._add_token_type_embeddings(memory_tokens)
        memory_tokens = self.memory_layernorm(memory_tokens)

        memory_attention = torch.ones(
            memory_tokens.size(0),
            memory_tokens.size(1),
            dtype=torch.long,
            device=memory_tokens.device,
        )

        return memory_tokens, memory_attention, None

    def forward(self, pixel_values, bbox_features, labels=None):
        memory_tokens, memory_attention, _ = self.encode_image_with_bbox(
            pixel_values=pixel_values,
            bbox_features=bbox_features,
        )

        encoder_outputs = BaseModelOutput(last_hidden_state=memory_tokens)

        outputs = self.mbart(
            encoder_outputs=encoder_outputs,
            attention_mask=memory_attention,
            labels=labels,
            return_dict=True,
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        pixel_values,
        bbox_features,
        tokenizer,
        num_beams=4,
        max_new_tokens=30,
        forced_bos_token_id=None,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
    ):
        self.eval()

        memory_tokens, memory_attention, _ = self.encode_image_with_bbox(
            pixel_values=pixel_values,
            bbox_features=bbox_features,
        )

        encoder_outputs = BaseModelOutput(last_hidden_state=memory_tokens)

        if forced_bos_token_id is None:
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(self.tgt_lang_code)

        generated_ids = self.mbart.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=memory_attention,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            forced_bos_token_id=forced_bos_token_id,
            decoder_start_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            early_stopping=(num_beams is not None and int(num_beams) > 1),
        )

        return generated_ids
