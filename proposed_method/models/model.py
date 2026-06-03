import torch
import torch.nn as nn
from transformers import AutoModel, MBartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from torch_geometric.utils import to_dense_batch

from .adapters import GATAdapterLarge, CrossAttentionFusionLayer

def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (the <eos> token) to the beginning."""
    if input_ids is None:
        return None
    
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = 2  # mBART EOS token is typically 2
    
    # Replace possible -100 values in shifted_input_ids with pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    
    return shifted_input_ids

class Siglip2MBartCaptioner(nn.Module):
    def __init__(self, siglip_name="google/siglip2-large-patch16-256", mbart_name="facebook/mbart-large-50", 
                 use_gate=True, use_fusion=True, use_mbart_encoder=False, num_gat_layers=9, num_heads=8,
                 freeze_mbart_decoder=False, use_lora=False, lora_r=16, lora_alpha=32,
                 gat_dropout=0.1, label_smoothing=0.0):
        super().__init__()
        
        self._is_peft = False
        self.label_smoothing = label_smoothing
        
        # 1. Vision & Text Encoders (SigLIP 2 Large) -> Outputs 1024-dim
        # Load with trust_remote_code=True to avoid sys.modules KeyError on MoE check
        self.siglip = AutoModel.from_pretrained(
            siglip_name,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        # FREEZE SigLIP completely to save VRAM (no gradients, no activation maps stored)
        self.siglip.eval()
        for param in self.siglip.parameters():
            param.requires_grad = False
        
        # 2. Decoder (mBART-50) -> Hidden size is 1024-dim
        self.mbart = MBartForConditionalGeneration.from_pretrained(mbart_name)
        self.config = self.mbart.config

        if freeze_mbart_decoder:
            print("Freezing mBART Decoder (Stage 1: train only GAT Adapter + Projector)")
            for param in self.mbart.model.decoder.parameters():
                param.requires_grad = False
            for param in self.mbart.lm_head.parameters():
                param.requires_grad = False
        elif use_lora:
            print(f"Applying LoRA to mBART Decoder (r={lora_r}, alpha={lora_alpha})")
            try:
                from peft import get_peft_model, LoraConfig
                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,

                    target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
                    lora_dropout=0.05,
                    bias="none",
                )
                self.mbart = get_peft_model(self.mbart, lora_config)
                self._is_peft = True
                self.mbart.print_trainable_parameters()
            except Exception as e:
                print(f"WARNING: LoRA failed ({type(e).__name__}: {e}). Freezing decoder instead.")
                for param in self.mbart.model.decoder.parameters():
                    param.requires_grad = False
                for param in self.mbart.lm_head.parameters():
                    param.requires_grad = False
        
        # 2.9 Spatial Bbox Projector: [cx, cy, w, h] → 1024-dim
        # Added to node features BEFORE GAT so the adapter propagates spatial context
        self.spatial_proj = nn.Linear(4, 1024)

        # 3. GAT Adapter - NO pooling, returns all node features
        self.graph_adapter = GATAdapterLarge(
            input_dim=1024,
            hidden_dim=1024,
            num_layers=num_gat_layers,
            num_heads=num_heads,
            dropout=gat_dropout
        )
        
        # 3.5 Semantic Projector: Align SigLIP node space to mBART semantic space
        # LayerNorm MUST be at the end — normalizes output magnitude to match mBART hidden space
        self.semantic_projector = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
        )
        
        # 4. Gate Fusion Layer 
        self.use_fusion = use_fusion
        self.use_mbart_encoder = use_mbart_encoder  # For image captioning, set to False
        if self.use_fusion:
            self.fusion_layer = CrossAttentionFusionLayer(
                d_model=1024,
                nhead=num_heads,
                use_gate=use_gate
            )

    @property
    def _decoder(self):
        """Get mBART Decoder module, handling peft (LoRA) wrapping.
        
        Without peft: self.mbart (MBartForConditionalGeneration) → .model (MBartModel) → .decoder
        With peft:    self.mbart (PeftModel) → .model (MBartForConditionalGeneration) → .model (MBartModel) → .decoder
        """
        if self._is_peft:
            return self.mbart.model.model.decoder
        return self.mbart.model.decoder

    @property
    def _lm_head(self):
        """Get lm_head, handling peft (LoRA) wrapping."""
        if self._is_peft:
            return self.mbart.model.lm_head
        return self.mbart.lm_head
            
    def forward(self, pixel_values, node_input_ids, edge_index, batch=None, node_bboxes=None, mbart_input_ids=None, labels=None):
        """
        Forward pass - Image Captioning with GAT Adapter.
        
        Args:
            pixel_values: [B, 3, H, W] - Pixel values of the image
            node_input_ids: [num_nodes, max_len] - Token IDs of node texts
            edge_index: [2, num_edges] - Edge index of the graph  
            batch: [num_nodes] - Batch tensor indicating which node belongs to which image [0,0,0,1,1,2,...]
            mbart_input_ids: [B, max_seq_len] - Decoder input token IDs (not used because shifted from labels)
            labels: [B, max_seq_len] - Ground truth target IDs
        
        Returns:
            outputs: Seq2SeqLMOutput with loss, logits, decoder_hidden_states
        """
        # ===== A. EXTRACT IMAGE FEATURES (FROZEN SigLIP — no grad needed) =====
        with torch.no_grad():
            image_features = self.siglip.get_image_features(pixel_values=pixel_values)
            if not isinstance(image_features, torch.Tensor):
                if hasattr(image_features, "image_embeds"):
                    image_features = image_features.image_embeds
                elif hasattr(image_features, "pooler_output"):
                    image_features = image_features.pooler_output
                elif isinstance(image_features, dict):
                    image_features = list(image_features.values())[0]
                else:
                    image_features = image_features[0]

        # ===== B. EXTRACT NODE FEATURES (FROZEN SigLIP — no grad needed) =====
        with torch.no_grad():
            node_features = self.siglip.get_text_features(input_ids=node_input_ids)
            if not isinstance(node_features, torch.Tensor):
                if hasattr(node_features, "text_embeds"):
                    node_features = node_features.text_embeds
                elif hasattr(node_features, "pooler_output"):
                    node_features = node_features.pooler_output
                elif isinstance(node_features, dict):
                    node_features = list(node_features.values())[0]
                else:
                    node_features = node_features[0]
        
        # Inject spatial embeddings into node features BEFORE GAT
            node_features = node_features + self.spatial_proj(node_bboxes.to(node_features.device, node_features.dtype))

        # GAT Adapter
        graph_features = self.graph_adapter(node_features, edge_index, batch=batch)
        
        # Semantic alignment projector
        graph_features = self.semantic_projector(graph_features)
        
        # Convert from sparse to dense: [num_all_nodes, 1024] → [B, max_num_nodes, 1024], with graph_mask indicating valid nodes
        dense_graph_features, graph_mask = to_dense_batch(graph_features, batch)
        
        # Fusion layer: Image + graph
        if self.use_fusion and image_features is not None:
            image_kv = image_features.unsqueeze(1)
            fused_features = self.fusion_layer(
                query=dense_graph_features,
                key_value=image_kv
            )
        else:
            fused_features = dense_graph_features
        
        # Mbart decoder bypass encoder + decoder
        # CRITICAL: Shift labels to prepare decoder inputs
        if labels is not None:
            decoder_input_ids = shift_tokens_right(labels, self.mbart.config.pad_token_id)
        else:
            # Fallback for inference (not used in training)
            decoder_input_ids = mbart_input_ids
        

        decoder_attention_mask = (decoder_input_ids != self.mbart.config.pad_token_id).long()

        # Call decoder directly - NOT the full model
        # encoder_hidden_states = fused_features from GAT + Fusion
        # encoder_attention_mask = graph_mask (1=valid node, 0=padding from to_dense_batch)
        decoder_outputs = self._decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=fused_features,
            encoder_attention_mask=graph_mask.long(),
            return_dict=True
        )
        
        # Compute logits from decoder outputs
        logits = self._lm_head(decoder_outputs.last_hidden_state)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=self.label_smoothing)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            decoder_hidden_states=decoder_outputs.hidden_states,
            encoder_last_hidden_state=fused_features,
        )

    def generate(self, pixel_values, node_input_ids, edge_index, batch=None, node_bboxes=None, **generate_kwargs):
        """
        Beam Search generation for BLEU-4 evaluation.
        Computes fused features then delegates to MBartForConditionalGeneration.generate().
        
        Args:
            pixel_values: [B, 3, H, W]
            node_input_ids: [num_nodes, max_len]
            edge_index: [2, num_edges]
            batch: [num_nodes] - PyG batch tensor
            **generate_kwargs: Passed to mbart.generate (e.g. num_beams, max_new_tokens, forced_bos_token_id)
        
        Returns:
            generated_ids: [B, seq_len] - Generated token IDs
        """
        with torch.no_grad():
            # Image features
            image_features = self.siglip.get_image_features(pixel_values=pixel_values)
            if not isinstance(image_features, torch.Tensor):
                if hasattr(image_features, "image_embeds"):
                    image_features = image_features.image_embeds
                elif hasattr(image_features, "pooler_output"):
                    image_features = image_features.pooler_output
                elif isinstance(image_features, dict):
                    image_features = list(image_features.values())[0]
                else:
                    image_features = image_features[0]

            # Node features
            node_features = self.siglip.get_text_features(input_ids=node_input_ids)
            if not isinstance(node_features, torch.Tensor):
                if hasattr(node_features, "text_embeds"):
                    node_features = node_features.text_embeds
                elif hasattr(node_features, "pooler_output"):
                    node_features = node_features.pooler_output
                elif isinstance(node_features, dict):
                    node_features = list(node_features.values())[0]
                else:
                    node_features = node_features[0]

            # Inject spatial embeddings
            if node_bboxes is not None:
                node_features = node_features + self.spatial_proj(node_bboxes.to(node_features.device, node_features.dtype))

            # GAT + Projector + Fusion
            graph_features = self.graph_adapter(node_features, edge_index, batch=batch)
            graph_features = self.semantic_projector(graph_features)
            dense_graph_features, graph_mask = to_dense_batch(graph_features, batch)

            if self.use_fusion and image_features is not None:
                image_kv = image_features.unsqueeze(1)
                fused_features = self.fusion_layer(query=dense_graph_features, key_value=image_kv)
            else:
                fused_features = dense_graph_features

            encoder_outputs = BaseModelOutput(last_hidden_state=fused_features)

            # Generate via mBART
            generate_kwargs.setdefault(
                "decoder_start_token_id",
                getattr(self.mbart.generation_config, "decoder_start_token_id", 2)
            )
            if "forced_bos_token_id" not in generate_kwargs:
                bos_id = getattr(self.mbart.generation_config, "forced_bos_token_id", None)
                if bos_id is None:
                    import warnings
                    warnings.warn(
                        "[Siglip2MBartCaptioner] forced_bos_token_id is not set — mBART decoder "
                        "may drift to German or other languages (multilingual leakage). "
                        "Pass forced_bos_token_id=tokenizer.lang_code_to_id['en_XX'] to model.generate().",
                        UserWarning, stacklevel=2
                    )
                else:
                    generate_kwargs["forced_bos_token_id"] = bos_id

        
            return self.mbart.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=graph_mask.long(),  # encoder attention mask → blocks padding nodes
                **generate_kwargs
            )

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None,
                                       attention_mask=None, encoder_outputs=None,
                                       use_cache=True, **kwargs):
        if past_key_values is not None:
            # KV-cache active: only feed the most recent token
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,                  #
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,   # encoder attention mask for cross-attention
            "use_cache": use_cache,
        }
