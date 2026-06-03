import torch
import numpy as np
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback
import argparse
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model import Siglip2MBartCaptioner

def parse_args():
    parser = argparse.ArgumentParser(description="Train SigLIP2-mBART with GAT Adapter")
    parser.add_argument("--data_dir", type=str, default="/content/drive/MyDrive/[Thesis] Improved_transformer/data/proposed_method", help="Path to data directory")
    # CRITICAL: Save to Google Drive, NOT /content (Colab local disk ~100GB). /content fills up fast with checkpoints.
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/[Thesis] Improved_transformer/checkpoints", help="Output directory (use Drive in Colab to avoid disk full)")
    parser.add_argument("--hf_repo_id", type=str, default=None, help="HuggingFace Repo ID (required when --push_to_hub)")
    parser.add_argument("--hf_token", type=str, default=None, help="HF Token for pushing to hub")
    parser.add_argument("--push_to_hub", action="store_true", default=False, help="Push checkpoints and final model to HuggingFace Hub")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint or 'True' to resume from last")

    # Architecture Arguments
    parser.add_argument("--mapping_network", type=str, default="gatl", help="Mapping network type")
    parser.add_argument("--num_gat_layers", type=int, default=9, help="Number of GAT layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads in GAT")
    parser.add_argument("--use_gate", action="store_true", default=True, help="Use gating mechanism in fusion")
    parser.add_argument("--use_fusion", action="store_true", default=True, help="Use fusion layer between image and graph")
    
    # Giải pháp 2: Decoder capacity control
    parser.add_argument("--freeze_decoder", action="store_true", default=False, help="Freeze mBART decoder entirely (Stage 1)")
    parser.add_argument("--use_lora", action="store_true", default=False, help="Apply LoRA to mBART decoder (r=8, alpha=16)")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (16 recommended; r=8 lacks capacity to overfit 10 samples)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling (keep at 2×rank)")
    
    # Training Arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for captioning")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--eval_strategy", type=str, default="epoch", help="Evaluation strategy")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="Save strategy (epoch/no/steps)")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Keep only N most recent checkpoints (saves disk space)")
    
    # Giải pháp 3: Regularization
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for GAT Adapter layers")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor (0.1 recommended)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
    
    # Hardware & Precision
    parser.add_argument("--bf16", action="store_true", default=False, help="Use bfloat16 (A100)")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--pin_memory", action="store_true", default=False, help="Pin memory for faster GPU transfers")
    parser.add_argument("--clear_cache", action="store_true", default=True, help="Clear HuggingFace/torch cache before training (frees ~2-3GB)")
    
    # Generation / Decoding
    parser.add_argument("--num_beams_eval", type=int, default=1, help="Beams for eval generation (1=greedy, more stable baseline)")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty during generation (1.0=off, 1.1-1.3 recommended)")
    
    # Logging
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps (set to 1 for isolation checks)")
    
    # Language Argument
    parser.add_argument("--lang", type=str, default="en", help="Language for captioning targets: 'en' or 'vi'")
    
    # Sanity-check / overfit mode
    parser.add_argument("--overfit_n", type=int, default=None, help="If set, limit train+eval to first N samples (overfit sanity check)")
    
    args = parser.parse_args()
    
    # Safety: If output_dir parent doesn't exist, fall back to local ./checkpoints
    import os
    output_parent = os.path.dirname(args.output_dir) or "."
    if not os.path.exists(output_parent):
        print(f"[WARNING] output_dir parent '{output_parent}' not found. Using ./checkpoints instead.")
        args.output_dir = "./checkpoints"
    
    return args

def run_training(args):
    """
    Main training function that can be called from CLI or notebook.
    
    Args:
        args: argparse.Namespace object with all required arguments
    """
    # ─── DISK SPACE MANAGEMENT ───────────────────────────────────────────
    # Colab /content local disk is ~100GB. With checkpoint overhead, fills up fast.
    # Solution 1: Clear HuggingFace cache to free ~2-3GB
    if getattr(args, 'clear_cache', True):
        import shutil
        cache_dirs = [
            os.path.expanduser("~/.cache/huggingface/hub"),
            os.path.expanduser("~/.cache/torch"),
        ]
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                try:
                    print(f"[Cache cleanup] Removing {cache_dir}")
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    print(f"  ✓ Freed ~2-3GB")
                except Exception as e:
                    print(f"  ✗ Could not clear: {e}")
    # Initialize Model with proper architecture arguments
    model = Siglip2MBartCaptioner(
        siglip_name="google/siglip2-large-patch16-256",
        mbart_name="facebook/mbart-large-50",
        use_gate=args.use_gate,
        use_fusion=args.use_fusion,
        use_mbart_encoder=False,  # Image captioning (skip source text encoder)
        num_gat_layers=args.num_gat_layers,
        num_heads=args.num_heads,
        # Giải pháp 2: Decoder capacity
        freeze_mbart_decoder=getattr(args, 'freeze_decoder', False),
        use_lora=getattr(args, 'use_lora', False),
        lora_r=getattr(args, 'lora_r', 8),
        lora_alpha=getattr(args, 'lora_alpha', 16),
        # Giải pháp 3: Regularization
        gat_dropout=getattr(args, 'dropout', 0.2),
        label_smoothing=getattr(args, 'label_smoothing', 0.1),
    )
    
    # Initialize your ViVG Dataset here
    from transformers import AutoProcessor, MBart50TokenizerFast
    from dataset import ViVGDataset
    from collate_fns import ViVGCollate
    
    siglip_proc = AutoProcessor.from_pretrained("google/siglip2-large-patch16-256")
    mbart_tok = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
    
    image_dir = os.path.join(args.data_dir, "../../img")
    
    # Determine mBART language code for forced_bos_token_id during generation
    lang_code = "vi_VN" if args.lang == "vi" else "en_XX"
    
    # Validate lang token ID — mBART-50 requires this to control the output language
    forced_bos_token_id = mbart_tok.convert_tokens_to_ids(lang_code)
    if forced_bos_token_id == mbart_tok.unk_token_id:
        raise ValueError(
            f"Language code '{lang_code}' not found in mBART-50 vocabulary. "
            f"Valid codes: vi_VN, en_XX, de_DE, fr_XX, ..."
        )
    print(f"Loading Datasets for language: {args.lang} | mBART lang_code: '{lang_code}' | forced_bos_token_id: {forced_bos_token_id}")

    # ── FIX: Lock target language into generation_config ONLY (new transformers API) ──
    # model.config must NOT have generation params — set them on generation_config instead.
    # Without this, Trainer calls generate() with no forced_bos_token_id → decoder outputs garbage.
    if not hasattr(model, "generation_config"):
        from transformers import GenerationConfig
        model.generation_config = GenerationConfig()
    # mBART-50 standard: decoder_start_token_id = eos_token_id (=2, i.e. </s>)
    # The language token is controlled ONLY via forced_bos_token_id.
    # Setting decoder_start_token_id = lang_id breaks the decoder startup sequence
    # (training always starts decoder with </s> via shift_tokens_right).
    model.generation_config.forced_bos_token_id = forced_bos_token_id
    model.generation_config.decoder_start_token_id = mbart_tok.eos_token_id  # = 2 (</s>)
    # Anti-repetition constraints — prevents self-reinforcing loops when cross-attention is weak
    model.generation_config.eos_token_id = mbart_tok.eos_token_id          # = 2 (</s>) — tell generate() when to stop
    model.generation_config.repetition_penalty = getattr(args, 'repetition_penalty', 1.2)
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 1.0
    model.generation_config.num_beams = getattr(args, 'num_beams_eval', 1)
    # max_new_tokens=30: captions are 5-15 words — 128 caused infinite hallucination
    # after correct keywords (model kept generating until hitting the cap)
    model.generation_config.max_new_tokens = 30
    print(f"Locked generation_config: forced_bos_token_id={forced_bos_token_id} ({lang_code}), "
          f"eos_token_id={mbart_tok.eos_token_id}, decoder_start_token_id={mbart_tok.eos_token_id} (eos)")

    train_dataset = ViVGDataset(
        json_path=os.path.join(args.data_dir, "train.json"),
        siglip_processor=siglip_proc,
        mbart_tokenizer=mbart_tok,
        image_dir=image_dir,
        lang=args.lang
    )
    
    eval_dataset = ViVGDataset(
        json_path=os.path.join(args.data_dir, "val.json"),
        siglip_processor=siglip_proc,
        mbart_tokenizer=mbart_tok,
        image_dir=image_dir,
        lang=args.lang
    )
    
    data_collator = ViVGCollate(pad_token_id=siglip_proc.tokenizer.pad_token_id)
    
    # ── Overfit sanity-check mode: slice to first N samples ──────────────────
    overfit_n = getattr(args, 'overfit_n', None)
    if overfit_n:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(min(overfit_n, len(train_dataset))))
        # FIX: eval = SAME samples as train — sanity check verifies the model CAN memorize,
        # not whether it generalizes. Using different val samples gives BLEU=0 trivially.
        eval_dataset  = train_dataset
        print(f"[Overfit mode] train={len(train_dataset)} samples | eval = SAME train samples")
    
    # ===== GIẢI PHÁP 1: Compute BLEU-4 trên Validation set =====
    def compute_metrics(eval_pred):
        """Tính BLEU-4 thực tế từ predictions sinh ra qua Beam Search."""
        predictions, labels = eval_pred
        # predictions: [N, seq_len] numpy int array (generated token ids)
        # labels: [N, seq_len] numpy int array (ground truth, -100 for padding)
        
        # Khi Trainer gom predictions từ nhiều batch có độ dài khác nhau,
        # các vị trí pad được điền bằng -100. Tokenizer Rust backend (MBart50TokenizerFast)
        # cố chuyển -100 thành u32 → OverflowError. Phải thay thế trước khi decode.
        predictions = np.where(predictions >= 0, predictions, mbart_tok.pad_token_id)
        decoded_preds = mbart_tok.batch_decode(predictions, skip_special_tokens=True)
        
        # Thay thế -100 bằng pad_token_id trước khi decode
        labels_for_decode = np.where(labels != -100, labels, mbart_tok.pad_token_id)
        decoded_labels = mbart_tok.batch_decode(labels_for_decode, skip_special_tokens=True)
        
        # Tính BLEU-4 (ưu tiên sacrebleu, fallback sang nltk)
        try:
            from sacrebleu.metrics import BLEU
            bleu = BLEU(max_ngram_order=4)
            bleu_score = bleu.corpus_score(decoded_preds, [decoded_labels]).score
        except ImportError:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            refs = [[ref.split()] for ref in decoded_labels]
            hyps = [pred.split() for pred in decoded_preds]
            bleu_score = corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1) * 100
        
        # --- DEBUG: uncomment to print sample predictions during eval (useful for sanity check)
        # import random as _random
        # n_show = min(5, len(decoded_preds))
        # show_indices = _random.sample(range(len(decoded_preds)), n_show)
        # for rank, i in enumerate(show_indices, 1):
        #     print(f"  [{rank}] GT  : {decoded_labels[i]}")
        #     print(f"  [{rank}] Pred: {decoded_preds[i]}")

        return {"bleu4": round(bleu_score, 4)}
    
    # Hugging Face TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        
        # Giải pháp 3: Regularization
        weight_decay=getattr(args, 'weight_decay', 0.01),
        
        remove_unused_columns=False,
        
        # Checkpointing Strategy — CRITICAL for Colab disk space
        # save_total_limit=N: keep only N most recent checkpoints (delete older ones)
        # For mBART-50 (~2.6GB), checkpoint overhead is huge. Set to 1-2 to save space.
        save_strategy=getattr(args, 'save_strategy', 'epoch'),
        save_total_limit=getattr(args, 'save_total_limit', 2),
        
        # Hugging Face Hub Integration
        push_to_hub=getattr(args, 'push_to_hub', False) and bool(getattr(args, 'hf_repo_id', None)),
        hub_model_id=args.hf_repo_id if getattr(args, 'push_to_hub', False) else None,
        hub_token=args.hf_token if getattr(args, 'push_to_hub', False) else None,
        hub_strategy="every_save" if getattr(args, 'push_to_hub', False) else "end",
        
        # Hardware
        bf16=getattr(args, 'bf16', False),
        dataloader_num_workers=getattr(args, 'dataloader_num_workers', 4),
        dataloader_pin_memory=getattr(args, 'pin_memory', False),
        
        # Giải pháp 1: Đánh giá bằng BLEU-4 (không phải eval_loss)
        eval_strategy=args.eval_strategy,
        logging_dir="./logs",
        logging_steps=getattr(args, 'logging_steps', 10),
        load_best_model_at_end=True,
        metric_for_best_model="bleu4",   # <-- BLEU-4 thay vì eval_loss
        greater_is_better=True,           # <-- BLEU càng cao càng tốt
        label_names=["labels"],           # <-- ensures Trainer knows which key is the label
    )
    
    # Custom Trainer class to handle our model's specific inputs
    class CustomTrainer(Trainer):
        # predict_with_generate=True: prediction_step sẽ dùng model.generate() (Beam Search)
        # thay vì teacher forcing để tính BLEU-4 thực tế trên Validation set.
        # (Tương đương flag predict_with_generate của Seq2SeqTrainer)
        predict_with_generate: bool = True

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Extract labels (will be used for loss computation in model)
            labels = inputs.pop("labels", None)
            
            # Call model with only expected arguments
            outputs = model(
                pixel_values=inputs.get("pixel_values"),
                node_input_ids=inputs.get("node_input_ids"),
                node_bboxes=inputs.get("node_bboxes", None),
                edge_index=inputs.get("edge_index"),
                batch=inputs.get("batch", None),
                labels=labels
            )
            
            loss = outputs.loss if outputs.loss is not None else None
            return (loss, outputs) if return_outputs else loss
        
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            """
            Override prediction_step để chạy Beam Search thực tế thay vì teacher forcing.
            Giải pháp 1: Đây là nơi BLEU-4 được tính - dùng model.generate() với num_beams=4.
            """
            inputs = self._prepare_inputs(inputs)
            labels = inputs.get("labels")
            
            with torch.no_grad():
                # Luôn tính loss (để log eval_loss song song với BLEU-4)
                loss, _ = self.compute_loss(model, dict(inputs), return_outputs=True)
                
                if prediction_loss_only or not self.predict_with_generate:
                    return loss.detach(), None, None
                
                # predict_with_generate=True: Sinh câu thực tế qua Beam Search
                # forced_bos_token_id bắt buộc để mBART-50 sinh đúng ngôn ngữ đích
                generated_tokens = model.generate(
                    pixel_values=inputs.get("pixel_values"),
                    node_input_ids=inputs.get("node_input_ids"),
                    node_bboxes=inputs.get("node_bboxes", None),
                    edge_index=inputs.get("edge_index"),
                    batch=inputs.get("batch"),
                    # max_new_tokens=30: captions are 5-15 words — 128 caused hallucination
                    # after correct keywords (model kept running until hitting the cap)
                    max_new_tokens=30,
                    num_beams=getattr(args, 'num_beams_eval', 1),
                    do_sample=False,                           # greedy/beam — deterministic for eval
                    early_stopping=True,
                    forced_bos_token_id=forced_bos_token_id,
                    eos_token_id=mbart_tok.eos_token_id,      # explicitly tell generate() to stop at </s>
                    repetition_penalty=getattr(args, 'repetition_penalty', 1.2),
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                )
            
            return loss.detach(), generated_tokens, labels
    
    # Initialize Custom Trainer
    # DISK FIX: Add aggressive cache cleanup callback after each epoch
    class CacheCleanupCallback(TrainerCallback):
        """Clean up HuggingFace cache and GPU memory after each epoch to prevent disk quota overflow."""
        def on_epoch_end(self, args, state, control, **kwargs):
            try:
                import gc
                import shutil
                gc.collect()
                torch.cuda.empty_cache()
                cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir, ignore_errors=True)
            except Exception:
                pass  # Silent fail
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,   # <-- BLEU-4 metrics
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            CacheCleanupCallback()  # DISK FIX: cleanup after every epoch
        ]
    )
    
    print("Starting Training...")
    resume = getattr(args, 'resume', None)
    trainer.train(resume_from_checkpoint=resume)
    
    # Push final model and configs to hub
    if getattr(args, 'push_to_hub', False) and getattr(args, 'hf_repo_id', None):
        trainer.push_to_hub("Training complete!")

    return trainer

def main():
    """CLI entrypoint: parse args and call run_training."""
    args = parse_args()
    run_training(args)

if __name__ == "__main__":
    main()
