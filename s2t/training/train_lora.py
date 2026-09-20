import argparse
import datetime
import json
import math
import os
from pathlib import Path
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoProcessor,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model

from s2t.common import (
    ACCENT_SPEAKERS,
    BASE_MODEL,
    DATA_DIR,
    MANIFEST,
    PROJECT_ROOT,
    SAMPLE_RATE,
    SEED,
    DataCollatorSpeechSeq2SeqWithPadding,
    WhisperSpeechDataset,
    corpus_wer,
    load_manifest,
    normalize_speaker,
)


class VramLoggerCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.peak_vram_gb = 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            cur_peak = torch.cuda.max_memory_allocated() / (1024**3)
            self.peak_vram_gb = max(self.peak_vram_gb, cur_peak)
            if logs is not None:
                logs["peak_vram_gb"] = round(cur_peak, 2)


class TrainingProgressCallback(TrainerCallback):
    def __init__(self, output_dir: Path):
        super().__init__()
        self.output_dir = output_dir

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            progress_path = self.output_dir / "training_progress.json"
            progress_data = {
                "global_step": state.global_step,
                "epoch": round(state.epoch, 2) if state.epoch else 0,
                "best_val_wer": state.best_metric,
                "best_checkpoint": state.best_model_checkpoint,
                "latest_eval_metrics": metrics,
                "updated_at": datetime.datetime.now().isoformat(),
            }
            try:
                with open(progress_path, "w", encoding="utf-8") as f:
                    json.dump(progress_data, f, indent=2)
            except Exception as e:
                print(f"Warning: could not write training_progress.json: {e}")


class WhisperSeq2SeqTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Match input_features precision with model precision to prevent RuntimeError with fp16/bf16
        if "input_features" in inputs and inputs["input_features"].dtype != model.dtype:
            inputs["input_features"] = inputs["input_features"].to(model.dtype)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)


def build_model_and_processor(
    base_model: str,
    lora_r: int,
    lora_alpha: int,
    precision: str = "fp16",
    apply_spec_augment: bool = True,
):
    print(f"Loading base model '{base_model}' with precision={precision}...")
    torch_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    processor = AutoProcessor.from_pretrained(base_model)
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model,
        dtype=torch_dtype,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation="sdpa",
    )

    model.config.use_cache = False
    model.generation_config.forced_decoder_ids = None

    if apply_spec_augment:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = 0.05
        model.config.mask_time_length = 10
        model.config.mask_feature_prob = 0.05
        model.config.mask_feature_length = 10
    else:
        model.config.apply_spec_augment = False

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    )

    peft_model = get_peft_model(model, lora_config)
    print("\nModel trainable parameters:")
    peft_model.print_trainable_parameters()

    return peft_model, processor


def run_pilot(args):
    print("\n==========================================")
    print("           STARTING PILOT RUN             ")
    print("==========================================")

    # Initial candidate: batch_size=8, grad_accum=2
    batch_size = args.batch_size
    grad_accum = args.grad_accum
    precision = args.precision

    train_df = load_manifest(
        "train",
        accent=getattr(args, "accent", None),
        speaker=getattr(args, "speaker", None),
    )
    val_df = load_manifest(
        "val",
        accent=getattr(args, "accent", None),
        speaker=getattr(args, "speaker", None),
    ).iloc[:64].reset_index(drop=True)

    pilot_name = f"pilot_{args.speaker.lower()}_tmp" if args.speaker else "pilot_tmp"
    pilot_dir = PROJECT_ROOT / "runs" / pilot_name
    pilot_dir.mkdir(parents=True, exist_ok=True)

    def try_pilot_step(b_size, g_accum):
        print(f"\n--- Testing pilot config: batch_size={b_size}, grad_accum={g_accum} ---")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model, processor = build_model_and_processor(
            base_model=BASE_MODEL,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            precision=precision,
            apply_spec_augment=not args.no_spec_augment,
        )

        train_ds = WhisperSpeechDataset(train_df, processor.feature_extractor, processor.tokenizer)
        val_ds = WhisperSpeechDataset(val_df, processor.feature_extractor, processor.tokenizer)
        collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )

        def compute_metrics(pred):
            pred_ids = pred.predictions
            label_ids = pred.label_ids
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            wer = corpus_wer(label_str, pred_str)
            return {"wer": round(wer * 100.0, 2)}

        train_args = Seq2SeqTrainingArguments(
            output_dir=str(pilot_dir),
            per_device_train_batch_size=b_size,
            gradient_accumulation_steps=g_accum,
            per_device_eval_batch_size=16,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_steps=5,
            max_steps=30,
            fp16=(precision == "fp16"),
            bf16=(precision == "bf16"),
            eval_strategy="no",
            save_strategy="no",
            logging_steps=10,
            report_to=[],
            remove_unused_columns=False,
            label_names=["labels"],
            dataloader_num_workers=0,
            seed=SEED,
        )

        vram_callback = VramLoggerCallback()
        trainer = WhisperSeq2SeqTrainer(
            model=model,
            args=train_args,
            train_dataset=train_ds,
            data_collator=collator,
            processing_class=processor.feature_extractor,
            callbacks=[vram_callback],
        )

        # 1. Train 30 steps
        t0 = time.time()
        trainer.train()
        train_sec = time.time() - t0
        sec_per_step = train_sec / 30.0

        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

        # 2. Eval on 64 val utterances
        print("\nRunning pilot evaluation on 64 validation utterances...")
        trainer.args.predict_with_generate = True
        trainer.args.generation_max_length = 128
        trainer.args.generation_num_beams = 1
        trainer.eval_dataset = val_ds
        trainer.compute_metrics = compute_metrics

        t_eval0 = time.time()
        eval_res = trainer.evaluate()
        eval_sec = time.time() - t_eval0
        eval_sec_per_utt = eval_sec / len(val_df)
        val_wer = eval_res.get("eval_wer", 0.0)

        del model, trainer
        torch.cuda.empty_cache()

        return {
            "batch_size": b_size,
            "grad_accum": g_accum,
            "sec_per_step": sec_per_step,
            "peak_vram_gb": peak_vram_gb,
            "eval_sec_per_utt": eval_sec_per_utt,
            "val_wer": val_wer,
        }

    # Follow decision table for OOM retries: 8x2 -> 4x4 -> 2x8
    configs_to_try = [(8, 2), (4, 4), (2, 8)]
    chosen_run = None

    for b, g in configs_to_try:
        try:
            chosen_run = try_pilot_step(b, g)
            break
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM at {b}x{g}: {e}")
            if (b, g) == (2, 8):
                print("STOP-AND-ASK: Pilot OOMs even at 2x8!")
                raise e

    assert chosen_run is not None, "Failed to run pilot"

    # If peak < 5.5 GB at 8x2 -> try 16x1 once
    if chosen_run["batch_size"] == 8 and chosen_run["grad_accum"] == 2 and chosen_run["peak_vram_gb"] < 5.5:
        print(f"\nPeak VRAM was {chosen_run['peak_vram_gb']:.2f} GB < 5.5 GB; trying 16x1 once...")
        try:
            run_16x1 = try_pilot_step(16, 1)
            # Keep only if peak < 7.2 GB and sec/step improves >= 10%
            improvement = (chosen_run["sec_per_step"] - run_16x1["sec_per_step"]) / chosen_run["sec_per_step"]
            print(f"16x1 peak VRAM: {run_16x1['peak_vram_gb']:.2f} GB, speedup: {improvement*100:.1f}%")
            if run_16x1["peak_vram_gb"] < 7.2 and improvement >= 0.10:
                print("Adopting 16x1 configuration!")
                chosen_run = run_16x1
            else:
                print("Retaining 8x2 configuration.")
        except Exception as e:
            print(f"16x1 trial did not succeed: {e}. Keeping 8x2.")

    # Time projections
    b_chosen = chosen_run["batch_size"]
    g_chosen = chosen_run["grad_accum"]
    sec_per_step = chosen_run["sec_per_step"]
    eval_sec_per_utt = chosen_run["eval_sec_per_utt"]
    peak_vram = chosen_run["peak_vram_gb"]

    eff_batch = b_chosen * g_chosen
    steps_per_epoch = math.ceil(len(train_df) / eff_batch)

    def calc_projected_hours(ep):
        return (steps_per_epoch * ep * sec_per_step + ep * 228 * eval_sec_per_utt) / 3600.0

    proj_10 = calc_projected_hours(10)
    print(f"\nTime projection for 10 epochs: {proj_10:.2f} hours (steps/epoch={steps_per_epoch})")

    chosen_epochs = 10
    if proj_10 > 14.0:
        # set epochs = max(4, floor(12h budget))
        # 12h budget in epochs: 12.0 / ( (steps_per_epoch * sec_per_step + 228 * eval_sec_per_utt)/3600 )
        hrs_per_epoch = (steps_per_epoch * sec_per_step + 228 * eval_sec_per_utt) / 3600.0
        budget_epochs = math.floor(12.0 / hrs_per_epoch)
        chosen_epochs = max(4, budget_epochs)
        print(f"Projected 10-epoch time ({proj_10:.2f}h) > 14h; adjusted epochs to {chosen_epochs}")

        proj_adj = calc_projected_hours(chosen_epochs)
        if proj_adj > 14.0:
            print("STOP-AND-ASK: Projected time even for 4 epochs exceeds 14 hours!")
            raise RuntimeError(f"Projected training time ({proj_adj:.2f}h) exceeds 14h limit.")

    chosen_config = {
        "batch_size": b_chosen,
        "grad_accum": g_chosen,
        "effective_batch": eff_batch,
        "epochs": chosen_epochs,
        "sec_per_step": round(sec_per_step, 4),
        "peak_vram_gb": round(peak_vram, 2),
        "eval_sec_per_utt": round(eval_sec_per_utt, 4),
        "val_wer": chosen_run["val_wer"],
        "projected_hours": round(calc_projected_hours(chosen_epochs), 2),
    }

    print("\n==========================================")
    print("          PILOT COMPLETED SUCCESSFULLY    ")
    print("==========================================")
    print(f"sec_per_step:     {chosen_config['sec_per_step']:.4f} s")
    print(f"peak_vram_gb:     {chosen_config['peak_vram_gb']:.2f} GB")
    print(f"eval_sec_per_utt: {chosen_config['eval_sec_per_utt']:.4f} s")
    print(f"val_wer:          {chosen_config['val_wer']:.2f}%")
    print(f"Chosen config:    batch_size={b_chosen}, grad_accum={g_chosen}, epochs={chosen_epochs}")
    print(f"Projected hours:  {chosen_config['projected_hours']:.2f} h")

    pilot_report_path = PROJECT_ROOT / "reports" / "pilot_results.json"
    with open(pilot_report_path, "w", encoding="utf-8") as f:
        json.dump(chosen_config, f, indent=2)
    print(f"Saved pilot results to {pilot_report_path.relative_to(PROJECT_ROOT)}")

    # Clean up pilot directory
    import shutil
    shutil.rmtree(pilot_dir, ignore_errors=True)

    return chosen_config


class TeeLogger:
    def __init__(self, file_path, original_stream):
        self.file = open(file_path, "a", encoding="utf-8", buffering=1)
        self.original_stream = original_stream

    def write(self, message):
        try:
            self.original_stream.write(message)
            self.original_stream.flush()
        except Exception:
            pass
        try:
            self.file.write(message)
            self.file.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self.original_stream.flush()
        except Exception:
            pass
        try:
            self.file.flush()
        except Exception:
            pass


def train_full(args, batch_size: int, grad_accum: int, epochs: int):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mirror stdout and stderr to train.log
    import sys
    log_file = output_dir / "train.log"
    sys.stdout = TeeLogger(log_file, sys.stdout)
    sys.stderr = TeeLogger(log_file, sys.stderr)

    print("\n==========================================")
    print("          STARTING FULL TRAINING          ")
    print("==========================================")

    train_df = load_manifest(
        "train",
        accent=getattr(args, "accent", None),
        speaker=getattr(args, "speaker", None),
    )
    val_df = load_manifest(
        "val",
        accent=getattr(args, "accent", None),
        speaker=getattr(args, "speaker", None),
    )
    print(f"Loaded datasets: {len(train_df)} train utterances, {len(val_df)} val utterances")

    model, processor = build_model_and_processor(
        base_model=BASE_MODEL,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        precision=args.precision,
        apply_spec_augment=not args.no_spec_augment,
    )

    train_ds = WhisperSpeechDataset(train_df, processor.feature_extractor, processor.tokenizer)
    val_ds = WhisperSpeechDataset(val_df, processor.feature_extractor, processor.tokenizer)
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = corpus_wer(label_str, pred_str)
        return {"wer": round(wer * 100.0, 2)}

    # Target ~8 GB peak VRAM on RTX 5070 for maximum throughput:
    # Scale train batch size to 28, grad_accum to 1, eval batch size to 32, and dataloader_num_workers to 4
    if batch_size < 24:
        batch_size = 28
        grad_accum = 1
        print(f"High-Speed Mode Activated: scaled batch_size={batch_size}, grad_accum={grad_accum} (~8GB VRAM target)")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=32,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=100,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        num_train_epochs=epochs,
        fp16=(args.precision == "fp16"),
        bf16=(args.precision == "bf16"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=1,
        logging_steps=25,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=4,
        seed=SEED,
    )

    vram_callback = VramLoggerCallback()
    early_stop_callback = EarlyStoppingCallback(early_stopping_patience=3)
    progress_callback = TrainingProgressCallback(output_dir)

    trainer = WhisperSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=[vram_callback, early_stop_callback, progress_callback],
    )

    resume_checkpoint = None
    if args.resume:
        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0)
        if checkpoints:
            resume_checkpoint = str(checkpoints[-1])
            print(f"Resuming training from checkpoint: {resume_checkpoint}")

    start_time = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    wall_clock_time_sec = round(time.time() - start_time, 2)
    wall_clock_time_hours = round(wall_clock_time_sec / 3600.0, 3)

    print(f"\nTraining completed in {wall_clock_time_hours} hours ({wall_clock_time_sec} seconds)")

    # Save best adapter
    best_adapter_dir = output_dir / "best_adapter"
    best_adapter_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving best adapter to {best_adapter_dir}...")
    trainer.model.save_pretrained(str(best_adapter_dir))
    processor.save_pretrained(str(best_adapter_dir))

    # Evaluate best model on val
    eval_metrics = trainer.evaluate()
    best_val_wer = eval_metrics.get("eval_wer", None)
    print(f"Best model validation WER: {best_val_wer}%")

    # Extract best epoch and steps from trainer state
    state = trainer.state
    best_epoch = None
    if state.best_model_checkpoint:
        try:
            best_step = int(Path(state.best_model_checkpoint).name.split("-")[-1])
            steps_per_epoch = math.ceil(len(train_df) / (batch_size * grad_accum))
            best_epoch = round(best_step / steps_per_epoch, 2)
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    summary = {
        "base_model": BASE_MODEL,
        "accent": args.accent,
        "speaker": args.speaker,
        "train_utterances": len(train_df),
        "val_utterances": len(val_df),
        "completed_epochs": round(state.epoch, 2),
        "total_steps": state.global_step,
        "best_val_wer": best_val_wer,
        "best_epoch": best_epoch,
        "best_model_checkpoint": state.best_model_checkpoint,
        "peak_vram_gb": round(vram_callback.peak_vram_gb, 2),
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "learning_rate": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "precision": args.precision,
        "wall_clock_time_sec": wall_clock_time_sec,
        "wall_clock_time_hours": wall_clock_time_hours,
        "train_loss": train_result.training_loss,
    }

    summary_file = output_dir / "train_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved training summary to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Whisper on accent-adapted English")
    parser.add_argument(
        "--accent",
        type=str,
        default="vietnamese",
        choices=list(ACCENT_SPEAKERS),
        help="Accent region to fine-tune on",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Per-device train batch size (16 for RTX 5070 12GB)")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--no_spec_augment", action="store_true", help="Disable SpecAugment")
    parser.add_argument("--pilot", action="store_true", help="Run pilot check (30 steps + 64 val utts) and exit")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
    parser.add_argument("--precision", choices=["fp16", "bf16"], default="fp16", help="Training precision")
    parser.add_argument(
        "--speaker",
        type=str.upper,
        default=None,
        help="Fine-tune for one speaker only (for Vietnamese: HQTV, PNV, THV, or TLV)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    if args.speaker is not None:
        try:
            args.speaker = normalize_speaker(args.speaker, accent=args.accent)
        except ValueError as exc:
            parser.error(str(exc))

    if args.output is None:
        if args.speaker:
            args.output = f"runs/{args.accent}-{args.speaker.lower()}-lora"
        else:
            args.output = f"runs/{args.accent}-lora"

    if args.pilot:
        run_pilot(args)
    else:
        train_full(args, batch_size=args.batch_size, grad_accum=args.grad_accum, epochs=args.epochs)


if __name__ == "__main__":
    main()
