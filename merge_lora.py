import argparse
import os
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration
from peft import PeftModel

from common import BASE_MODEL, PROJECT_ROOT, SAMPLE_RATE, load_audio_16k, normalize_text


def merge_lora_and_save(adapter_path: str, out_dir_path: str, base_model_id: str = BASE_MODEL):
    adapter_dir = Path(adapter_path)
    out_dir = Path(out_dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model '{base_model_id}' (fp16)...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id,
        dtype=torch.float16,
        device_map="cpu",
    )

    print(f"Loading LoRA adapter from '{adapter_dir}'...")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    print("Merging LoRA weights into base model and unloading...")
    merged_model = peft_model.merge_and_unload()
    merged_model.to(dtype=torch.float16)

    print(f"Saving merged standalone model to '{out_dir}'...")
    merged_model.save_pretrained(str(out_dir), safe_serialization=True)
    if hasattr(merged_model, "generation_config") and merged_model.generation_config is not None:
        merged_model.generation_config.save_pretrained(str(out_dir))

    print(f"Saving processor and tokenizer to '{out_dir}'...")
    processor = AutoProcessor.from_pretrained(base_model_id)
    processor.save_pretrained(str(out_dir))

    print("Copying normalizer.json into merged directory...")
    try:
        norm_path = hf_hub_download(repo_id=base_model_id, filename="normalizer.json")
        dest_norm_path = out_dir / "normalizer.json"
        shutil.copy(norm_path, str(dest_norm_path))
        print(f"Successfully copied normalizer.json to {dest_norm_path}")
    except Exception as e:
        print(f"Warning: could not copy normalizer.json: {e}")

    # Smoke test: Reload saved model and transcribe one train wav
    print("\n--- Running Smoke Test on Reloaded Merged Model ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Reloading model from '{out_dir}' on {device}...")
    reloaded_model = WhisperForConditionalGeneration.from_pretrained(
        str(out_dir),
        dtype=torch.float16,
        device_map=device,
    )
    reloaded_model.eval()
    reloaded_processor = AutoProcessor.from_pretrained(str(out_dir))

    candidate_wavs = list((PROJECT_ROOT / "data").glob("**/arctic_a0001.wav"))
    if candidate_wavs:
        smoke_audio_path = candidate_wavs[0]
    else:
        all_wavs = list((PROJECT_ROOT / "data").glob("**/*.wav"))
        assert all_wavs, "No audio wav found in data for smoke test"
        smoke_audio_path = all_wavs[0]

    audio = load_audio_16k(smoke_audio_path)
    inputs = reloaded_processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device, dtype=torch.float16)

    with torch.inference_mode():
        pred_ids = reloaded_model.generate(input_features)
    hyp = reloaded_processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

    print(f"Smoke test audio: {smoke_audio_path.relative_to(PROJECT_ROOT)}")
    print(f"Transcription:    '{hyp}'")
    print(f"Normalized:       '{normalize_text(hyp)}'")
    print("\nSMOKE TEST PASSED! Standalone merged model ready.")


def main():
    parser = argparse.ArgumentParser(description="Merge PEFT LoRA adapter into standalone Whisper model")
    parser.add_argument(
        "--adapter",
        type=str,
        default="runs/medium-en-lora/best_adapter",
        help="Path to best LoRA adapter directory",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="models/whisper-medium-en-vi-accent",
        help="Target directory for merged model",
    )
    parser.add_argument(
        "--base",
        type=str,
        default=BASE_MODEL,
        help="Base model Hub ID",
    )
    args = parser.parse_args()

    merge_lora_and_save(adapter_path=args.adapter, out_dir_path=args.out, base_model_id=args.base)


if __name__ == "__main__":
    main()
