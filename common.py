import os
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import soundfile as sf
import soxr
import torch
from torch.utils.data import Dataset
import jiwer
from transformers import WhisperTokenizer

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST = DATA_DIR / "manifest.csv"
SPEAKERS = ["HQTV", "PNV", "THV", "TLV"]
BASE_MODEL = "openai/whisper-medium.en"
SAMPLE_RATE = 16000
SEED = 42

ACCENT_SPEAKERS = {
    "vietnamese": ["HQTV", "PNV", "THV", "TLV"],
    "arabic": ["ABA", "SKA", "YBAA", "ZHAA"],
    "chinese": ["BWC", "LXC", "NCC", "TXHC"],
    "hindi": ["ASI", "RRBI", "SVBI", "TNI"],
    "korean": ["HJK", "HKK", "YDCK", "YKWK"],
    "spanish": ["EBVS", "ERMS", "MBMPS", "NJS"],
}

_hub_tokenizer = None


def get_hub_normalizer():
    global _hub_tokenizer
    if _hub_tokenizer is None:
        _hub_tokenizer = WhisperTokenizer.from_pretrained(BASE_MODEL)
    return _hub_tokenizer.normalize


def normalize_text(text: str, tokenizer: Optional[Any] = None) -> str:
    if tokenizer is not None and hasattr(tokenizer, "normalize"):
        try:
            res = tokenizer.normalize(text)
            if res is not None:
                return res
        except Exception:
            pass
    norm_fn = get_hub_normalizer()
    return norm_fn(text)


def corpus_wer(refs: List[str], hyps: List[str], tokenizer: Optional[Any] = None) -> float:
    norm_refs = [normalize_text(r, tokenizer) for r in refs]
    norm_hyps = []
    for h in hyps:
        nh = normalize_text(h, tokenizer)
        if not nh.strip():
            nh = "empty"
        norm_hyps.append(nh)
    norm_refs = [nr if nr.strip() else "empty" for nr in norm_refs]
    return float(jiwer.wer(norm_refs, norm_hyps))


def corpus_cer(refs: List[str], hyps: List[str], tokenizer: Optional[Any] = None) -> float:
    norm_refs = [normalize_text(r, tokenizer) for r in refs]
    norm_hyps = []
    for h in hyps:
        nh = normalize_text(h, tokenizer)
        if not nh.strip():
            nh = "empty"
        norm_hyps.append(nh)
    norm_refs = [nr if nr.strip() else "empty" for nr in norm_refs]
    return float(jiwer.cer(norm_refs, norm_hyps))


def get_manifest_path(accent: Optional[str] = None) -> pathlib.Path:
    if accent and accent.lower() != "vietnamese":
        accent_manifest = DATA_DIR / accent.lower() / "manifest.csv"
        return accent_manifest
    if (DATA_DIR / "vietnamese" / "manifest.csv").exists():
        return DATA_DIR / "vietnamese" / "manifest.csv"
    return MANIFEST


def load_manifest(
    split: Optional[str] = None,
    accent: Optional[str] = None,
    manifest_path: Optional[Union[str, pathlib.Path]] = None,
) -> pd.DataFrame:
    if manifest_path is not None:
        target_path = pathlib.Path(manifest_path)
    elif accent is not None:
        target_path = get_manifest_path(accent)
    else:
        target_path = MANIFEST

    if not target_path.exists():
        raise FileNotFoundError(f"Manifest not found at {target_path}")
    df = pd.read_csv(target_path)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


def load_audio_16k(path: Union[str, pathlib.Path]) -> np.ndarray:
    path_str = str(path)
    if not os.path.isabs(path_str):
        path_str = str(PROJECT_ROOT / path_str)
    audio, sr = sf.read(path_str, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLE_RATE:
        audio = soxr.resample(audio, sr, SAMPLE_RATE)
    return audio.astype(np.float32)


class WhisperSpeechDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_extractor: Any, tokenizer: Any):
        self.df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        audio = load_audio_16k(row["path"])
        features = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE).input_features[0]
        labels = self.tokenizer(str(row["text"])).input_ids
        return {"input_features": features, "labels": labels}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor, np.ndarray]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
