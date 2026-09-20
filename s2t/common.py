import os
from s2t.datasets.cmu_arctic import CMU_ACCENT_SPEAKERS
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

from s2t.paths import PROJECT_ROOT
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
    "personal": ["PERSONAL"],
    "librispeech": [],
    **CMU_ACCENT_SPEAKERS,
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


def _normalized_pairs(refs, hyps, tokenizer=None):
    def normalized(text):
        value = normalize_text(text, tokenizer)
        return value if value.strip() else "empty"
    return [normalized(r) for r in refs], [normalized(h) for h in hyps]


def corpus_wer(refs: List[str], hyps: List[str], tokenizer: Optional[Any] = None) -> float:
    return float(jiwer.wer(*_normalized_pairs(refs, hyps, tokenizer)))


def corpus_cer(refs: List[str], hyps: List[str], tokenizer: Optional[Any] = None) -> float:
    return float(jiwer.cer(*_normalized_pairs(refs, hyps, tokenizer)))


def _splitmerge_edits(norm_ref: str, norm_hyp: str) -> tuple:
    """Return (standard_errors, splitmerge_tolerant_errors, n_ref_words) for one pair
    of ALREADY-normalized strings."""
    out = jiwer.process_words([norm_ref], [norm_hyp])
    ref_words = norm_ref.split()
    hyp_words = norm_hyp.split()
    standard = out.substitutions + out.deletions + out.insertions
    forgiven = 0
    # jiwer >= 3: out.alignments is a list (per sentence) of AlignmentChunk
    chunks = out.alignments[0]
    i = 0
    while i < len(chunks):
        if chunks[i].type == "equal":
            i += 1
            continue
        # group consecutive non-equal chunks into one error run
        j = i
        while j < len(chunks) and chunks[j].type != "equal":
            j += 1
        r0, r1 = chunks[i].ref_start_idx, chunks[j - 1].ref_end_idx
        h0, h1 = chunks[i].hyp_start_idx, chunks[j - 1].hyp_end_idx
        ref_join = "".join(ref_words[r0:r1])
        hyp_join = "".join(hyp_words[h0:h1])
        if ref_join and ref_join == hyp_join:
            # count the edits inside this run and forgive them
            for c in chunks[i:j]:
                if c.type == "substitute":
                    forgiven += c.ref_end_idx - c.ref_start_idx
                elif c.type == "delete":
                    forgiven += c.ref_end_idx - c.ref_start_idx
                elif c.type == "insert":
                    forgiven += c.hyp_end_idx - c.hyp_start_idx
        i = j
    return standard, standard - forgiven, len(ref_words)


def corpus_wer_splitmerge(refs: List[str], hyps: List[str], tokenizer: Optional[Any] = None) -> float:
    """WER where split/merge-only differences (good bye vs goodbye) are not errors.
    Takes RAW strings, like corpus_wer."""
    tot_err = 0
    tot_ref = 0
    for r, h in zip(refs, hyps):
        nr = normalize_text(r, tokenizer)
        nh = normalize_text(h, tokenizer)
        nr = nr if nr.strip() else "empty"
        nh = nh if nh.strip() else "empty"
        _, tolerant, n_ref = _splitmerge_edits(nr, nh)
        tot_err += tolerant
        tot_ref += n_ref
    return float(tot_err) / max(tot_ref, 1)


def normalize_speaker(speaker: str, accent: Optional[str] = None) -> str:
    """Return a canonical speaker ID and optionally validate its accent."""
    speaker_id = speaker.strip().upper()
    known_speakers = {spk for speakers in ACCENT_SPEAKERS.values() for spk in speakers}
    if speaker_id not in known_speakers:
        raise ValueError(
            f"Unknown speaker '{speaker}'. Valid speaker IDs: {', '.join(sorted(known_speakers))}"
        )
    if accent is not None:
        accent_key = accent.strip().lower()
        if accent_key not in ACCENT_SPEAKERS:
            raise ValueError(
                f"Unknown accent '{accent}'. Valid accents: {', '.join(ACCENT_SPEAKERS)}"
            )
        if speaker_id not in ACCENT_SPEAKERS[accent_key]:
            raise ValueError(f"Speaker {speaker_id} is not a {accent_key} speaker")
    return speaker_id


def get_speaker_data_dir(speaker: str, accent: Optional[str] = None) -> pathlib.Path:
    speaker_id = normalize_speaker(speaker, accent=accent)
    return DATA_DIR / "speakers" / speaker_id


def get_manifest_path(
    accent: Optional[str] = None,
    speaker: Optional[str] = None,
) -> pathlib.Path:
    if speaker is not None:
        return get_speaker_data_dir(speaker, accent=accent) / "manifest.csv"
    if accent and accent.lower() != "vietnamese":
        accent_manifest = DATA_DIR / accent.lower() / "manifest.csv"
        return accent_manifest
    if (DATA_DIR / "vietnamese" / "manifest.csv").exists():
        return DATA_DIR / "vietnamese" / "manifest.csv"
    return MANIFEST


def load_manifest(
    split: Optional[str] = None,
    accent: Optional[str] = None,
    speaker: Optional[str] = None,
    manifest_path: Optional[Union[str, pathlib.Path]] = None,
) -> pd.DataFrame:
    if manifest_path is not None:
        target_path = pathlib.Path(manifest_path)
    elif accent is not None or speaker is not None:
        target_path = get_manifest_path(accent=accent, speaker=speaker)
    else:
        target_path = MANIFEST

    if not target_path.exists():
        hint = ""
        if speaker is not None:
            speaker_id = normalize_speaker(speaker, accent=accent)
            accent_arg = accent or "vietnamese"
            hint = (
                f" Run 'python -m s2t.datasets.prepare_data --accent {accent_arg} "
                f"--speaker {speaker_id}' first."
            )
        raise FileNotFoundError(f"Manifest not found at {target_path}.{hint}")
    df = pd.read_csv(target_path)
    if speaker is not None:
        speaker_id = normalize_speaker(speaker, accent=accent)
        if "speaker" not in df.columns:
            raise ValueError(f"Manifest {target_path} has no 'speaker' column")
        df = df[df["speaker"].astype(str).str.upper() == speaker_id]
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    if df.empty:
        details = []
        if split is not None:
            details.append(f"split={split}")
        if speaker is not None:
            details.append(f"speaker={speaker_id}")
        suffix = f" for {', '.join(details)}" if details else ""
        raise ValueError(f"Manifest {target_path} contains no rows{suffix}")
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
        extracted = self.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_attention_mask=True,
        )
        features = extracted.input_features[0]
        labels = self.tokenizer(str(row["text"])).input_ids
        return {
            "input_features": features,
            "attention_mask": extracted.attention_mask[0],
            "labels": labels,
        }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor, np.ndarray]]]) -> Dict[str, torch.Tensor]:
        input_features = [
            {
                "input_features": feature["input_features"],
                "attention_mask": feature.get("attention_mask"),
            }
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
