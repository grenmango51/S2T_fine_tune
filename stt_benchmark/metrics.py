from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import jiwer

import s2t.common as common


@dataclass
class UtteranceResult:
    utt_id: str
    speaker: Optional[str]
    ref_raw: str
    hyp_raw: str
    ref_norm: str
    hyp_norm: str
    wer: float
    cer: float
    latency_s: float
    cost_usd: Optional[float]
    model_name: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UtteranceResult":
        return cls(**data)


@dataclass
class ModelResults:
    model_name: str
    model_type: str
    utterance_results: List[UtteranceResult]
    corpus_wer: float
    corpus_cer: float
    mean_latency_s: float
    median_latency_s: float
    total_cost_usd: Optional[float]
    speaker_results: Dict[str, Dict[str, Any]]
    total_utterances: int
    successful_utterances: int
    failed_utterances: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["utterance_results"] = [u.to_dict() for u in self.utterance_results]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelResults":
        utts = [UtteranceResult.from_dict(u) for u in data.get("utterance_results", [])]
        data_copy = dict(data)
        data_copy["utterance_results"] = utts
        return cls(**data_copy)


def compute_utterance_result(
    utt_id: str,
    ref_raw: str,
    hyp_raw: str,
    latency_s: float,
    model_name: str,
    speaker: Optional[str] = None,
    cost_usd: Optional[float] = None,
    error: Optional[str] = None,
) -> UtteranceResult:
    """Normalize reference and hypothesis and compute per-utterance WER/CER."""
    ref_str = str(ref_raw) if ref_raw is not None else ""
    hyp_str = str(hyp_raw) if hyp_raw is not None else ""

    norm_ref = common.normalize_text(ref_str)
    norm_hyp = common.normalize_text(hyp_str)

    # Align with common.py: non-empty or fallback to 'empty'
    norm_ref_eval = norm_ref if norm_ref.strip() else "empty"
    norm_hyp_eval = norm_hyp if norm_hyp.strip() else "empty"

    try:
        sample_wer = float(jiwer.wer([norm_ref_eval], [norm_hyp_eval]))
        sample_cer = float(jiwer.cer([norm_ref_eval], [norm_hyp_eval]))
    except Exception:
        sample_wer = 1.0
        sample_cer = 1.0

    return UtteranceResult(
        utt_id=utt_id,
        speaker=speaker,
        ref_raw=ref_str,
        hyp_raw=hyp_str,
        ref_norm=norm_ref,
        hyp_norm=norm_hyp,
        wer=round(sample_wer * 100, 2),
        cer=round(sample_cer * 100, 2),
        latency_s=latency_s,
        cost_usd=cost_usd,
        model_name=model_name,
        error=error,
    )


def aggregate_model_results(
    model_name: str,
    model_type: str,
    utterance_results: List[UtteranceResult],
) -> ModelResults:
    """Aggregate utterance results into corpus-level metrics and speaker breakdowns."""
    if not utterance_results:
        return ModelResults(
            model_name=model_name,
            model_type=model_type,
            utterance_results=[],
            corpus_wer=0.0,
            corpus_cer=0.0,
            mean_latency_s=0.0,
            median_latency_s=0.0,
            total_cost_usd=0.0,
            speaker_results={},
            total_utterances=0,
            successful_utterances=0,
            failed_utterances=0,
        )

    all_refs = [u.ref_norm for u in utterance_results]
    all_hyps = [u.hyp_norm for u in utterance_results]

    overall_wer = common.corpus_wer(all_refs, all_hyps) * 100
    overall_cer = common.corpus_cer(all_refs, all_hyps) * 100

    latencies = [u.latency_s for u in utterance_results if u.latency_s is not None and u.latency_s > 0]
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    med_lat = float(np.median(latencies)) if latencies else 0.0

    costs = [u.cost_usd for u in utterance_results if u.cost_usd is not None]
    total_cost = sum(costs) if costs else (0.0 if model_type == "local" else None)

    successful = sum(1 for u in utterance_results if not u.error)
    failed = sum(1 for u in utterance_results if u.error)

    # Per-speaker analysis
    speakers = sorted(list({u.speaker for u in utterance_results if u.speaker}))
    speaker_results: Dict[str, Dict[str, Any]] = {}
    for spk in speakers:
        spk_utts = [u for u in utterance_results if u.speaker == spk]
        spk_refs = [u.ref_norm for u in spk_utts]
        spk_hyps = [u.hyp_norm for u in spk_utts]
        spk_wer = common.corpus_wer(spk_refs, spk_hyps) * 100
        spk_cer = common.corpus_cer(spk_refs, spk_hyps) * 100
        speaker_results[spk] = {
            "wer": round(spk_wer, 2),
            "cer": round(spk_cer, 2),
            "count": len(spk_utts),
        }

    return ModelResults(
        model_name=model_name,
        model_type=model_type,
        utterance_results=utterance_results,
        corpus_wer=round(overall_wer, 2),
        corpus_cer=round(overall_cer, 2),
        mean_latency_s=round(mean_lat, 3),
        median_latency_s=round(med_lat, 3),
        total_cost_usd=round(total_cost, 4) if total_cost is not None else None,
        speaker_results=speaker_results,
        total_utterances=len(utterance_results),
        successful_utterances=successful,
        failed_utterances=failed,
    )
