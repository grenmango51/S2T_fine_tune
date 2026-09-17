# Future benchmarks (not built yet)

- **Native-speaker control (CMU ARCTIC):** evaluate zero-shot and fine-tuned models on speakers `bdl` + `slt` from the original CMU ARCTIC corpus (festvox, ~200 MB) restricted to our exact test sentence IDs, to measure the accent gap on identical text and confirm no regression on native speech.
- **LibriSpeech test-clean forgetting check:** run the fine-tuned model on LibriSpeech test-clean (~350 MB) and compare against zero-shot `medium.en` (~3% WER public reference) to quantify catastrophic forgetting, mirroring the methodology of Bagat et al. 2025 (MAS-LoRA, arXiv 2505.20006).
- **Published reference points:** quote McGuire 2025 (arXiv 2503.06924; Whisper large-v3 on L2-ARCTIC read speech, mean MER 0.054, Vietnamese/Chinese the worst L1s) and the Bagat et al. 2025 LoRA-vs-full-FT L2-ARCTIC WER tables in the eval report as external context, with the caveat that their sentence subsets and metrics differ from ours.
