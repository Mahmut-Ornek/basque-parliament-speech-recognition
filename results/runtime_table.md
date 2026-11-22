# Hardware and runtime

<!-- Written by src/runtime_table.py. Do not edit by hand. -->
Generated 2026-11-22 17:16 UTC.

## Inference speed (pretrained models)

Every model processed the same 100 dev clips at batch size 1, after 5 warm-up clips. The clock runs from "audio is in memory" to "text is returned"; model loading is excluded. RTFx = audio seconds / compute seconds (higher is faster). RTF = compute seconds / audio seconds (the inverse; lower is faster). Medians and 95th percentiles are seconds per clip.

| Run | Model | Params (M) | Hardware | Precision | RTFx | RTF | Median s/clip | P95 s/clip |
|---|---|---:|---|---|---:|---:|---:|---:|
| `whisper-small-zeroshot` | openai/whisper-small | 242 | Tesla T4 | float16 | 13.54 | 0.074 | 0.401 | 0.889 |
| `whisper-large-v3` | openai/whisper-large-v3 | 1543 | Tesla T4 | float16 | 5.64 | 0.177 | 1.062 | 1.801 |
| `hitz-ctc` | HiTZ/stt_eu_conformer_ctc_large | 122 | Tesla T4 | float32 | 121.57 | 0.008 | 0.050 | 0.060 |
| `hitz-transducer` | HiTZ/stt_eu_conformer_transducer_large | 119 | Tesla T4 | float32 | 90.28 | 0.011 | 0.067 | 0.091 |
| `parakeet-v3-basque` | itzune/parakeet-tdt-0.6b-v3-basque | 627 | Tesla T4 | float32 | 67.01 | 0.015 | 0.089 | 0.119 |
| `wav2vec2-bp1` | gttsehu/wav2vec2-xls-r-300m-bp1-es_eu | 315 | Tesla T4 | float16 | 118.15 | 0.008 | 0.056 | 0.066 |
| `wav2vec2-cv-eu` | stefan-it/wav2vec2-large-xlsr-53-basque | 315 | Tesla T4 | float16 | 119.39 | 0.008 | 0.056 | 0.066 |
| `mhubert-basque` | Ansu/mHubert-basque-ASR | 94 | Tesla T4 | float16 | 162.70 | 0.006 | 0.042 | 0.047 |

- `whisper-small-zeroshot`: the dev/test accuracy run used **NVIDIA GeForce RTX 4070 Laptop GPU**; the timing above is from Tesla T4.
- `whisper-large-v3`: the dev/test accuracy run used **NVIDIA GeForce RTX 3080 Laptop GPU**; the timing above is from Tesla T4.
- `wav2vec2-bp1`: the dev/test accuracy run used **NVIDIA GeForce RTX 4070 Laptop GPU**; the timing above is from Tesla T4.
- `wav2vec2-cv-eu`: the dev/test accuracy run used **NVIDIA GeForce RTX 4070 Laptop GPU**; the timing above is from Tesla T4.
- `mhubert-basque`: the dev/test accuracy run used **NVIDIA GeForce RTX 3080 Laptop GPU**; the timing above is from Tesla T4.

Speed measured here is for batch size 1 on one GPU model. It says how the models compare under identical conditions, not how fast any of them can go with batching.

## Training time (our fine-tuned runs)

| Run | GPU | Best step | Training time (min) |
|---|---|---:|---:|
| `ft-authors_seed1` | NVIDIA GeForce RTX 4070 Laptop GPU | 2000 | 100.8 |
| `ft-authors_seed2` | NVIDIA GeForce RTX 4070 Laptop GPU | 2000 | 76.5 |
| `ft-authors_seed3` | NVIDIA GeForce RTX 3080 Laptop GPU | 2000 | 97.1 |
| `ft-ours_seed1` | NVIDIA GeForce RTX 4070 Laptop GPU | 2000 | 82.7 |
| `ft-ours_seed2` | NVIDIA GeForce RTX 4070 Laptop GPU | 1500 | 78.6 |
| `ft-ours_seed3` | NVIDIA GeForce RTX 3080 Laptop GPU | 2500 | 97.7 |
| `ft-raw_seed1` | NVIDIA GeForce RTX 4070 Laptop GPU | 2500 | 90.2 |
| `ft-raw_seed2` | NVIDIA GeForce RTX 4070 Laptop GPU | 2000 | 79.2 |
| `ft-raw_seed3` | NVIDIA GeForce RTX 3080 Laptop GPU | 2500 | 97.1 |
| `ft-rawmatched_seed1` | NVIDIA GeForce RTX 4070 Laptop GPU | 2500 | 87.8 |
| `ft-rawmatched_seed2` | NVIDIA GeForce RTX 4070 Laptop GPU | 2500 | 87.2 |
| `ft-rawmatched_seed3` | NVIDIA GeForce RTX 3080 Laptop GPU | 2000 | 97.2 |

The fine-tuned runs were not timed separately: they have the architecture and decoding settings of `whisper-small-zeroshot`, but zero-shot Whisper-small loops on some clips (see `report/error_analysis.md`), which makes its row slower than the fine-tuned models would be.
