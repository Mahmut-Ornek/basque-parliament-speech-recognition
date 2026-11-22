# Contributions — Tuğra Geler (2309011089, @Tugrii)

## Role Summary
Owned the NVIDIA NeMo ASR baselines, the baseline training-data & leakage audit across all 8 pretrained models, and the deterministic error-type, morphology, and Weighted WER analysis pipeline.

---

## Owned Files & Deliverables

| Area | Files | Description |
|---|---|---|
| **NeMo Baselines** | `src/run_nemo.py`, `requirements-nemo.txt` | Inference & RTFx timing script for NeMo models and pinned Colab T4 dependency environment |
| **Model Predictions** | `results/predictions/hitz-ctc_*`, `hitz-transducer_*`, `parakeet-v3-basque_*` | Full `dev` (1,397 clips) and `test` (1,521 clips) raw CSV outputs + JSON metadata with HF revisions |
| **Training-Data Audit** | `report/baseline_audit.md` | Dataset provenance, split exposure, and speaker-overlap audit for all 8 baselines |
| **Error & Morphology Analysis** | `src/error_types.py`, `results/error_types.csv`, `report/error_analysis.md` | Rule-based `jiwer` error breakdown, Weighted WER, Whisper loop detection, and written analysis |

---

## Detailed Contributions

### 1. NVIDIA NeMo Baselines (`src/run_nemo.py`, `requirements-nemo.txt`)
- Implemented the NeMo ASR evaluation and timing pipeline (`src/run_nemo.py`) outputting unnormalised CSV predictions and structured JSON metadata (Hugging Face commit revisions, GPU specs, warmup-excluded RTFx/latency).
- Generated initial Colab T4 dependency pins in `requirements-nemo.txt` (later extended by Mahmut with `nemo-toolkit[asr]` and `nv-one-logger` pins for fresh Colab installs).
- Ran full `dev` and `test` inference on Colab T4 for all three NeMo baselines:
  - **`hitz-ctc`** (`HiTZ/stt_eu_conformer_ctc_large`): `0.0525` dev WER / `0.0475` test WER
  - **`hitz-transducer`** (`HiTZ/stt_eu_conformer_transducer_large`): `0.0462` dev WER / `0.0428` test WER
  - **`parakeet-v3-basque`** (`itzune/parakeet-tdt-0.6b-v3-basque`): `0.0443` dev WER / `0.0468` test WER

### 2. Baseline Training-Data Audit (`report/baseline_audit.md`)
- Audited the model cards and training corpora of all 8 evaluated baselines (`whisper-large-v3`, `whisper-small-zeroshot`, `hitz-ctc`, `hitz-transducer`, `parakeet-v3-basque`, `wav2vec2-bp1`, `wav2vec2-cv-eu`, `mhubert-basque`).
- Verified that `parakeet-v3-basque` explicitly held out `test_parl` (no test leakage) and confirmed `mhubert-basque` provenance on `HiTZ/composite_corpus_eu_v2.1`.
- Documented cross-split speaker overlap and added the caveat that composite-corpus models may have slightly optimistic `dev` scores due to a small validation overlap with official `dev` (while `test` remains unaffected).

### 3. Error-Type, Morphology & Weighted WER Analysis (`src/error_types.py`, `results/error_types.csv`, `report/error_analysis.md`)
- Built `src/error_types.py` using `src/normalize.py` and `jiwer` word alignments to classify every error deterministically into five mutually exclusive categories: **number (digit) errors**, **suffix-only morphological errors**, **other substitutions**, **deletions**, and **insertions**.
- Calibrated the suffix-only prefix threshold (`>= 4` chars and `>= 60%` shared prefix) and **Weighted WER** (`suffix_weight = 0.5`) on `dev` (testing `0.50 / 0.60 / 0.70` sensitivity) before inspecting `test`.
- Implemented repetition-loop clip detection to quantify Whisper decoding collapse near `max_new_tokens=128`.
- Generated `results/error_types.csv` (all 20 runs × 2 splits = 40 rows) and authored `report/error_analysis.md`, highlighting:
  - **Morphology:** Suffix-only errors account for **16.1%–19.1%** of all test errors in strong in-domain baselines and **14.6%–15.9%** in fine-tuned `whisper-small`.
  - **Fine-tuning gains:** Cuts `whisper-small` test repetition loops from **97 clips (6.4%)** down to **0–2 clips (0.0%–0.1%)** and drops digit-based number errors from 28 (`111` in `whisper-large-v3`) down to **1–2 on `test` (`0` on `dev`)**.