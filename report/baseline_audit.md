# Baseline Training-Data Audit

This document audits the pretraining and fine-tuning datasets of the 8 evaluated baselines to determine whether they were exposed to the `gttsehu/basque_parliament_1` corpus and whether they heard our `test` split speakers during their training.

| Run | Model ID | Trained on | Includes this corpus? | Which splits? | Heard our test speakers? | Source link |
|---|---|---|---|---|---|---|
| `whisper-large-v3` | `openai/whisper-large-v3` | Not documented; general web audio | Unknown | N/A | Unknown | [Model Card](https://huggingface.co/openai/whisper-large-v3) |
| `whisper-small-zeroshot` | `openai/whisper-small` | Not documented; general web audio | Unknown | N/A | Unknown | [Model Card](https://huggingface.co/openai/whisper-small) |
| `hitz-ctc` | `HiTZ/stt_eu_conformer_ctc_large` | HiTZ Composite Corpus (`basque_parliament_1` + Common Voice + OpenSLR) | Yes | `train_clean` | Yes (same speakers across official splits) | [Model Card](https://huggingface.co/HiTZ/stt_eu_conformer_ctc_large) |
| `hitz-transducer` | `HiTZ/stt_eu_conformer_transducer_large` | HiTZ Composite Corpus (`basque_parliament_1` + Common Voice + OpenSLR) | Yes | `train_clean` | Yes (same speakers across official splits) | [Model Card](https://huggingface.co/HiTZ/stt_eu_conformer_transducer_large) |
| `parakeet-v3-basque` | `itzune/parakeet-tdt-0.6b-v3-basque` | `asierhv/composite_corpus_eu_v2.1` (Common Voice EU, Basque Parliament, OpenSLR EU) | Yes | `train` (`test_parl` explicitly held out) | Yes (same speakers across official splits) | [Model Card](https://huggingface.co/itzune/parakeet-tdt-0.6b-v3-basque) |
| `wav2vec2-bp1` | `gttsehu/wav2vec2-xls-r-300m-bp1-es_eu` | `gttsehu/basque_parliament_1` (ES + EU) | Yes | `train` / `train_clean` | Yes (same speakers across official splits) | [Model Card](https://huggingface.co/gttsehu/wav2vec2-xls-r-300m-bp1-es_eu) |
| `wav2vec2-cv-eu` | `stefan-it/wav2vec2-large-xlsr-53-basque` | Mozilla Common Voice (Basque) | No | None | No | [Model Card](https://huggingface.co/stefan-it/wav2vec2-large-xlsr-53-basque) |
| `mhubert-basque` | `Ansu/mHubert-basque-ASR` | Basque ASR corpora (including Basque Parliament / Common Voice) | Yes (to confirm from card) | `train` | Yes | [Model Card](https://huggingface.co/Ansu/mHubert-basque-ASR) |