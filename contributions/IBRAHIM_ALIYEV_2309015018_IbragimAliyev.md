# Ibrahim Aliyev — 2309015018 — @IbragimAliyev

Whisper-small zero-shot run, wav2vec 2.0 baselines, fine-tuning script and runs.

## Scripts

- `src/transcribe.py`: runs a pretrained or fine-tuned model on `dev` or `test` and writes the raw
  output to `results/predictions/<model>_<split>.csv`, with the settings, hardware, package versions
  and timing in the matching `.json`. Whisper models use greedy decoding, the language forced to
  Basque and `max_new_tokens=128`. CTC models (wav2vec 2.0, mHuBERT, ...) are loaded with
  `AutoModelForCTC`. It refuses quick checks on `test` and never overwrites a prediction file
  without `--force`.
- `src/finetune_whisper.py`: fine-tunes Whisper-small on one condition (`raw`, `ours`, `authors`,
  `rawmatched`) with one seed. The settings are the same for every run: 2,500 steps of 16 clips
  (8 × 2 gradient accumulation), AdamW with lr 1e-5 and weight decay 0.01, 250 warm-up steps then
  linear decay, gradient clipping at 1.0, fp16 autocast and gradient checkpointing. Every 500 steps
  it transcribes all of `dev` and logs the WER (after `clean.normalise`) and the evaluation time;
  the checkpoint with the lowest dev WER is kept. It never reads `test`. The log is
  `results/runs/ft-<condition>_seed<N>.json`.

## Runs

| Run | Split | Status |
|---|---|---|
| `whisper-small-zeroshot` | test, dev | done (18 Nov 2026) |
| `wav2vec2-cv-eu` (`stefan-it/wav2vec2-large-xlsr-53-basque`) | test, dev | done (18 Nov 2026) |
| `wav2vec2-bp1` (`gttsehu/wav2vec2-xls-r-300m-bp1-es_eu`) | dev | done (18 Nov 2026); test at the final scoring |
| `ft-raw` seed 1 (pilot) | dev during training | done (18 Nov 2026, 90 min); best dev WER 0.2087 at step 2500; waiting for the group's decision on the curve |
| `ft-raw` seeds 2, 3 | dev during training | to do |
| `ft-rawmatched`, `ft-authors`, `ft-ours` seeds 1, 2 | dev during training | to do |

Hardware: NVIDIA GeForce RTX 4070 Laptop GPU (8 GB), `torch 2.9.1+cu126` (the driver does not
support the CUDA 13 build; the build is recorded in every JSON).

This file is updated as my work lands in the repo.
