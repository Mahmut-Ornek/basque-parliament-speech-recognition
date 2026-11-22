# Ibrahim Aliyev — 2309015018 — @IbragimAliyev

**Role:** Whisper-small zero-shot run, wav2vec 2.0 baselines, fine-tuning script and runs.

## Code

- `src/transcribe.py`: runs a pretrained model or a fine-tuned checkpoint on `dev` or `test` and
  writes the raw output to `results/predictions/<model>_<split>.csv`, with the model revision,
  decoding settings, precision, hardware, package versions and timing in the matching `.json`.
  - Whisper: greedy decoding, language forced to Basque, `max_new_tokens=128` for every run, and a
    count of clips that hit the token limit.
  - CTC models are loaded with `AutoModelForCTC`, so the same script runs wav2vec 2.0 and mHuBERT.
  - `--split` has no default, quick checks (`--limit`) are refused on `test`, and an existing
    prediction file is never overwritten without `--force`, so `test` is run once per model.
- `src/finetune_whisper.py`: fine-tunes Whisper-small on one condition (`raw`, `rawmatched`,
  `authors`, `ours`) with one seed. The training set and the run name follow from the condition,
  so a run cannot be logged under the wrong name.
  - Frozen settings for all twelve runs: 2,500 steps of 16 clips (8 × 2 gradient accumulation),
    AdamW (lr 1e-5, weight decay 0.01), 250 warm-up steps then linear decay to 0, gradient
    clipping at 1.0, fp16 autocast with fp32 weights, gradient checkpointing.
  - Every 500 steps it transcribes all 1,397 `dev` clips and logs the WER (after
    `clean.normalise`), the evaluation time and the clips at the token limit; the checkpoint
    with the lowest dev WER is kept. It never reads `test`.
  - The log, `results/runs/ft-<condition>_seed<N>.json`, is rewritten after every evaluation, so
    a stopped run keeps its curve.

## Baselines

- `whisper-small-zeroshot` (`openai/whisper-small`), on `dev` and `test`. Whisper-small loops on
  Basque: 128 of the 1,521 test clips hit the token limit, and the official score includes them.
- `wav2vec2-cv-eu` (`stefan-it/wav2vec2-large-xlsr-53-basque`, trained on Common Voice), on `dev`
  and `test`.
- `wav2vec2-bp1` (`gttsehu/wav2vec2-xls-r-300m-bp1-es_eu`, the corpus authors' model), on `dev`
  and `test`. Flagged to the group: it was trained on this corpus's `train` split, which shares
  speakers with `dev` and `test`, so its score is not comparable with the others.

## Fine-tuning runs

- Ran the pilot, `ft-raw` seed 1 (18 Nov), and posted its dev-WER curve for the group's decision
  on the step count (kept at 2,500).
- Trained eight of the twelve runs on my GPU: `ft-raw`, `ft-rawmatched`, `ft-authors` and
  `ft-ours`, seeds 1 and 2 (18–21 Nov, 76–90 min each). Mahmut trained seed 3 with the same
  script and settings.
- Transcribed `dev` and `test` with the best checkpoint of each of my eight runs, with
  `transcribe.py`, the same way as the seed-3 runs.
- Best dev WER per run, as used to pick the checkpoint (the official scores are in the results
  table, from `src/score_all.py`):

| Condition | Seed 1 | Seed 2 |
|---|---|---|
| `ft-raw` | 0.2087 (step 2500) | 0.2067 (step 2000) |
| `ft-rawmatched` | 0.2156 (step 2500) | 0.2170 (step 2500) |
| `ft-authors` | 0.2083 (step 2000) | 0.2092 (step 2000) |
| `ft-ours` | 0.2137 (step 2000) | 0.2155 (step 1500) |

- Every curve flattens after about step 1,500. About 95–100 of the 1,397 `dev` clips still hit the
  token limit at every evaluation, so the fine-tuned models also loop on about 7% of clips.

## Notes for reproduction

- Hardware: NVIDIA GeForce RTX 4070 Laptop GPU (8 GB), peak training memory 5.0 GB.
- `torch 2.9.1+cu126`: the laptop's driver does not support the CUDA 13 build. `requirements.txt`
  pins `torch==2.9.1` without a suffix, and the build is recorded in every JSON.
- The dev WER of a prediction file can differ from the training log in the fourth decimal (for
  example 0.2084 against 0.2087 for `ft-raw` seed 1): the training evaluation uses fp32 weights,
  `transcribe.py` loads the checkpoint in fp16. All twelve fine-tuned models were transcribed the
  same way.
