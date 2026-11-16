# Basque Parliamentary Speech Recognition — Project 3

Speech recognition for Basque (`eu`) parliamentary speech. The project has two parts:

1. **Baselines:** eight pretrained models scored on the same test clips, with no training by us.
2. **Our contribution:** does cleaning the training transcripts help a fine-tuned Whisper-small,
   and can our cleaning match the corpus authors' own clean split?

> **Status (16 Nov 2026):** project setup. This README describes the layout only. Results,
> reproduction steps and the metric justification are added during the week, and every number
> will come from files written by the scripts in `src/`.

## Data

[Basque Parliament Speech Corpus 1.0](https://huggingface.co/datasets/gttsehu/basque_parliament_1)
(`gttsehu/basque_parliament_1`, Basque part, CC0). We take a raw sample of the `train` split,
clean it ourselves, and test on the official Basque `test` split. The `dev` split is used for
every decision (thresholds, settings, scoring policy); the `test` split is used once per model,
for the final scores.

## Team

| Member | Role |
|---|---|
| Mahmut (coordinator) | Repository, raw sample, cleaning script and log, data tests |
| Shahem | Scoring code and statistics, scoring policy, metric justification, README, presentation |
| Ibrahim | Whisper-small zero-shot run, wav2vec 2.0 baseline, fine-tuning script and runs |
| Tuğra | NeMo baseline models, error-type analysis, speaker analysis |
| Seleni | Whisper large-v3 and Qwen3-ASR runs, timing helper, figures, licence and citation files |

Each member's own account of their work is in `contributions/`.

## Repository layout

| Path | What it holds |
|---|---|
| `contributions/` | One file per member, written and committed by that member |
| `data/raw/audio/{train,dev,test}/` | MP3 clips, unchanged from the corpus |
| `data/raw/manifests/` | `train_sample.csv`, `dev.csv`, `test.csv` (columns: `clip_id, path, text, duration, speaker_id`) |
| `data/cleaned/` | Cleaned training manifests and `removal_log.csv` (every removed clip and the rule that removed it) |
| `src/` | All scripts |
| `tests/` | `pytest` tests for the data, the cleaning and the scoring |
| `results/predictions/` | One CSV (raw model output) and one JSON (settings, hardware, timing) per run |
| `results/runs/` | Training logs (JSON) |
| `results/figures/` | PNG figures |
| `report/` | Scoring policy, metric justification, error analysis |

Root files:

| File | Purpose |
|---|---|
| `README.md` | This file |
| `requirements.txt` | Pinned packages for scoring, cleaning and Whisper fine-tuning |
| `.gitignore` | Keeps model weights, caches, secrets and notebook checkpoints out of the repo |

The `.gitkeep` files only make the empty folders visible on GitHub; each one is deleted when
the first real file lands in its folder.

## Rules we follow

- Model weights are not committed (`models/` is ignored). We commit the run code, the training
  logs and the predictions.
- Tokens and keys are never committed; they live in a local `.env` file.
- Every installed package is pinned with `==`.
- Nobody edits a results table by hand; the scoring script writes it.
