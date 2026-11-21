# Results (dev split, 1397 clips)

Written by `src/score_all.py`. Do not edit by hand.

Scoring policy: `report/scoring_policy.md`. WER and CER are corpus-level, after `src/normalize.py`. 95% intervals: bootstrap over clips, 2,000 resamples, seed 42.

## Baselines (pretrained models, no training by us)

Heard our test speakers: from `report/baseline_audit.md`. A model that heard them was trained on this corpus, so the comparison is in its favour.

| Run | Model | Heard our test speakers? | WER | WER 95% CI | CER | CER 95% CI |
|---|---|---|---|---|---|---|
| `whisper-large-v3` | `openai/whisper-large-v3` | Unknown | not scored yet | | | |
| `whisper-small-zeroshot` | `openai/whisper-small` | Unknown | 0.9711 | [0.9229, 1.0220] | 0.3683 | [0.3379, 0.3995] |
| `hitz-ctc` | `HiTZ/stt_eu_conformer_ctc_large` | Yes | 0.0525 | [0.0484, 0.0570] | 0.0154 | [0.0140, 0.0169] |
| `hitz-transducer` | `HiTZ/stt_eu_conformer_transducer_large` | Yes | 0.0462 | [0.0424, 0.0501] | 0.0152 | [0.0137, 0.0168] |
| `parakeet-v3-basque` | `itzune/parakeet-tdt-0.6b-v3-basque` | Yes | 0.0443 | [0.0407, 0.0483] | 0.0145 | [0.0131, 0.0160] |
| `wav2vec2-bp1` | `gttsehu/wav2vec2-xls-r-300m-bp1-es_eu` | Yes | 0.0496 | [0.0456, 0.0541] | 0.0143 | [0.0129, 0.0158] |
| `wav2vec2-cv-eu` | `stefan-it/wav2vec2-large-xlsr-53-basque` | No | 0.3050 | [0.2949, 0.3150] | 0.0620 | [0.0592, 0.0648] |
| `mhubert-basque` | `Ansu/mHubert-basque-ASR` | Yes | not scored yet | | | |

## Our contribution (Whisper-small fine-tuned)

Mean over seeds ± SD over seeds. The interval is the bootstrap over clips of the per-clip mean over seeds. None of our training sets contains a `dev` or `test` speaker.

| Run | Training set | Seeds | WER | WER 95% CI | CER | CER 95% CI | WER per seed |
|---|---|---|---|---|---|---|---|
| `ft-raw` | train_raw.csv: raw sample, leakage rule only (9.24 h) | 3 | 0.2088 | [0.1991, 0.2191] | 0.0524 | [0.0480, 0.0580] | 0.2088 |
| `ft-rawmatched` | train_rawmatched.csv: random part of the raw sample, same hours as the cleaned sets | 3 | 0.2136 | [0.2044, 0.2230] | 0.0538 | [0.0493, 0.0592] | 0.2136 |
| `ft-authors` | train_authors.csv: the corpus authors' clean split | 3 | 0.2052 | [0.1971, 0.2142] | 0.0487 | [0.0462, 0.0512] | 0.2052 |
| `ft-ours` | train_ours.csv: our cleaning script | 3 | 0.2159 | [0.2071, 0.2253] | 0.0523 | [0.0496, 0.0552] | 0.2159 |

## Paired comparisons (fixed before any test score)

Difference = A − B; negative means A makes fewer errors. Paired bootstrap on the same clips; two-sided p-value.

| A | B | Question | WER diff | 95% CI | p | CER diff | 95% CI | p |
|---|---|---|---|---|---|---|---|---|
| `ft-ours` | `ft-rawmatched` | Does cleaning help beyond having less data? | +0.0023 | [-0.0035, +0.0078] | 0.427 | -0.0014 | [-0.0067, +0.0020] | 0.598 |
| `ft-ours` | `ft-authors` | Does our cleaning match the authors' cleaning? | +0.0107 | [+0.0055, +0.0159] | < 0.001 | +0.0036 | [+0.0019, +0.0055] | < 0.001 |
| `ft-authors` | `ft-rawmatched` | Does the authors' cleaning help beyond having less data? | -0.0083 | [-0.0147, -0.0031] | < 0.001 | -0.0051 | [-0.0101, -0.0019] | < 0.001 |
| `ft-raw` | `ft-rawmatched` | Do more but noisier hours win? | -0.0047 | [-0.0098, +0.0001] | 0.053 | -0.0014 | [-0.0028, -0.0000] | 0.050 |
| `ft-authors` | `whisper-small-zeroshot` | How much does fine-tuning help? (A chosen by mean dev WER) | -0.7659 | [-0.8171, -0.7175] | < 0.001 | -0.3197 | [-0.3503, -0.2898] | < 0.001 |
