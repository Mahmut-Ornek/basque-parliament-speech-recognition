# Results (dev split, 1397 clips)

Written by `src/score_all.py`. Do not edit by hand.

Scoring policy: `report/scoring_policy.md`. WER and CER are corpus-level, after `src/normalize.py`. 95% intervals: bootstrap over clips, 2,000 resamples, seed 42.

## Baselines (pretrained models, no training by us)

Heard our test speakers: from `report/baseline_audit.md`. A model that heard them was trained on this corpus, so the comparison is in its favour.

| Run | Model | Heard our test speakers? | WER | WER 95% CI | CER | CER 95% CI |
|---|---|---|---|---|---|---|
| `whisper-large-v3` | `openai/whisper-large-v3` | Unknown | 0.4272 | [0.4146, 0.4407] | 0.1204 | [0.1131, 0.1286] |
| `whisper-small-zeroshot` | `openai/whisper-small` | Unknown | 0.9711 | [0.9229, 1.0220] | 0.3683 | [0.3379, 0.3995] |
| `hitz-ctc` | `HiTZ/stt_eu_conformer_ctc_large` | Yes | 0.0525 | [0.0484, 0.0570] | 0.0154 | [0.0140, 0.0169] |
| `hitz-transducer` | `HiTZ/stt_eu_conformer_transducer_large` | Yes | 0.0462 | [0.0424, 0.0501] | 0.0152 | [0.0137, 0.0168] |
| `parakeet-v3-basque` | `itzune/parakeet-tdt-0.6b-v3-basque` | Yes | 0.0443 | [0.0407, 0.0483] | 0.0145 | [0.0131, 0.0160] |
| `wav2vec2-bp1` | `gttsehu/wav2vec2-xls-r-300m-bp1-es_eu` | Yes | 0.0496 | [0.0456, 0.0541] | 0.0143 | [0.0129, 0.0158] |
| `wav2vec2-cv-eu` | `stefan-it/wav2vec2-large-xlsr-53-basque` | No | 0.3050 | [0.2949, 0.3150] | 0.0620 | [0.0592, 0.0648] |
| `mhubert-basque` | `Ansu/mHubert-basque-ASR` | Yes | 0.0604 | [0.0559, 0.0650] | 0.0149 | [0.0136, 0.0163] |

## Our contribution (Whisper-small fine-tuned)

Mean over seeds ± SD over seeds. The interval is the bootstrap over clips of the per-clip mean over seeds. None of our training sets contains a `dev` or `test` speaker.

| Run | Training set | Seeds | WER | WER 95% CI | CER | CER 95% CI | WER per seed |
|---|---|---|---|---|---|---|---|
| `ft-raw` | train_raw.csv: raw sample, leakage rule only (9.24 h) | 1 2 3 | 0.2081 ± 0.0010 | [0.1997, 0.2169] | 0.0511 ± 0.0012 | [0.0481, 0.0544] | 0.2084 0.2069 0.2088 |
| `ft-rawmatched` | train_rawmatched.csv: random part of the raw sample, same hours as the cleaned sets | 1 2 3 | 0.2153 ± 0.0018 | [0.2068, 0.2243] | 0.0532 ± 0.0005 | [0.0502, 0.0563] | 0.2154 0.2171 0.2136 |
| `ft-authors` | train_authors.csv: the corpus authors' clean split | 1 2 3 | 0.2077 ± 0.0022 | [0.1995, 0.2163] | 0.0494 ± 0.0006 | [0.0470, 0.0519] | 0.2087 0.2092 0.2052 |
| `ft-ours` | train_ours.csv: our cleaning script | 1 2 3 | 0.2150 ± 0.0009 | [0.2067, 0.2238] | 0.0522 ± 0.0004 | [0.0495, 0.0549] | 0.2140 0.2151 0.2159 |

## Paired comparisons (fixed before any test score)

Difference = A − B; negative means A makes fewer errors. Paired bootstrap on the same clips; two-sided p-value.

| A | B | Question | WER diff | 95% CI | p | CER diff | 95% CI | p |
|---|---|---|---|---|---|---|---|---|
| `ft-ours` | `ft-rawmatched` | Does cleaning help beyond having less data? | -0.0004 | [-0.0038, +0.0033] | 0.842 | -0.0011 | [-0.0030, +0.0006] | 0.241 |
| `ft-ours` | `ft-authors` | Does our cleaning match the authors' cleaning? | +0.0073 | [+0.0034, +0.0111] | < 0.001 | +0.0028 | [+0.0016, +0.0039] | < 0.001 |
| `ft-authors` | `ft-rawmatched` | Does the authors' cleaning help beyond having less data? | -0.0076 | [-0.0117, -0.0037] | < 0.001 | -0.0038 | [-0.0059, -0.0021] | < 0.001 |
| `ft-raw` | `ft-rawmatched` | Do more but noisier hours win? | -0.0073 | [-0.0105, -0.0040] | < 0.001 | -0.0022 | [-0.0031, -0.0012] | < 0.001 |
| `ft-authors` | `whisper-small-zeroshot` | How much does fine-tuning help? (A chosen by mean dev WER) | -0.7634 | [-0.8145, -0.7142] | < 0.001 | -0.3189 | [-0.3499, -0.2887] | < 0.001 |
