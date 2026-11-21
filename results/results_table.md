# Results (test split, 1521 clips)

Written by `src/score_all.py`. Do not edit by hand.

Scoring policy: `report/scoring_policy.md`. WER and CER are corpus-level, after `src/normalize.py`. 95% intervals: bootstrap over clips, 2,000 resamples, seed 42.

## Baselines (pretrained models, no training by us)

Heard our test speakers: from `report/baseline_audit.md`. A model that heard them was trained on this corpus, so the comparison is in its favour.

| Run | Model | Heard our test speakers? | WER | WER 95% CI | CER | CER 95% CI |
|---|---|---|---|---|---|---|
| `whisper-large-v3` | `openai/whisper-large-v3` | Unknown | not scored yet | | | |
| `whisper-small-zeroshot` | `openai/whisper-small` | Unknown | 1.0117 | [0.9636, 1.0656] | 0.4016 | [0.3718, 0.4345] |
| `hitz-ctc` | `HiTZ/stt_eu_conformer_ctc_large` | Yes | 0.0475 | [0.0436, 0.0513] | 0.0151 | [0.0137, 0.0166] |
| `hitz-transducer` | `HiTZ/stt_eu_conformer_transducer_large` | Yes | 0.0428 | [0.0392, 0.0464] | 0.0151 | [0.0137, 0.0167] |
| `parakeet-v3-basque` | `itzune/parakeet-tdt-0.6b-v3-basque` | Yes | 0.0468 | [0.0432, 0.0504] | 0.0157 | [0.0142, 0.0173] |
| `wav2vec2-bp1` | `gttsehu/wav2vec2-xls-r-300m-bp1-es_eu` | Yes | not scored yet | | | |
| `wav2vec2-cv-eu` | `stefan-it/wav2vec2-large-xlsr-53-basque` | No | 0.3338 | [0.3232, 0.3443] | 0.0699 | [0.0670, 0.0730] |
| `mhubert-basque` | `Ansu/mHubert-basque-ASR` | Yes | not scored yet | | | |

## Our contribution (Whisper-small fine-tuned)

Mean over seeds ± SD over seeds. The interval is the bootstrap over clips of the per-clip mean over seeds. None of our training sets contains a `dev` or `test` speaker.

| Run | Training set | Seeds | WER | WER 95% CI | CER | CER 95% CI | WER per seed |
|---|---|---|---|---|---|---|---|
| `ft-raw` | train_raw.csv: raw sample, leakage rule only (9.24 h) | 3 | 0.2158 | [0.2070, 0.2257] | 0.0565 | [0.0526, 0.0614] | 0.2158 |
| `ft-rawmatched` | train_rawmatched.csv: random part of the raw sample, same hours as the cleaned sets | 3 | 0.2226 | [0.2137, 0.2319] | 0.0554 | [0.0527, 0.0582] | 0.2226 |
| `ft-authors` | train_authors.csv: the corpus authors' clean split | 3 | 0.2158 | [0.2068, 0.2248] | 0.0520 | [0.0495, 0.0547] | 0.2158 |
| `ft-ours` | train_ours.csv: our cleaning script | 3 | 0.2186 | [0.2097, 0.2281] | 0.0542 | [0.0516, 0.0569] | 0.2186 |

## Paired comparisons (fixed before any test score)

Difference = A − B; negative means A makes fewer errors. Paired bootstrap on the same clips; two-sided p-value.

| A | B | Question | WER diff | 95% CI | p | CER diff | 95% CI | p |
|---|---|---|---|---|---|---|---|---|
| `ft-ours` | `ft-rawmatched` | Does cleaning help beyond having less data? | -0.0040 | [-0.0089, +0.0010] | 0.116 | -0.0012 | [-0.0028, +0.0002] | 0.098 |
| `ft-ours` | `ft-authors` | Does our cleaning match the authors' cleaning? | +0.0028 | [-0.0019, +0.0078] | 0.257 | +0.0021 | [+0.0007, +0.0036] | 0.002 |
| `ft-authors` | `ft-rawmatched` | Does the authors' cleaning help beyond having less data? | -0.0068 | [-0.0124, -0.0015] | 0.010 | -0.0034 | [-0.0051, -0.0018] | < 0.001 |
| `ft-raw` | `ft-rawmatched` | Do more but noisier hours win? | -0.0068 | [-0.0120, -0.0013] | 0.019 | +0.0011 | [-0.0017, +0.0051] | 0.658 |
| `ft-authors` | `whisper-small-zeroshot` | How much does fine-tuning help? (A chosen by mean dev WER) | -0.7960 | [-0.8491, -0.7470] | < 0.001 | -0.3496 | [-0.3822, -0.3202] | < 0.001 |
