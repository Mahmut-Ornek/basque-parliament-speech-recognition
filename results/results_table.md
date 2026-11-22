# Results (test split, 1521 clips)

Written by `src/score_all.py`. Do not edit by hand.

Scoring policy: `report/scoring_policy.md`. WER and CER are corpus-level, after `src/normalize.py`. 95% intervals: bootstrap over clips, 2,000 resamples, seed 42.

## Baselines (pretrained models, no training by us)

Heard our test speakers: from `report/baseline_audit.md`. A model that heard them was trained on this corpus, so the comparison is in its favour.

| Run | Model | Heard our test speakers? | WER | WER 95% CI | CER | CER 95% CI |
|---|---|---|---|---|---|---|
| `whisper-large-v3` | `openai/whisper-large-v3` | Unknown | 0.4378 | [0.4234, 0.4532] | 0.1256 | [0.1185, 0.1335] |
| `whisper-small-zeroshot` | `openai/whisper-small` | Unknown | 1.0117 | [0.9636, 1.0656] | 0.4016 | [0.3718, 0.4345] |
| `hitz-ctc` | `HiTZ/stt_eu_conformer_ctc_large` | Yes | 0.0475 | [0.0436, 0.0513] | 0.0151 | [0.0137, 0.0166] |
| `hitz-transducer` | `HiTZ/stt_eu_conformer_transducer_large` | Yes | 0.0428 | [0.0392, 0.0464] | 0.0151 | [0.0137, 0.0167] |
| `parakeet-v3-basque` | `itzune/parakeet-tdt-0.6b-v3-basque` | Yes | 0.0468 | [0.0432, 0.0504] | 0.0157 | [0.0142, 0.0173] |
| `wav2vec2-bp1` | `gttsehu/wav2vec2-xls-r-300m-bp1-es_eu` | Yes | 0.0467 | [0.0429, 0.0506] | 0.0145 | [0.0131, 0.0160] |
| `wav2vec2-cv-eu` | `stefan-it/wav2vec2-large-xlsr-53-basque` | No | 0.3338 | [0.3232, 0.3443] | 0.0699 | [0.0670, 0.0730] |
| `mhubert-basque` | `Ansu/mHubert-basque-ASR` | Yes | 0.0594 | [0.0551, 0.0637] | 0.0163 | [0.0149, 0.0178] |

## Our contribution (Whisper-small fine-tuned)

Mean over seeds ± SD over seeds. The interval is the bootstrap over clips of the per-clip mean over seeds. None of our training sets contains a `dev` or `test` speaker.

| Run | Training set | Seeds | WER | WER 95% CI | CER | CER 95% CI | WER per seed |
|---|---|---|---|---|---|---|---|
| `ft-raw` | train_raw.csv: raw sample, leakage rule only (9.24 h) | 1 2 3 | 0.2160 ± 0.0015 | [0.2076, 0.2249] | 0.0555 ± 0.0008 | [0.0524, 0.0587] | 0.2176 0.2146 0.2158 |
| `ft-rawmatched` | train_rawmatched.csv: random part of the raw sample, same hours as the cleaned sets | 1 2 3 | 0.2234 ± 0.0011 | [0.2148, 0.2327] | 0.0560 ± 0.0011 | [0.0532, 0.0591] | 0.2229 0.2247 0.2226 |
| `ft-authors` | train_authors.csv: the corpus authors' clean split | 1 2 3 | 0.2166 ± 0.0014 | [0.2078, 0.2253] | 0.0526 ± 0.0007 | [0.0499, 0.0553] | 0.2159 0.2182 0.2158 |
| `ft-ours` | train_ours.csv: our cleaning script | 1 2 3 | 0.2207 ± 0.0038 | [0.2122, 0.2295] | 0.0552 ± 0.0013 | [0.0522, 0.0585] | 0.2184 0.2250 0.2186 |

## Paired comparisons (fixed before any test score)

Difference = A − B; negative means A makes fewer errors. Paired bootstrap on the same clips; two-sided p-value.

| A | B | Question | WER diff | 95% CI | p | CER diff | 95% CI | p |
|---|---|---|---|---|---|---|---|---|
| `ft-ours` | `ft-rawmatched` | Does cleaning help beyond having less data? | -0.0027 | [-0.0071, +0.0015] | 0.199 | -0.0008 | [-0.0032, +0.0016] | 0.521 |
| `ft-ours` | `ft-authors` | Does our cleaning match the authors' cleaning? | +0.0041 | [+0.0002, +0.0078] | 0.038 | +0.0026 | [+0.0011, +0.0045] | < 0.001 |
| `ft-authors` | `ft-rawmatched` | Does the authors' cleaning help beyond having less data? | -0.0068 | [-0.0114, -0.0024] | 0.002 | -0.0034 | [-0.0057, -0.0015] | < 0.001 |
| `ft-raw` | `ft-rawmatched` | Do more but noisier hours win? | -0.0074 | [-0.0113, -0.0038] | 0.001 | -0.0005 | [-0.0028, +0.0017] | 0.678 |
| `ft-authors` | `whisper-small-zeroshot` | How much does fine-tuning help? (A chosen by mean dev WER) | -0.7951 | [-0.8487, -0.7470] | < 0.001 | -0.3490 | [-0.3819, -0.3198] | < 0.001 |
