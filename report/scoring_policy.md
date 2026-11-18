# Scoring policy

**Frozen on 18 November 2026, before any test score was computed. Decided on `dev` only.**
Code: `src/normalize.py` (normalisation) and `src/score_all.py` (metrics and statistics).
Tests: `tests/test_normalize.py`, `tests/test_score.py`.

## 1. What the references look like

We checked all 1,397 references of the Basque `dev` split (`data/raw/manifests/dev.csv`, the
corpus transcripts unchanged):

| Property | Dev references |
|---|---|
| Upper-case letters | none |
| Punctuation | none |
| Digits | none: numbers are written as words (`bi mila hogeita bi` for 2022) |
| Unicode form | all NFC |
| Extra spaces | none |
| Characters used | `a`–`z`, the space, and `á é í ó ú ñ` |

`tests/test_normalize.py` checks this on every dev reference: normalising them changes nothing.

**Consequence: we report normalised WER only.** Orthographic WER compares text with its
capitals and punctuation. Our references have neither, so a model's capitals and punctuation
would count as errors against text that never contained them. The orthographic score would
measure formatting, not recognition.

## 2. Normalisation (applied to reference and hypothesis alike)

1. **Unicode NFC.** An accented letter becomes one character (`á`, not `a` + accent mark), so
   the same word cannot differ only in how it is encoded.
2. **Lowercase.** The references are lowercase.
3. **Punctuation becomes a space.** Every Unicode punctuation character (category P:
   `. , ; : ! ? ¿ ¡ ' " - « » ( ) %` and the like). A space rather than nothing, so
   `lehendakari-ordea` and `lehendakari ordea` score the same and two words around a comma do
   not merge into one.
4. **Whitespace collapsed.** Runs of spaces, tabs or line breaks become one space; leading and
   trailing spaces are removed.

**Not changed:**

- **Letters, accents included** (`á é í ó ú ñ ü ç`): they are part of Basque and Spanish
  spelling, and the references use them.
- **Digits.** Numbers are not rewritten into words: nobody in the group can check Basque number
  words, and a wrong rewrite rule would hide or invent errors. A model that writes `2022` where
  the reference says `bi mila hogeita bi` is counted wrong on that number. Whisper models can
  write digits (zero-shot Whisper-small does in 12 of the 1,397 dev clips; the CTC models never
  do), so they lose some points here. `report/error_analysis.md` counts number errors as their
  own error type, so the size of this effect is reported openly.
- **Symbols that are not punctuation** (`€ + =`): kept, and count as errors.
- **Spanish words** inside Basque speech: kept. The speakers code-switch, and the references
  write those words as spoken.
- **Prediction files** keep the raw model output. Only the scorer normalises.

The same four steps are used by `clean.normalise`, which picks the fine-tuning checkpoint on
`dev`. `tests/test_normalize.py` checks that both functions give the same text on every dev
reference and every dev prediction.

## 3. Metrics

- **WER** (word error rate) = (substitutions + deletions + insertions) / reference words.
- **CER** (character error rate): the same on characters, spaces included.
- Both are **corpus-level**: all errors over all reference words (or characters) of the split,
  from `jiwer` 4.0.0's alignment. Not the mean of per-clip rates, which would give a 3-word
  clip the same weight as a 30-word clip.
- **WER can exceed 1.** Insertions have no limit: a model that repeats itself until it hits its
  token limit makes more errors than there are reference words.
- Why both, and which is the main metric: `report/metric_justification.md`.

## 4. Statistics

- **95% confidence interval:** bootstrap over clips, 2,000 resamples, seed 42, percentile
  method. The same resamples are used for every run.
- **Fine-tuning conditions** (`ft-raw`, `ft-rawmatched`, `ft-authors`, `ft-ours`) have several
  seeds. The table shows the mean over seeds, the SD over seeds and each seed's WER. The
  interval is the bootstrap of the per-clip error count averaged over seeds.
- **Paired comparisons**, fixed before any test score (difference A − B, negative means A
  makes fewer errors; paired bootstrap on the same clips; two-sided p-value = 2 × the share of
  resamples on the other side of zero, at most 1):

  | # | A | B | Question |
  |---|---|---|---|
  | 1 | `ft-ours` | `ft-rawmatched` | Does cleaning help beyond having less data? |
  | 2 | `ft-ours` | `ft-authors` | Does our cleaning match the authors' cleaning? |
  | 3 | `ft-authors` | `ft-rawmatched` | Does the authors' cleaning help beyond having less data? |
  | 4 | `ft-raw` | `ft-rawmatched` | Do more but noisier hours win? |
  | 5 | best fine-tuned condition | `whisper-small-zeroshot` | How much does fine-tuning help? |

  The "best fine-tuned condition" in comparison 5 is the one with the lowest mean **dev** WER.

## 5. Rules

- A run is scored only if it has exactly one prediction for every clip of the split; a partial
  file is skipped with a message, never scored on a subset.
- Models without Basque support (`canary-1b-v2`, `parakeet-tdt-0.6b-v3`, `Qwen3-ASR-1.7B`) are
  never scored, even if a prediction file exists.
- `test` is scored once per model, for the final table. Every choice (this policy, thresholds,
  settings, checkpoints, the best condition in comparison 5) is made on `dev`.
- `results/results_table.*` and `results/paired_comparisons.csv` are written by
  `python src/score_all.py` and never edited by hand. `python src/score_all.py --split dev`
  writes the same tables for `dev`, with names ending in `_dev`.
- A change to this policy needs the whole group's agreement and is recorded here with its date.
