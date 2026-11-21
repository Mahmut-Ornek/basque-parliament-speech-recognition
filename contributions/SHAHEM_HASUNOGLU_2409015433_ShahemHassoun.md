# Shahem Hasunoglu — 2409015433 — @ShahemHassoun

**Role:** scoring code and statistics, scoring policy, metric justification, README, presentation.

**Question I answer:** under one fixed scoring policy, which differences between models are real
and which are noise?

## Scoring policy (`report/scoring_policy.md`)

- Checked all 1,397 `dev` references before fixing the policy: no capital letters, no punctuation,
  no digits (numbers written as words), all Unicode NFC. So orthographic WER cannot be measured,
  and we report normalised WER only.
- Fixed the normalisation (NFC, lowercase, punctuation becomes a space, spaces collapsed; accents
  and digits kept, numbers not rewritten) and froze it on 18 November, before any test score.
- Fixed the five paired comparisons before any result existed, and the rule that `test` is
  scored once per model while every choice is made on `dev`.

## Code (`src/normalize.py`, `src/score_all.py`)

- `normalize.py`: the policy as code, identical to `clean.normalise` (used to pick checkpoints).
- `score_all.py`:
  - corpus-level WER and CER from `jiwer`'s alignment;
  - 95% confidence intervals by bootstrap over clips (2,000 resamples, seed 42, the same resamples
    for every run);
  - fine-tuning conditions as mean and SD over seeds;
  - the five paired comparisons (difference, 95% interval, two-sided p-value), with the best
    fine-tuned condition chosen on `dev`;
  - the results table in two parts, "Baselines" and "Our contribution", with the "heard our test
    speakers" column read from the baseline audit;
  - writes `results/` and the README's results section, so no number is typed by hand;
  - refuses partial prediction files and never scores a model without Basque support.

## Tests (`tests/test_normalize.py`, `tests/test_score.py`)

- The lecture example (`the cat sat on the mat` / `the cat sit on the`: 1 substitution,
  1 deletion, WER 0.3333, CER 0.2273).
- Normalisation rules, idempotence, NFC, and equality with `clean.normalise` on every dev
  reference and prediction; a check that the dev references are already normalised.
- Corpus-level rather than mean-of-clips WER, WER above 1 from insertions, empty outputs,
  reproducible resamples, paired tests, seed averaging, and an end-to-end run on a small repo.

## Reports and documentation

- `report/metric_justification.md`: WER (main metric), CER (partial credit for Basque word
  endings), word accuracy, orthographic versus normalised WER, RTFx, confidence intervals.
- `README.md`: data, the speaker split, cleaning, baselines and which heard our test speakers,
  baselines not run, metrics, results, how to reproduce, seeds and hardware, every folder and
  root file.

## Presentation

Slides and the talk.
