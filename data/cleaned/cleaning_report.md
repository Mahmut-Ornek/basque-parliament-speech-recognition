# Cleaning report

Written by `src/clean.py`. Do not edit by hand.

Raw sample: 10515 clips, 20.00 h.
After rule 1 (leakage, applied to every condition): 4844 clips, 9.24 h (`train_raw.csv`, the training set of `ft-raw`). No training clip shares a speaker, a sentence or audio with dev or test.
Kept by all our rules: 4466 clips, 8.62 h.

## Thresholds (all chosen on dev)

- Rule 3, characters per second: the 1st and 99th percentiles on dev are 7.85 and 20.60. A training clip is removed below 7.07 or above 22.66 (10% beyond them).
- Rule 3, duration: the duration column and the decoded audio differ by at most 0.01 s on dev. A training clip is removed above 0.11 s, or if the duration disagrees with the times in its file name.
- Rule 5, judge CER threshold: 0.33. On 1397 dev clips Whisper's median CER on untouched transcripts is 0.089. The threshold catches 96.3% of the transcripts we broke on purpose (swapped 100.0%, truncated 88.9%, shifted 100.0%) and removes 4.7% of the untouched ones.
  Digits and symbols are dropped from both sides before the CER is computed, because the references write numbers as words. The number words in the reference still count as deleted, so the table below checks whether the rule mostly removes clips with numbers.
- Rule 6, P(Spanish) threshold: 0.87 (removes 0.9% of the dev clips, all Basque).

Applied to dev (never done for real), the rules would remove: broken text 0, judge 66, wrong language 13 of 1397 clips.

## Judge rule and numbers

Clips that reached rule 5, split by whether Whisper's output contains a digit.

| Whisper output | Clips judged | Removed by rule 5 | Rate |
|---|---|---|---|
| with digits | 101 | 26 | 25.7% |
| without digits | 4621 | 182 | 3.9% |

## Removed per rule

| Rule | Reason | Clips | Hours |
|---|---|---|---|
| 1 | leakage_text | 4 | 0.01 |
| 1 | leakage_audio_overlap | 0 | 0.00 |
| 1 | held_out_speaker | 5241 | 9.86 |
| 1 | unknown_speaker | 426 | 0.89 |
| 2 | broken_text | 0 | 0.00 |
| 3 | speaking_rate | 109 | 0.18 |
| 3 | duration_mismatch | 0 | 0.00 |
| 4 | repeated_text | 0 | 0.00 |
| 4 | repeated_audio_overlap | 13 | 0.02 |
| 5 | judge_cer | 208 | 0.32 |
| 6 | wrong_language | 48 | 0.11 |
| | **total** | 6049 | 11.38 |

## Our filter against the authors' filter

Clips of `train_raw.csv` (rules 2-6 against the authors' `train_clean`). The authors' filter comes from an automatic score, not a human check: a reference, not the truth.

| | Authors keep | Authors remove |
|---|---|---|
| **We keep** | 3908 | 558 |
| **We remove** | 235 | 143 |

## Training sets (equal hours)

| File | Clips | Hours |
|---|---|---|
| `train_ours.csv` | 4264 | 8.230 |
| `train_authors.csv` | 4143 | 8.230 |
| `train_rawmatched.csv` | 4311 | 8.228 |

`train_raw.csv` (4844 clips, 9.241 h) is not size-matched: `ft-raw` tests whether more but noisier hours win.
