# Data
 
## Source
 
[Basque Parliament Speech Corpus 1.0](https://huggingface.co/datasets/gttsehu/basque_parliament_1)
(`gttsehu/basque_parliament_1`), dataset commit `46208b3001c500aa53ddb5f4a9da81493bbb8067`.
Licence: CC0 1.0. Paper: Varona et al. 2024, [doi:10.3390/app14051951](https://doi.org/10.3390/app14051951).
 
We use only the clips the corpus labels as Basque (`eu`). Clips labelled Spanish (`es`) or
bilingual (`bi`) are left out of every split. The corpus's `train` clips were cut and aligned
automatically; its `dev` and `test` clips were checked by hand.
 
## What is in this folder
 
| Path | Content | Written by |
|---|---|---|
| `raw/audio/{train,dev,test}/NNN/` | MP3 clips, bytes unchanged from the corpus, in numbered folders of at most 500 files | `src/make_sample.py` |
| `raw/manifests/train_sample.csv` | Our raw training sample (20 h) | `src/make_sample.py` |
| `raw/manifests/dev.csv`, `test.csv` | Every Basque clip of the official `dev` and `test` splits | `src/make_sample.py` |
| `cleaned/train_raw.csv` | Raw sample minus leakage (rule 1): training set of `ft-raw` | `src/clean.py` |
| `cleaned/train_ours.csv` | Clips kept by all our rules, size-matched: `ft-ours` | `src/clean.py` |
| `cleaned/train_authors.csv` | Clips of `train_raw.csv` that are in the authors' `train_clean`: `ft-authors` | `src/clean.py` |
| `cleaned/train_rawmatched.csv` | Random clips of `train_raw.csv`, same hours: `ft-rawmatched` | `src/clean.py` |
| `cleaned/removal_log.csv` | Every removed clip: `clip_id, rule, value, threshold` | `src/clean.py` |
| `cleaned/judge_{dev,train}.csv` | Whisper large-v3 output, language probabilities and decoded duration per clip | `src/clean.py --judge` |
| `cleaned/judge_{dev,train}.json` | Judge model, revision, decoding settings and hardware | `src/clean.py --judge` |
| `cleaned/cleaning_report.md` | Thresholds, calibration, removals per rule, comparison with the authors | `src/clean.py` |
| `split_report.md` | Speakers, clips and hours per split; overlap checks | `src/split_report.py` |
 
All manifests have the columns `clip_id, path, text, duration, speaker_id`. `text` is the corpus
transcript, unchanged. `duration` is in seconds. `path` is relative to the repository root. A
`clip_id` is the corpus file name without extension, for example
`12-077_20220211_01_382.04_392.24`: session (with its date), start second, end second.
 
The corpus uses `speaker_id` `0` next to its numbered speakers, and its documentation does not
explain it. We treat it as "no identified speaker".
 
## Raw sample
 
`src/make_sample.py` keeps the Basque clips of the official `train` split (216,201 clips), puts
them in a random order with seed 42, and takes clips from the start of that order until 20 hours
are reached. All eleven `train` archives were used, so every Basque training clip had the same
chance of being chosen. The sample covers 702 sessions from 2013 to 2021:
 
| Year | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 |
|---|---|---|---|---|---|---|---|---|---|
| Clips | 1569 | 1415 | 1040 | 920 | 1250 | 1245 | 1085 | 713 | 1278 |
 
Audio on disk: train 221.2 MB, dev 28.6 MB, test 31.1 MB, 280.8 MB in total.
 
The repository holds the whole 20-hour raw sample, but training uses at most 9.24 hours of it.
About half the raw sample came from speakers who also speak in the February 2022 `dev` and `test`
sessions, so the speaker rule below removed them. That share also shows that the speaker leak
was real, not theoretical.
 
## Splits and leakage
 
Rule from the course: all clips of one speaker go to the same split. We keep the corpus's own
documented `dev` and `test` splits (hand-checked, and the same clips for every system we score).
They are not speaker-separate from the corpus's `train`, so we made **our training sets**
separate: rule 1 below removes every training clip whose speaker appears in `dev` or `test`, and
every clip with `speaker_id` 0, because its speaker cannot be checked. This applies to all four
training sets.
 
| Split | Clips | Hours | Speakers | Clips without a speaker | Sessions |
|---|---|---|---|---|---|
| raw sample (before rule 1) | 10515 | 20.00 | 146 | 427 | 702 |
| `train_raw` | 4844 | 9.24 | 107 | 0 | 646 |
| `train_ours` | 4264 | 8.23 | 102 | 0 | 636 |
| `train_authors` | 4143 | 8.23 | 103 | 0 | 636 |
| `train_rawmatched` | 4311 | 8.23 | 107 | 0 | 638 |
| `dev` | 1397 | 2.58 | 32 | 48 | 3 |
| `test` | 1521 | 2.81 | 25 | 46 | 3 |
 
Checks, from `split_report.md`:
 
- Every training set shares **0 speakers** and **0 sessions** with `dev` and with `test`.
- **0 test clips** (and 0 dev clips) have a sentence that appears word for word in any training
  set. Before rule 1, 5 test clips and 1 dev clip had their sentence in the raw sample.
- The speaker rule is what removes most of the raw sample: 5,241 clips (9.86 h) by speakers of
  `dev` or `test`, and 426 clips (0.89 h) without a speaker. Many of the members who speak in
  February 2022 also spoke in earlier years.
- `dev` and `test` come from different sessions of February 2022 but share 18 speakers (of 32 in
  `dev` and 25 in `test`). `dev` is never used for training, only to choose thresholds, settings
  and checkpoints.
- 46 `test` clips have no speaker id, so they cannot be checked against any split.
## Cleaning
 
`src/clean.py` only **removes** training clips. It never rewrites a transcript and never changes
`dev` or `test`. Nobody in the group speaks Basque, so every decision is a rule, and every
threshold is chosen on the hand-checked `dev` split, never on `test`.
 
The corpus's own quality score (`PRR`) is never read: it is how the authors built `train_clean`,
and using it would make our filter a copy of theirs. Whisper large-v3 is used as a **warning
signal**, not as ground truth. It was chosen as the judge because, unlike the HiTZ models, it was
not trained on this corpus.
 
Rules, in this order. A clip is logged under the first rule that removes it.
 
| # | Rule | Removes a clip when | Threshold (from `dev`) |
|---|---|---|---|
| 1 | Leakage | its sentence is in `dev`/`test`, its audio overlaps a `dev`/`test` clip, its speaker speaks in `dev`/`test`, or it has no speaker | — |
| 2 | Broken text | the transcript is empty or has a character outside the Basque and Spanish alphabets | — |
| 3 | Timing | characters per second fall outside the `dev` 1st–99th percentile range widened by 10%, or the duration disagrees with the times in the file name or with the decoded audio | 7.07–22.66 chars/s; 0.11 s |
| 4 | Repeats | the same sentence appears more than 3 times, or the audio overlaps a kept clip of the same session | 3 copies |
| 5 | Judge | the character error rate between Whisper's output (language forced to Basque) and the transcript is too high | CER > 0.33 |
| 6 | Wrong language | Whisper's language detection, run separately on each clip, gives Spanish a high probability | P(Spanish) ≥ 0.87 |
 
Rule 1 is leakage prevention, not cleaning, so it applies to every training condition. Rules 2–6
are our cleaning and apply only to `train_ours.csv`.
 
### Calibrating the judge without a Basque speaker
 
Whisper is weak on Basque: on the untouched `dev` transcripts its median CER is 0.089. To find a
threshold, we broke `dev` transcripts on purpose in three ways (another clip's transcript, the
last third of the words cut off, the next clip's transcript) and chose the threshold that best
separates broken from untouched clips.
 
| | Rate at CER > 0.33 |
|---|---|
| Broken transcripts caught | 96.3% (swapped 100%, truncated 88.9%, shifted 100%) |
| Untouched `dev` transcripts removed | 4.7% (66 of 1,397) |
 
This is also our evidence that the rule works: it catches almost every misaligned transcript we
made, at the cost of removing about 1 good clip in 20. Rule 6 would remove 0.9% of the `dev` clips
(13 of 1,397), all of which are Basque, so that is its false-removal rate.
 
**Numbers.** The references write numbers as words, and Whisper often writes digits. We drop
digits and symbols from both sides before computing the CER, but the number words in the reference
still count as missing. Rule 5 therefore removes clips with numbers more often: 25.7% of the clips
whose Whisper output has a digit (26 of 101), against 3.9% of the others (182 of 4,621). At the
lower rate we would expect about 4 of those 101 clips to go, so about 22 clips (under 0.5% of
`train_raw`) are removed because they contain numbers, not because they are misaligned.
 
### What was removed
 
| Rule | Reason | Clips | Hours |
|---|---|---|---|
| 1 | sentence also in `dev`/`test` | 4 | 0.01 |
| 1 | audio overlaps `dev`/`test` | 0 | 0.00 |
| 1 | speaker also in `dev`/`test` | 5241 | 9.86 |
| 1 | no speaker id | 426 | 0.89 |
| 2 | broken text | 0 | 0.00 |
| 3 | speaking rate | 109 | 0.18 |
| 3 | duration mismatch | 0 | 0.00 |
| 4 | repeated sentence | 0 | 0.00 |
| 4 | overlapping audio | 13 | 0.02 |
| 5 | judge CER | 208 | 0.32 |
| 6 | wrong language | 48 | 0.11 |
| | **total** | 6049 | 11.38 |
 
After rule 1, 4,844 clips (9.24 h) remain. Rules 2–6 remove 378 of them (0.62 h, 7.8%) and keep
4,466 clips (8.62 h).
 
### Our filter against the authors' filter
 
Clips of `train_raw.csv`. The authors' `train_clean` comes from an automatic score, not a human
check, so it is a reference, not the truth.
 
| | Authors keep | Authors remove |
|---|---|---|
| **We keep** | 3908 | 558 |
| **We remove** | 235 | 143 |
 
- The two filters agree on 83.6% of the clips, but much of that is because both keep most clips.
  Cohen's kappa, which corrects for that, is 0.18: weak agreement.
- The authors remove 14.5% of all clips, but 37.8% of the clips we remove, so a clip we remove is
  about 2.6 times as likely to be one they remove too.
- We find 20.4% of the clips they remove. Their score comes from the corpus's own recognition and
  alignment pipeline; our rules see only the transcript, its timing and one Whisper
  transcription, so the two filters catch partly different problems.
- Weak agreement does not tell which filter is right; neither is checked by a person. It does
  mean that `ft-ours` and `ft-authors` really train on different clips, which is what makes their
  comparison informative.
### Size matching
 
The fine-tuning conditions must differ in which clips they see, not in how much audio. Let H be
the smaller of our kept clips (8.62 h) and the authors' kept clips (8.23 h): H = 8.23 h. Our set
is cut to H at random (seed 42), and `train_rawmatched.csv` is a random H hours of
`train_raw.csv` (seed 42).
 
| File | Clips | Hours | Condition |
|---|---|---|---|
| `train_raw.csv` | 4844 | 9.24 | `ft-raw` (not size-matched: do more but noisier hours win?) |
| `train_ours.csv` | 4264 | 8.23 | `ft-ours` |
| `train_authors.csv` | 4143 | 8.23 | `ft-authors` |
| `train_rawmatched.csv` | 4311 | 8.23 | `ft-rawmatched` |
 
## Reproduce
 
```
python src/make_sample.py --hours 20 --archives all   # downloads about 16.5 GB once
python src/clean.py --judge dev                       # GPU; Whisper large-v3
python src/clean.py --judge train                     # GPU; resumes if stopped
python src/clean.py                                   # no GPU; rules and output files
python src/split_report.py                            # split_report.md
```
 
The judge steps save Whisper's output, so the last two steps give identical files on any machine.
Running any step twice gives identical files. Seed 42 is used for sampling, for the calibration
and for size matching.
 