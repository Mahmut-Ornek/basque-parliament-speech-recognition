# Metric justification

Which metrics we report, which one is the main metric, and why. The metrics are the ones in
the Hugging Face audio course, chapter 5, "Evaluation metrics for ASR"
(https://huggingface.co/learn/audio-course/chapter5/evaluation). How they are computed is fixed
in `report/scoring_policy.md`; the code is `src/normalize.py` and `src/score_all.py`.

**Summary.** Our main metric is **normalised WER**. We report **normalised CER** next to it for
every run, with a 95% bootstrap interval for both. Speed is reported as **RTFx**. We do not
report word accuracy or orthographic WER, for the reasons below.

| Metric | Reported? | Role |
|---|---|---|
| WER (normalised) | Yes | Main metric |
| CER (normalised) | Yes | Second metric, partial credit for word endings |
| Word accuracy (1 − WER) | No | The same information as WER |
| Orthographic WER | No | Cannot be measured: the references have no capitals or punctuation |
| RTFx (and RTF, its inverse) | Yes | Speed, from `src/timing.py` |

## WER: word error rate (main metric)

WER = (S + D + I) / N: substitutions, deletions and insertions after aligning the model's words
with the reference words, divided by the number of reference words. The course's example:
reference `the cat sat on the mat`, prediction `the cat sit on the` gives 1 substitution,
1 deletion and 0 insertions, so WER = 2 / 6 = 0.333. This exact example is a test in
`tests/test_score.py`.

Why it is the main metric:

- It is the standard metric for speech recognition, and the one every baseline's model card
  reports. Our numbers can be read next to theirs.
- It counts what a reader of the transcript sees: a word is right or wrong.

**WER has no upper bound.** Insertions are not limited, so a model that keeps adding words
can make more errors than there are reference words. Zero-shot Whisper-small shows this: it
often repeats a phrase until it reaches its 128-token output limit, and its test WER is above 1
(see `results/results_table.md`). A WER above 1 is therefore not a bug in the scorer.

We compute WER at **corpus level**: all errors in the split divided by all reference words. The
mean of per-clip WERs would give a 3-word clip the same weight as a 30-word clip.

## CER: character error rate (second metric)

The same formula on characters, spaces included. The course's example gives 5 errors in 22
characters, CER = 0.2273, also a test in `tests/test_score.py`.

Why we report it for every run: **Basque builds words by adding endings.** One word can carry
case, number and definiteness (`legebiltzar` "parliament", `legebiltzarrean` "in the
parliament", `legebiltzarkideok` "you members of parliament"). If a model gets the stem right
and one ending wrong, WER counts the whole word as an error, while CER counts only the wrong
letters. CER therefore separates "almost right" from "completely wrong", which WER cannot.

When WER and CER rank two runs differently, the model is getting different kinds of errors
(many small ending errors against fewer but larger errors). The README reports both, so these
cases are visible rather than hidden by a single number.

## Word accuracy (not reported separately)

Word accuracy = 1 − WER. It is the same information turned into "higher is better", so it adds
nothing that WER does not already say. It is also misleading when WER is above 1: zero-shot
Whisper-small would have a negative word accuracy. The course notes that it is rarely quoted.

## Orthographic or normalised WER

Orthographic WER compares text exactly as written, capitals and punctuation included.
Normalised WER first removes those differences. The course points to the ESB benchmark, which
argues for orthographic WER, because a transcript with correct capitals and punctuation is more
useful to a reader.

**We report normalised WER only, because orthographic WER cannot be measured on this corpus.**
We checked all 1,397 `dev` references (`report/scoring_policy.md`, section 1): they have no
capital letters, no punctuation and no digits; numbers are written as words. A model that writes
correct capitals and punctuation would be counted wrong for them against references that never
had them. Orthographic WER here would measure formatting, not recognition.

Normalisation, applied to the reference and the model output alike: Unicode NFC, lowercase,
punctuation becomes a space, spaces collapsed. Accented letters and digits are kept. Numbers are
**not** rewritten into words, so a model that writes `2022` where the reference says
`bi mila hogeita bi` is counted wrong on that number. On `dev`, zero-shot Whisper-small writes
digits in 12 of 1,397 clips; our fine-tuned Whisper models, which learned from these references,
and the CTC and transducer baselines write none. `report/error_analysis.md` counts number errors
as their own error type so this effect is visible.

## RTFx: speed

RTFx = audio seconds / compute seconds; higher is faster (an RTFx of 50 means one hour of
audio takes about 72 seconds). It is the speed metric of the Hugging Face Open ASR Leaderboard.
We also give the real-time factor (RTF = 1 / RTFx) and the median and 95th-percentile time per
clip.

All models are timed the same way by `src/timing.py`: from "audio loaded in memory" to "text
returned", model loading excluded, 5 warm-up clips first, batch size 1, on a Colab T4.

## Confidence intervals and paired tests

Every WER and CER comes with a 95% confidence interval from a bootstrap over clips (2,000
resamples, seed 42). Two fine-tuning conditions are compared with a **paired** bootstrap on the
same clips, which gives the interval of the difference and a p-value. With 1,521 test clips, a
difference of a few tenths of a WER point can be noise; the intervals say which differences are
real. The five comparisons were fixed before any test score existed; they are listed in
`report/scoring_policy.md`, section 4.
