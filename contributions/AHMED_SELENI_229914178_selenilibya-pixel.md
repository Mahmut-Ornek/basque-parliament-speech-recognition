# Contribution: Ahmed Seleni (229914178, @selenilibya-pixel)

> Note from the coordinator (Mahmut Ornek, 22 November): the version of this file that was
> uploaded still had the template placeholders unfilled, and it described work that is not in
> the repository. This version lists only what the repository history shows. Everything below
> comes from the commit of 22 November (3976da8) and the files it added.

## What I did

All in one commit on 22 November, uploaded through the GitHub web interface:

- **`LICENSE`** (first version): MIT licence for our code, plus a paragraph saying the data in
  `data/` is the Basque Parliament Speech Corpus 1.0 (CC0) and that the pretrained models keep
  their own licences.
- **`CITATION.md`** (first version): the corpus, its paper (doi:10.3390/app14051951) with a
  BibTeX entry, and the models with their links and licences, including the non-commercial
  CC-BY-NC-SA licence of `Ansu/mHubert-basque-ASR`.
- **`src/timing.py`** (first version): a timing helper that measures every model on the same
  100 dev clips, at batch size 1, after 5 warm-up clips, from "audio in memory" to "text
  returned", with model loading excluded. It records RTFx, the real-time factor, the median and
  95th-percentile time per clip, the GPU and the precision.
- **`src/runtime_table.py`** (first version): reads the timing JSON files and the training logs,
  and writes `results/runtime_table.md`.

## How I worked
- I used an AI assistant (Claude) to draft `LICENSE`, `CITATION.md`, `timing.py` and
  `runtime_table.py`.

## What did not get done by me
- The files were uploaded to the repository root under browser-copy names. Mahmut moved them to
  the paths in the task file.
- Mahmut then corrected them: `CITATION.md` (it listed three models we do not run and missed
  two we do), the copyright line in `LICENSE`, and the timing and runtime-table scripts (token
  limit, NeMo loading, models we do not run, training time).
- The `whisper-large-v3` and `mhubert-basque` runs, the timing pass and the runtime table were
  done by Mahmut. The figures are Shahem's.
