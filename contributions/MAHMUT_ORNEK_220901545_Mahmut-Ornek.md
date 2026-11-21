# Mahmut Ornek — 220901545 — @Mahmut-Ornek

**Role:** coordinator; data, cleaning and data tests; seed-3 fine-tuning runs.

## Coordination

- Created the repository, the folder layout, `.gitignore` and the first README; registered the
  project on the language tracker; added the team as collaborators.
- Checked every member's commits for a linked GitHub username, and ran the daily status in the group
  chat.
- Turned the teacher's 18 November emails into changes for the group: training sets with no
  speaker from `dev` or `test`, and baselines run only when their model card lists Basque.
- Final check of the README against the code and the submission checklist.

## Data

- `src/make_sample.py`: downloads the corpus with `huggingface_hub` (its loading script no longer
  works with current `datasets`), keeps the Basque clips, draws a 20-hour raw training sample
  from all eleven `train` archives in a seeded, nested order, exports the full `dev` and `test`
  splits, and writes the audio in folders of at most 500 files plus the three manifests.
- `data/README.md`: source, licence, sampling, splits, cleaning rules and all their numbers.

## Cleaning

- `src/clean.py`: six removal rules, applied in order, with every removed clip logged in
  `data/cleaned/removal_log.csv`. Nobody in the group speaks Basque, so every threshold is chosen
  on the hand-checked `dev` split, and the corpus's own quality score is never read.
  - Rule 1 (leakage), applied to every training set: removes clips whose sentence, audio or
    speaker also appears in `dev` or `test`, and clips without a speaker id.
  - Rules 2–6 (our cleaning): broken text, speaking rate and duration, repeats and overlapping
    audio, a Whisper large-v3 judge, and wrong language.
  - The judge threshold is calibrated by breaking `dev` transcripts on purpose: it catches 96.3%
    of them and removes 4.7% of untouched ones.
  - Writes the four training sets (`train_raw`, `train_ours`, `train_authors`,
    `train_rawmatched`), size-matched to equal hours, and `cleaning_report.md`.
- `src/split_report.py`: speakers, clips and hours per split, and the leak checks the teacher
  asked for (0 shared speakers and 0 test sentences in every training set).

## Tests

- `tests/test_data.py`: every cleaning rule on a hand-made manifest with planted bad clips,
  reproducibility (running the scripts twice gives identical files), nested sampling, and checks
  on the real data (no shared clip, speaker or sentence between training sets and `dev`/`test`;
  every manifest row points to an existing audio file).

## Fine-tuning and decisions

- Trained the seed-3 run of all four conditions (`ft-raw`, `ft-rawmatched`, `ft-authors`,
  `ft-ours`) on my GPU with Ibrahim's script, and transcribed `dev` and `test` with each best
  checkpoint.
- Counted every reference with Whisper's tokenizer (longest: 75 tokens), which settled that the
  128-token limit stays and that clips hitting it are repetition loops.
