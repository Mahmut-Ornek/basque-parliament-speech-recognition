# Data

## Source

[Basque Parliament Speech Corpus 1.0](https://huggingface.co/datasets/gttsehu/basque_parliament_1)
(`gttsehu/basque_parliament_1`), dataset commit `46208b3001c500aa53ddb5f4a9da81493bbb8067`.
Licence: CC0 1.0. Paper: Varona et al. 2024, [doi:10.3390/app14051951](https://doi.org/10.3390/app14051951).

We use only the clips the corpus labels as Basque (`eu`). Clips labelled Spanish (`es`) or
bilingual (`bi`) are left out of every split.

## What is in this folder

| Path | Content |
|---|---|
| `raw/audio/{train,dev,test}/NNN/` | MP3 clips, bytes unchanged from the corpus, in numbered folders of at most 500 files |
| `raw/manifests/train_sample.csv` | Our raw training sample |
| `raw/manifests/dev.csv`, `test.csv` | Every Basque clip of the official `dev` and `test` splits |

Manifest columns: `clip_id, path, text, duration, speaker_id`. `text` is the corpus transcript,
unchanged. `duration` is in seconds. `path` is relative to the repository root. A `clip_id` is the
corpus file name without extension, for example `12-077_20220211_01_382.04_392.24`: session
(with its date), start second, end second.

## Sizes

| Split | Clips | Hours | Speakers | Clips with unknown speaker | Size (MB) |
|---|---|---|---|---|---|
| train (our sample) | 10515 | 20.00 | 146 | 427 | 221.2 |
| dev | 1397 | 2.58 | 32 | 48 | 28.6 |
| test | 1521 | 2.81 | 25 | 46 | 31.1 |
| total | | | | | 280.8 |

The corpus uses `speaker_id` `0` next to its numbered speakers. The corpus documentation does not
explain it. We treat it as "no identified speaker": it is not counted in the Speakers column, and
its clips are counted separately.

## How the training sample was made

`src/make_sample.py` reads the corpus metadata, keeps the Basque clips of the official `train`
split (216,201 clips), puts them in a random order with seed 42, and takes clips from the start of
that order until 20 hours are reached. All eleven `train` archives were used, so every Basque
training clip had the same chance of being chosen.

The order is nested: the first 12 hours of the sample are its first 6,350 rows, and the first 10
hours are its first 5,277 rows.

The sample covers 702 sessions from 2013 to 2021:

| Year | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 |
|---|---|---|---|---|---|---|---|---|---|
| Clips | 1569 | 1415 | 1040 | 920 | 1250 | 1245 | 1085 | 713 | 1278 |

The `dev` and `test` clips come from sessions held in February 2022, after every training session.

## Reproduce

```
python src/make_sample.py --hours 20 --archives all
```

This downloads about 16.5 GB once into the HuggingFace cache and writes the audio and the three
manifests. Running it again gives identical files.
