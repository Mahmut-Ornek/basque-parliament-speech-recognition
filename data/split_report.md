# Split report

Written by `src/split_report.py`. Do not edit by hand.

## Splits

| Split | Clips | Hours | Speakers | Clips without a speaker | Sessions |
|---|---|---|---|---|---|
| train_sample (raw) | 10515 | 20.00 | 146 | 427 | 702 |
| train_raw | 4844 | 9.24 | 107 | 0 | 646 |
| train_ours | 4264 | 8.23 | 102 | 0 | 636 |
| train_authors | 4143 | 8.23 | 103 | 0 | 636 |
| train_rawmatched | 4311 | 8.23 | 107 | 0 | 638 |
| dev | 1397 | 2.58 | 32 | 48 | 3 |
| test | 1521 | 2.81 | 25 | 46 | 3 |

Speakers counts identified speakers only; speaker_id 0 is treated as unknown.

## Training sets against dev and test

| Training set | Speakers shared with dev | with test | Sessions shared with dev / test | Dev clips whose sentence is in the set | Test clips whose sentence is in the set |
|---|---|---|---|---|---|
| train_sample (raw) | 32 | 25 | 0 / 0 | 1 | 5 |
| train_raw | 0 | 0 | 0 / 0 | 0 | 0 |
| train_ours | 0 | 0 | 0 / 0 | 0 | 0 |
| train_authors | 0 | 0 | 0 / 0 | 0 | 0 |
| train_rawmatched | 0 | 0 | 0 / 0 | 0 | 0 |

## Dev against test

- Speakers in both dev and test: 18 of 32 dev speakers and 25 test speakers. Sessions in both: 0.
- Test clips without a speaker (cannot be checked against any split): 46.
- Dev clips whose speaker is identified and not in test: 573 clips, 1.06 h, 14 speakers.
- Dev is never used for training; it is used to choose thresholds, settings and checkpoints.
