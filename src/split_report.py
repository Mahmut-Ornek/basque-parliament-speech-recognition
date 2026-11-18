"""Check and report how the data is split, as the teacher asks for Project 3:

- for every split: clips (utterances), hours, identified speakers, clips without a speaker, sessions;
- for every training set against dev and test: shared speakers, shared sessions, and how many dev
  and test sentences appear word for word (after normalisation) in the training set;
- dev against test: shared speakers and sessions (both are the corpus's own documented splits).

It reads the manifests in data/raw/manifests/ and, when they exist, the training sets in
data/cleaned/. It writes data/split_report.md. Nothing is changed.

    python src/split_report.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import clean
import make_sample

UNKNOWN = make_sample.UNKNOWN_SPEAKER
TRAINING_SETS = ["train_raw", "train_ours", "train_authors", "train_rawmatched"]


def speakers(rows: list[dict]) -> set[str]:
    return {r["speaker_id"] for r in rows} - {UNKNOWN}


def sessions(rows: list[dict]) -> set[str]:
    return {t[0] for t in (clean.clip_times(r["clip_id"]) for r in rows) if t}


def sentences(rows: list[dict]) -> set[str]:
    return {clean.normalise(r["text"]) for r in rows} - {""}


def verbatim(held_out: list[dict], train_sentences: set[str]) -> int:
    """Number of held-out clips whose normalised sentence also appears in the training set."""
    return sum(clean.normalise(r["text"]) in train_sentences for r in held_out)


def build(splits: dict[str, list[dict]]) -> str:
    dev, test = splits["dev"], splits["test"]
    lines = ["# Split report", "", "Written by `src/split_report.py`. Do not edit by hand.", "",
             "## Splits", "",
             "| Split | Clips | Hours | Speakers | Clips without a speaker | Sessions |", "|---|---|---|---|---|---|"]
    for name, rows in splits.items():
        lines.append(
            f"| {name} | {len(rows)} | {clean.hours(rows):.2f} | {len(speakers(rows))} | "
            f"{sum(r['speaker_id'] == UNKNOWN for r in rows)} | {len(sessions(rows))} |"
        )
    lines += ["", "Speakers counts identified speakers only; speaker_id 0 is treated as unknown.", "",
              "## Training sets against dev and test", "",
              "| Training set | Speakers shared with dev | with test | Sessions shared with dev / test | "
              "Dev clips whose sentence is in the set | Test clips whose sentence is in the set |",
              "|---|---|---|---|---|---|"]
    for name, rows in splits.items():
        if name in ("dev", "test"):
            continue
        s, sess, sent = speakers(rows), sessions(rows), sentences(rows)
        lines.append(
            f"| {name} | {len(s & speakers(dev))} | {len(s & speakers(test))} | "
            f"{len(sess & sessions(dev))} / {len(sess & sessions(test))} | {verbatim(dev, sent)} | {verbatim(test, sent)} |"
        )

    shared = speakers(dev) & speakers(test)
    dev_only = [r for r in dev if r["speaker_id"] not in speakers(test) and r["speaker_id"] != UNKNOWN]
    lines += ["", "## Dev against test", "",
              f"- Speakers in both dev and test: {len(shared)} of {len(speakers(dev))} dev speakers and "
              f"{len(speakers(test))} test speakers. Sessions in both: {len(sessions(dev) & sessions(test))}.",
              f"- Test clips without a speaker (cannot be checked against any split): "
              f"{sum(r['speaker_id'] == UNKNOWN for r in test)}.",
              f"- Dev clips whose speaker is identified and not in test: {len(dev_only)} clips, "
              f"{clean.hours(dev_only):.2f} h, {len(speakers(dev_only))} speakers.",
              "- Dev is never used for training; it is used to choose thresholds, settings and checkpoints."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=str(Path(__file__).resolve().parents[1]), help="repository root")
    args = ap.parse_args(argv)
    root = Path(args.out_root)
    manifests, cleaned = root / "data" / "raw" / "manifests", root / "data" / "cleaned"
    splits = {"train_sample (raw)": clean.read_csv(manifests / "train_sample.csv")}
    for name in TRAINING_SETS:
        if (cleaned / f"{name}.csv").is_file():
            splits[name] = clean.read_csv(cleaned / f"{name}.csv")
    splits["dev"] = clean.read_csv(manifests / "dev.csv")
    splits["test"] = clean.read_csv(manifests / "test.csv")
    report = build(splits)
    (root / "data" / "split_report.md").write_text(report, encoding="utf-8", newline="\n")
    print(report)


if __name__ == "__main__":
    main()
