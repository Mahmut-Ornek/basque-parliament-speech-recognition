"""Rule-based ASR error-type analysis, Weighted WER, and repetition-loop counter.

Reads raw predictions from results/predictions/<run>_<split>.csv, normalises both
references and hypotheses using src/normalize.py, aligns words with jiwer, and
classifies every word error into mutually exclusive categories:

  1. number      : reference or predicted word contains a digit (0-9)
  2. suffix_only : substituted word pair sharing a prefix of >= 4 chars and
                   >= 60% of the longer word (correct stem, wrong ending)
  3. other_sub   : all remaining substitutions
  4. deletion    : remaining reference words deleted by the model
  5. insertion   : remaining hypothesis words inserted by the model

Weighted WER weights suffix-only errors at 0.5 and all other errors at 1.0.
Also counts clips exhibiting repetition loops (e.g., Whisper models looping
until max_new_tokens=128).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import jiwer

# Allow running directly via `python src/error_types.py`
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from normalize import normalize  # noqa: E402

BASELINE_ORDER = [
    "hitz-ctc",
    "hitz-transducer",
    "parakeet-v3-basque",
    "wav2vec2-bp1",
    "wav2vec2-cv-eu",
    "mhubert-basque",
    "whisper-large-v3",
    "whisper-small-zeroshot",
]


def has_digit(word: str) -> bool:
    return any(ch.isdigit() for ch in word)


def shared_prefix_len(w1: str, w2: str) -> int:
    n = 0
    for a, b in zip(w1, w2):
        if a == b:
            n += 1
        else:
            break
    return n


def is_suffix_only(ref_w: str, hyp_w: str, min_prefix: int = 4, min_ratio: float = 0.60) -> bool:
    if ref_w == hyp_w:
        return False
    max_len = max(len(ref_w), len(hyp_w))
    if max_len == 0:
        return False
    pref = shared_prefix_len(ref_w, hyp_w)
    return pref >= min_prefix and (pref / max_len) >= min_ratio


def is_looping_clip(norm_hyp: str) -> bool:
    """Detect repetition loops (consecutive repeated n-grams or extreme low diversity)."""
    words = norm_hyp.split()
    n = len(words)
    if n < 12:
        return False

    # Consecutive repeated n-grams (1 to 20 words)
    for k in range(1, min(21, n // 3 + 1)):
        min_reps = 4 if k == 1 else 3
        i = 0
        limit = n - k * min_reps
        while i <= limit:
            pat = words[i : i + k]
            reps = 1
            while i + (reps + 1) * k <= n and words[i + reps * k : i + (reps + 1) * k] == pat:
                reps += 1
            if reps >= min_reps:
                span = reps * k
                if span >= 8 or (i + span >= n - 2 and span >= 6):
                    return True
                i += span
            else:
                i += 1

    # Catch slightly drifting long loops (e.g. near 128-token ceiling)
    if n >= 30 and (len(set(words)) / n) < 0.35:
        return True

    return False


def load_manifest(path: Path) -> dict[str, str]:
    refs: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            refs[row["clip_id"]] = row["text"]
    return refs


def load_predictions(path: Path) -> dict[str, str]:
    preds: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            preds[row["clip_id"]] = row.get("hypothesis") or ""
    return preds


def analyze_run(
    refs: dict[str, str],
    preds: dict[str, str],
    min_prefix: int = 4,
    min_ratio: float = 0.60,
    suffix_weight: float = 0.5,
) -> dict:
    ref_words_total = 0
    suffix_err = 0
    number_err = 0
    other_sub = 0
    deletion = 0
    insertion = 0
    loop_clips = 0
    clip_count = 0

    for clip_id, ref_raw in refs.items():
        if clip_id not in preds:
            continue
        clip_count += 1
        ref_norm = normalize(ref_raw)
        hyp_norm = normalize(preds[clip_id])

        if is_looping_clip(hyp_norm):
            loop_clips += 1

        ref_tokens = ref_norm.split()
        hyp_tokens = hyp_norm.split()
        ref_words_total += len(ref_tokens)

        if not ref_tokens and not hyp_tokens:
            continue
        if not ref_tokens:
            for hw in hyp_tokens:
                if has_digit(hw):
                    number_err += 1
                else:
                    insertion += 1
            continue
        if not hyp_tokens:
            for rw in ref_tokens:
                if has_digit(rw):
                    number_err += 1
                else:
                    deletion += 1
            continue

        out = jiwer.process_words(ref_norm, hyp_norm)
        r_words = out.references[0]
        h_words = out.hypotheses[0]

        for chunk in out.alignments[0]:
            ctype = chunk.type
            if ctype == "equal":
                continue
            elif ctype == "substitute":
                r_slice = r_words[chunk.ref_start_idx : chunk.ref_end_idx]
                h_slice = h_words[chunk.hyp_start_idx : chunk.hyp_end_idx]
                paired = min(len(r_slice), len(h_slice))
                for rw, hw in zip(r_slice[:paired], h_slice[:paired]):
                    if has_digit(rw) or has_digit(hw):
                        number_err += 1
                    elif is_suffix_only(rw, hw, min_prefix=min_prefix, min_ratio=min_ratio):
                        suffix_err += 1
                    else:
                        other_sub += 1
                for rw in r_slice[paired:]:
                    if has_digit(rw):
                        number_err += 1
                    else:
                        deletion += 1
                for hw in h_slice[paired:]:
                    if has_digit(hw):
                        number_err += 1
                    else:
                        insertion += 1
            elif ctype == "delete":
                for rw in r_words[chunk.ref_start_idx : chunk.ref_end_idx]:
                    if has_digit(rw):
                        number_err += 1
                    else:
                        deletion += 1
            elif ctype == "insert":
                for hw in h_words[chunk.hyp_start_idx : chunk.hyp_end_idx]:
                    if has_digit(hw):
                        number_err += 1
                    else:
                        insertion += 1

    total_errors = suffix_err + number_err + other_sub + deletion + insertion
    wer = total_errors / ref_words_total if ref_words_total else 0.0
    weighted_errors = suffix_weight * suffix_err + (total_errors - suffix_err)
    weighted_wer = weighted_errors / ref_words_total if ref_words_total else 0.0
    suffix_share_of_sub = (
        suffix_err / (suffix_err + other_sub) if (suffix_err + other_sub) else 0.0
    )
    suffix_share_of_err = suffix_err / total_errors if total_errors else 0.0

    return {
        "clips": clip_count,
        "ref_words": ref_words_total,
        "total_errors": total_errors,
        "suffix_only": suffix_err,
        "number_err": number_err,
        "other_sub": other_sub,
        "deletion": deletion,
        "insertion": insertion,
        "suffix_pct_err": round(100.0 * suffix_share_of_err, 2),
        "suffix_pct_sub": round(100.0 * suffix_share_of_sub, 2),
        "wer": round(wer, 4),
        "weighted_wer": round(weighted_wer, 4),
        "wer_delta": round(wer - weighted_wer, 4),
        "loop_clips": loop_clips,
        "loop_pct": round(100.0 * loop_clips / clip_count, 2) if clip_count else 0.0,
    }


def sort_runs(run_names: list[str]) -> list[str]:
    baselines = [r for r in BASELINE_ORDER if r in run_names]
    others = sorted(r for r in run_names if r not in BASELINE_ORDER)
    return baselines + others


def discover_runs(pred_dir: Path, split: str) -> dict[str, Path]:
    found: dict[str, Path] = {}
    suffix = f"_{split}.csv"
    for p in sorted(pred_dir.glob(f"*{suffix}")):
        if "_sample" in p.name:
            continue
        run_name = p.name[: -len(suffix)]
        found[run_name] = p
    return found


def format_markdown_table(rows: list[dict]) -> str:
    header = (
        "| Run | WER | Weighted WER | Δ WER | Suffix-only | Suffix % (of err) | "
        "Number | Other Sub | Del | Ins | Loop Clips |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| `{r['run']}` | {r['wer']:.4f} | {r['weighted_wer']:.4f} | "
            f"-{r['wer_delta']:.4f} | {r['suffix_only']} | {r['suffix_pct_err']:.1f}% | "
            f"{r['number_err']} | {r['other_sub']} | {r['deletion']} | {r['insertion']} | "
            f"{r['loop_clips']} ({r['loop_pct']:.1f}%) |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ASR error types and weighted WER.")
    parser.add_argument("--dev-manifest", type=Path, default=Path("data/raw/manifests/dev.csv"))
    parser.add_argument("--test-manifest", type=Path, default=Path("data/raw/manifests/test.csv"))
    parser.add_argument("--pred-dir", type=Path, default=Path("results/predictions"))
    parser.add_argument("--out-csv", type=Path, default=Path("results/error_types.csv"))
    parser.add_argument("--min-prefix", type=int, default=4)
    parser.add_argument("--min-ratio", type=float, default=0.60)
    parser.add_argument("--suffix-weight", type=float, default=0.5)
    args = parser.parse_args()

    manifests = {
        "dev": load_manifest(args.dev_manifest),
        "test": load_manifest(args.test_manifest),
    }

    all_csv_rows: list[dict] = []

    # 1. Dev threshold sensitivity check (fixing rule on dev before looking at test)
    print("=== DEV THRESHOLD CALIBRATION (min_prefix=4) ===")
    dev_runs = discover_runs(args.pred_dir, "dev")
    cal_models = [m for m in ["hitz-ctc", "parakeet-v3-basque", "whisper-small-zeroshot", "ft-ours_seed3"] if m in dev_runs]
    for ratio in (0.50, 0.60, 0.70):
        parts = []
        for m in cal_models:
            res = analyze_run(manifests["dev"], load_predictions(dev_runs[m]), min_prefix=4, min_ratio=ratio)
            parts.append(f"{m}: {res['suffix_only']} ({res['suffix_pct_err']:.1f}%)")
        print(f"ratio>={ratio:.2f} -> " + " | ".join(parts))

    # 2. Full analysis per split
    for split in ("dev", "test"):
        runs = discover_runs(args.pred_dir, split)
        ordered = sort_runs(list(runs.keys()))
        split_rows = []
        for run in ordered:
            preds = load_predictions(runs[run])
            res = analyze_run(
                manifests[split],
                preds,
                min_prefix=args.min_prefix,
                min_ratio=args.min_ratio,
                suffix_weight=args.suffix_weight,
            )
            row = {"split": split, "run": run, **res}
            split_rows.append(row)
            all_csv_rows.append(row)

        print(f"\n=== {split.upper()} ERROR ANALYSIS ===")
        print(format_markdown_table(split_rows))

    # 3. Save CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if all_csv_rows:
        fieldnames = list(all_csv_rows[0].keys())
        with args.out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_csv_rows)
        print(f"\nSaved {args.out_csv} ({len(all_csv_rows)} rows).")


if __name__ == "__main__":
    main()