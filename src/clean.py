"""Remove bad clips from the raw training sample. Nobody in the group speaks Basque, so every
decision is made by a rule, and every threshold comes from the hand-checked `dev` split.

The script only removes training clips. It never rewrites a transcript and never changes `dev`
or `test`. The corpus quality score (PRR) is never read. Whisper's output is used as a warning
signal on training clips, not as ground truth.

Three steps, run from the repository root:

    python src/clean.py --judge dev      # Whisper large-v3 on dev (GPU), saved to data/cleaned/
    python src/clean.py --judge train    # the same on the training sample; resumes if stopped
    python src/clean.py                  # apply the rules and write the cleaned manifests

The third step needs no GPU: it reads the saved judge outputs, so it gives identical files on
every machine. Without the judge outputs it prints what rules 1-4 would remove and writes nothing.

Rules, in this order (a clip is logged under the first rule that removes it):
    1 leakage         the normalised transcript is also in dev or test, the audio overlaps them, the
                      speaker also speaks in dev or test, or the speaker is unknown (speaker_id 0)
    2 broken text     empty, or a character outside the Basque and Spanish alphabets
    3 timing          characters per second outside the range seen on dev, or a duration that
                      disagrees with the file name or with the decoded audio
    4 repeats         the same transcript more than 3 times, or audio that overlaps a kept clip
    5 judge           CER between Whisper's output and the transcript above a threshold
    6 wrong language  Whisper's language detection is confident the clip is Spanish

Rule 1 is leakage, not cleaning: it applies to every training condition. The raw sample without
the rule-1 clips is train_raw.csv, the training set of ft-raw.

Outputs in data/cleaned/: train_raw.csv, train_ours.csv, train_authors.csv, train_rawmatched.csv (equal hours),
removal_log.csv (clip_id, rule, value, threshold) and cleaning_report.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import jiwer

import make_sample

SEED = 42
JUDGE_MODEL = "openai/whisper-large-v3"
ALPHABET = set("abcdefghijklmnopqrstuvwxyzñçáéíóúü ")  # Basque and Spanish letters, and the space
MAX_COPIES = 3
NAME_TOLERANCE = 0.02  # seconds; file names and the duration column are rounded to 0.01
RATE_PERCENTILES = (1, 99)  # rule 3 bounds come from these dev percentiles, not from single clips
RATE_MARGIN = 0.10  # and are widened by 10%
AUDIO_MARGIN = 0.10  # seconds added to the largest duration gap seen on dev
MAX_FALSE_LANGUAGE_RATE = 0.01  # rule 6 may remove at most 1% of the (all-Basque) dev clips

MANIFEST_COLUMNS = list(make_sample.MANIFEST_COLUMNS)
JUDGE_COLUMNS = ["clip_id", "hypothesis", "es_prob", "eu_prob", "top_language", "audio_seconds"]
LOG_COLUMNS = ["clip_id", "rule", "value", "threshold"]
RULE_NUMBER = {
    "leakage_text": 1, "leakage_audio_overlap": 1, "held_out_speaker": 1, "unknown_speaker": 1,
    "broken_text": 2,
    "speaking_rate": 3, "duration_mismatch": 3,
    "repeated_text": 4, "repeated_audio_overlap": 4,
    "judge_cer": 5,
    "wrong_language": 6,
}


# --- small helpers -------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalise(text: str) -> str:
    """Normal form used for matching and for the judge CER (cleaning only; the scoring policy
    is in report/scoring_policy.md): Unicode NFC, lowercase, punctuation to space, single spaces."""
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return " ".join(text.split())


def judge_text(text: str) -> str:
    """Text compared by the judge rule: normalised, and only letters of the two alphabets kept.
    The references write numbers as words and Whisper often writes digits, so digits (and
    symbols such as % or €) are dropped instead of being counted as errors."""
    return " ".join("".join(c if c in ALPHABET else " " for c in normalise(text)).split())


def has_digit(text: str) -> bool:
    return any(c.isdigit() for c in text)


def clip_times(clip_id: str) -> tuple[str, float, float] | None:
    """('12-077_20220211_01', 382.04, 392.24) from '12-077_20220211_01_382.04_392.24'."""
    m = re.match(r"^(.+)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)$", clip_id)
    return (m.group(1), float(m.group(2)), float(m.group(3))) if m else None


def cer(reference: str, hypothesis: str) -> float:
    return jiwer.cer(reference, hypothesis) if hypothesis.strip() else 1.0


def hours(rows: list[dict]) -> float:
    return sum(float(r["duration"]) for r in rows) / 3600


# --- the rules: each returns {clip_id: (rule name, value, threshold)} ------------------------

def find_leakage(rows: list[dict], held_out: dict[str, list[dict]]) -> dict:
    """Train clips that share a sentence, audio or a speaker with dev or test. The teacher's rule:
    all clips of one speaker go to the same split. A clip without an identified speaker cannot be
    checked, so it is removed too."""
    texts, intervals, speakers = {}, defaultdict(list), {}
    for split, split_rows in held_out.items():
        for r in split_rows:
            texts.setdefault(normalise(r["text"]), split)
            if r["speaker_id"] != make_sample.UNKNOWN_SPEAKER:
                speakers.setdefault(r["speaker_id"], split)
            t = clip_times(r["clip_id"])
            if t:
                intervals[t[0]].append((t[1], t[2]))
    removed = {}
    for r in rows:
        text = normalise(r["text"])
        if text and text in texts:
            removed[r["clip_id"]] = ("leakage_text", texts[text], "")
            continue
        t = clip_times(r["clip_id"])
        if t:
            overlap = max((min(end, t[2]) - max(start, t[1]) for start, end in intervals.get(t[0], [])), default=0)
            if overlap > 0:
                removed[r["clip_id"]] = ("leakage_audio_overlap", f"{overlap:.2f}", "0")
                continue
        if r["speaker_id"] == make_sample.UNKNOWN_SPEAKER:
            removed[r["clip_id"]] = ("unknown_speaker", r["speaker_id"], "")
        elif r["speaker_id"] in speakers:
            removed[r["clip_id"]] = ("held_out_speaker", r["speaker_id"], speakers[r["speaker_id"]])
    return removed


def find_broken_text(rows: list[dict]) -> dict:
    removed = {}
    for r in rows:
        text = normalise(r["text"])
        outside = "".join(sorted(set(text) - ALPHABET))
        if not text:
            removed[r["clip_id"]] = ("broken_text", "empty", "")
        elif outside:
            removed[r["clip_id"]] = ("broken_text", outside, "")
    return removed


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile of an already sorted list (same as numpy's default)."""
    pos = (len(sorted_values) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def speaking_rate(row: dict) -> float:
    return len(normalise(row["text"])) / max(float(row["duration"]), 1e-6)


def timing_thresholds(dev: list[dict], judge_dev: dict | None) -> dict:
    """The range of characters per second on dev, and the largest gap between the duration
    column and the decoded audio on dev. Dev is hand-checked, so everything in it is valid;
    a training clip is removed only if it lies beyond that range by a margin."""
    rates = sorted(speaking_rate(r) for r in dev if normalise(r["text"]))
    low, high = (percentile(rates, q) for q in RATE_PERCENTILES)
    th = {"dev_rate_min": low, "dev_rate_max": high, "dev_audio_gap": None, "audio_gap": None}
    th["rate_min"], th["rate_max"] = low * (1 - RATE_MARGIN), high * (1 + RATE_MARGIN)
    if judge_dev is not None:
        th["dev_audio_gap"] = max(abs(float(judge_dev[r["clip_id"]]["audio_seconds"]) - float(r["duration"])) for r in dev)
        th["audio_gap"] = th["dev_audio_gap"] + AUDIO_MARGIN
    return th


def find_timing_outliers(rows: list[dict], th: dict, judge: dict | None) -> dict:
    removed = {}
    for r in rows:
        rate, duration = speaking_rate(r), float(r["duration"])
        t = clip_times(r["clip_id"])
        if rate < th["rate_min"] or rate > th["rate_max"]:
            removed[r["clip_id"]] = ("speaking_rate", f"{rate:.2f}", f"{th['rate_min']:.2f}-{th['rate_max']:.2f}")
        elif t and abs((t[2] - t[1]) - duration) > NAME_TOLERANCE:
            removed[r["clip_id"]] = ("duration_mismatch", f"{abs((t[2] - t[1]) - duration):.2f}", f"{NAME_TOLERANCE}")
        elif judge is not None and th["audio_gap"] is not None:
            gap = abs(float(judge[r["clip_id"]]["audio_seconds"]) - duration)
            if gap > th["audio_gap"]:
                removed[r["clip_id"]] = ("duration_mismatch", f"{gap:.2f}", f"{th['audio_gap']:.2f}")
    return removed


def find_repeats(rows: list[dict]) -> dict:
    """Keep the first MAX_COPIES clips of each transcript (in sample order). Then, inside each
    session, drop a clip whose time range overlaps a clip that starts earlier and is kept."""
    removed, seen = {}, Counter()
    totals = Counter(normalise(r["text"]) for r in rows)
    for r in rows:
        text = normalise(r["text"])
        seen[text] += 1
        if seen[text] > MAX_COPIES:
            removed[r["clip_id"]] = ("repeated_text", str(totals[text]), str(MAX_COPIES))
    sessions = defaultdict(list)
    for r in rows:
        t = clip_times(r["clip_id"])
        if t and r["clip_id"] not in removed:
            sessions[t[0]].append((t[1], t[2], r["clip_id"]))
    for clips in sessions.values():
        last_end = float("-inf")
        for start, end, clip_id in sorted(clips):
            if start < last_end - 1e-6:
                removed[clip_id] = ("repeated_audio_overlap", f"{last_end - start:.2f}", "0")
            else:
                last_end = end
    return removed


def find_judge_disagreement(rows: list[dict], judge: dict, threshold: float) -> dict:
    removed = {}
    for r in rows:
        value = cer(judge_text(r["text"]), judge_text(judge[r["clip_id"]]["hypothesis"]))
        if value > threshold:
            removed[r["clip_id"]] = ("judge_cer", f"{value:.3f}", f"{threshold:.2f}")
    return removed


def find_wrong_language(rows: list[dict], judge: dict, threshold: float | None) -> dict:
    if threshold is None:
        return {}
    return {
        r["clip_id"]: ("wrong_language", judge[r["clip_id"]]["es_prob"], f"{threshold:.2f}")
        for r in rows
        if float(judge[r["clip_id"]]["es_prob"]) >= threshold
    }


# --- thresholds from dev ---------------------------------------------------------------------

def calibrate_cer(dev: list[dict], judge_dev: dict) -> dict:
    """Whisper is weak on Basque, so its CER is high even on good clips. Break the hand-checked
    dev transcripts on purpose in three ways and pick the threshold that best separates broken
    from untouched clips: the one with the largest (catch rate - false removal rate)."""
    clips = [(judge_text(r["text"]), judge_text(judge_dev[r["clip_id"]]["hypothesis"])) for r in dev]
    clips = [(ref, hyp) for ref, hyp in clips if ref]
    refs = [ref for ref, _ in clips]
    n = len(clips)
    order = list(range(n))
    random.Random(SEED).shuffle(order)
    swapped = {order[k]: refs[order[(k + 1) % n]] for k in range(n)}  # another clip's transcript
    broken = {"swapped": [], "truncated": [], "shifted": []}
    for i, (ref, hyp) in enumerate(clips):
        words = ref.split()
        variants = {
            "swapped": swapped[i],
            "truncated": " ".join(words[: len(words) - max(1, len(words) // 3)]),  # last third cut off
            "shifted": refs[(i + 1) % n],  # the next clip's transcript
        }
        for kind, bad_ref in variants.items():
            if bad_ref and bad_ref != ref:
                broken[kind].append(cer(bad_ref, hyp))
    untouched = [cer(ref, hyp) for ref, hyp in clips]
    all_broken = [v for values in broken.values() for v in values]

    def rate(values: list[float], t: float) -> float:
        return sum(v > t for v in values) / len(values)

    best = max((k / 100 for k in range(5, 201)), key=lambda t: (rate(all_broken, t) - rate(untouched, t), t))
    return {
        "threshold": best,
        "dev_clips": n,
        "median_cer_untouched": sorted(untouched)[n // 2],
        "false_removal_rate": rate(untouched, best),
        "catch_rate": rate(all_broken, best),
        "catch_rate_by_kind": {kind: rate(values, best) for kind, values in broken.items()},
    }


def calibrate_language(dev: list[dict], judge_dev: dict) -> dict:
    """The lowest P(Spanish) threshold that would remove at most 1% of the dev clips, which are
    all Basque. If even 0.99 removes more, rule 6 is switched off."""
    probs = [float(judge_dev[r["clip_id"]]["es_prob"]) for r in dev]
    for k in range(50, 100):
        false_rate = sum(p >= k / 100 for p in probs) / len(probs)
        if false_rate <= MAX_FALSE_LANGUAGE_RATE:
            return {"threshold": k / 100, "false_removal_rate": false_rate}
    return {"threshold": None, "false_removal_rate": None}


# --- applying the rules ------------------------------------------------------------------------

def clean(train, dev, test, judge_train=None, judge_dev=None):
    """Return (kept rows, removal log, thresholds). Without judge outputs only rules 1-4 run."""
    have_judge = judge_train is not None and judge_dev is not None
    info = {"timing": timing_thresholds(dev, judge_dev)}
    steps = [
        lambda rows: find_leakage(rows, {"dev": dev, "test": test}),
        find_broken_text,
        lambda rows: find_timing_outliers(rows, info["timing"], judge_train),
        find_repeats,
    ]
    if have_judge:
        info["cer"] = calibrate_cer(dev, judge_dev)
        info["language"] = calibrate_language(dev, judge_dev)
        steps += [
            lambda rows: find_judge_disagreement(rows, judge_train, info["cer"]["threshold"]),
            lambda rows: find_wrong_language(rows, judge_train, info["language"]["threshold"]),
        ]
        # Evidence that the rules are safe: how many hand-checked dev clips they would remove.
        info["dev_would_remove"] = {
            "broken_text": len(find_broken_text(dev)),
            "judge_cer": len(find_judge_disagreement(dev, judge_dev, info["cer"]["threshold"])),
            "wrong_language": len(find_wrong_language(dev, judge_dev, info["language"]["threshold"])),
        }
    kept, log = list(train), []
    for number, step in enumerate(steps, start=1):
        removed = step(kept)
        if number == 5:  # how the judge rule treats clips with and without digits
            info["judge_by_digits"] = {
                group: (
                    sum(has_digit(judge_train[r["clip_id"]]["hypothesis"]) == group for r in kept),
                    sum(has_digit(judge_train[r["clip_id"]]["hypothesis"]) == group for r in kept if r["clip_id"] in removed),
                )
                for group in (True, False)
            }
        log += [dict(zip(LOG_COLUMNS, (r["clip_id"], *removed[r["clip_id"]]))) for r in kept if r["clip_id"] in removed]
        kept = [r for r in kept if r["clip_id"] not in removed]
    return kept, log, info


def random_subset(rows: list[dict], target_hours: float) -> list[dict]:
    """A seeded random subset of at most `target_hours`, returned in the original order."""
    shuffled = sorted(rows, key=lambda r: r["clip_id"])
    random.Random(SEED).shuffle(shuffled)
    chosen, total = set(), 0.0
    for r in shuffled:
        if total + float(r["duration"]) > target_hours * 3600 + 1e-6:
            break
        chosen.add(r["clip_id"])
        total += float(r["duration"])
    return [r for r in rows if r["clip_id"] in chosen]


def authors_clean_ids(tsv_path: Path) -> set[str]:
    """Clip ids of the corpus authors' own clean split (only the path column is used)."""
    ids = set()
    for r in make_sample.read_rows(tsv_path):
        p = PurePosixPath(r["path"])
        ids.update({p.stem, str(p.with_suffix("")).replace("/", "__")})
    return ids


def build_report(train, train_raw, kept, log, info, authors, outputs) -> str:
    duration = {r["clip_id"]: float(r["duration"]) for r in train}
    lines = ["# Cleaning report", "", "Written by `src/clean.py`. Do not edit by hand.", ""]
    lines += [
        f"Raw sample: {len(train)} clips, {hours(train):.2f} h.",
        f"After rule 1 (leakage, applied to every condition): {len(train_raw)} clips, {hours(train_raw):.2f} h "
        "(`train_raw.csv`, the training set of `ft-raw`). No training clip shares a speaker, a sentence or audio with dev or test.",
        f"Kept by all our rules: {len(kept)} clips, {hours(kept):.2f} h.", "",
    ]

    lines += ["## Thresholds (all chosen on dev)", ""]
    t = info["timing"]
    lines.append(
        f"- Rule 3, characters per second: the {RATE_PERCENTILES[0]}st and {RATE_PERCENTILES[1]}th percentiles on dev are "
        f"{t['dev_rate_min']:.2f} and {t['dev_rate_max']:.2f}. A training clip is removed below {t['rate_min']:.2f} or above "
        f"{t['rate_max']:.2f} ({RATE_MARGIN:.0%} beyond them)."
    )
    if t["audio_gap"] is not None:
        lines.append(
            f"- Rule 3, duration: the duration column and the decoded audio differ by at most {t['dev_audio_gap']:.2f} s on dev. "
            f"A training clip is removed above {t['audio_gap']:.2f} s, or if the duration disagrees with the times in its file name."
        )
    c, lang = info["cer"], info["language"]
    by_kind = ", ".join(f"{kind} {rate:.1%}" for kind, rate in c["catch_rate_by_kind"].items())
    lines.append(
        f"- Rule 5, judge CER threshold: {c['threshold']:.2f}. On {c['dev_clips']} dev clips Whisper's median CER on "
        f"untouched transcripts is {c['median_cer_untouched']:.3f}. The threshold catches {c['catch_rate']:.1%} of the "
        f"transcripts we broke on purpose ({by_kind}) and removes {c['false_removal_rate']:.1%} of the untouched ones."
    )
    lines.append(
        "  Digits and symbols are dropped from both sides before the CER is computed, because the references "
        "write numbers as words. The number words in the reference still count as deleted, so the table below checks "
        "whether the rule mostly removes clips with numbers."
    )
    if lang["threshold"] is None:
        lines.append("- Rule 6 is switched off: no threshold up to 0.99 keeps the false removals on dev at or below 1%.")
    else:
        lines.append(f"- Rule 6, P(Spanish) threshold: {lang['threshold']:.2f} (removes {lang['false_removal_rate']:.1%} of the dev clips, all Basque).")
    d = info["dev_would_remove"]
    lines += ["", f"Applied to dev (never done for real), the rules would remove: broken text {d['broken_text']}, "
              f"judge {d['judge_cer']}, wrong language {d['wrong_language']} of {len(outputs['dev'])} clips.", ""]

    lines += ["## Judge rule and numbers", "",
              "Clips that reached rule 5, split by whether Whisper's output contains a digit.", "",
              "| Whisper output | Clips judged | Removed by rule 5 | Rate |", "|---|---|---|---|"]
    for label, group in (("with digits", True), ("without digits", False)):
        judged, removed_n = info["judge_by_digits"][group]
        lines.append(f"| {label} | {judged} | {removed_n} | {removed_n / judged:.1%} |" if judged else f"| {label} | 0 | 0 | - |")
    lines += [""]

    lines += ["## Removed per rule", "", "| Rule | Reason | Clips | Hours |", "|---|---|---|---|"]
    counts = Counter(row["rule"] for row in log)
    secs = Counter()
    for row in log:
        secs[row["rule"]] += duration[row["clip_id"]]
    for name, number in RULE_NUMBER.items():
        lines.append(f"| {number} | {name} | {counts[name]} | {secs[name] / 3600:.2f} |")
    lines += [f"| | **total** | {len(log)} | {sum(secs.values()) / 3600:.2f} |", ""]

    kept_ids = {r["clip_id"] for r in kept}
    cells = Counter((r["clip_id"] in kept_ids, r["clip_id"] in authors) for r in train_raw)
    lines += [
        "## Our filter against the authors' filter", "",
        "Clips of `train_raw.csv` (rules 2-6 against the authors' `train_clean`). The authors' filter comes from an "
        "automatic score, not a human check: a reference, not the truth.", "",
        "| | Authors keep | Authors remove |", "|---|---|---|",
        f"| **We keep** | {cells[(True, True)]} | {cells[(True, False)]} |",
        f"| **We remove** | {cells[(False, True)]} | {cells[(False, False)]} |", "",
        "## Training sets (equal hours)", "", "| File | Clips | Hours |", "|---|---|---|",
    ]
    for name in ("train_ours", "train_authors", "train_rawmatched"):
        lines.append(f"| `{name}.csv` | {len(outputs[name])} | {hours(outputs[name]):.3f} |")
    lines += ["", f"`train_raw.csv` ({len(train_raw)} clips, {hours(train_raw):.3f} h) is not size-matched: "
              "`ft-raw` tests whether more but noisier hours win."]
    return "\n".join(lines) + "\n"


# --- the judge pass (GPU) ------------------------------------------------------------------------

def judge_path(out_root: Path, split: str) -> Path:
    return out_root / "data" / "cleaned" / f"judge_{split}.csv"


def load_judge(out_root: Path, split: str, rows: list[dict]) -> dict | None:
    """The saved judge output for a split, or None if it is missing or incomplete."""
    path = judge_path(out_root, split)
    if not path.is_file():
        print(f"{path.name} not found. Run: python src/clean.py --judge {split}")
        return None
    judge = {r["clip_id"]: r for r in read_csv(path)}
    missing = sum(r["clip_id"] not in judge for r in rows)
    if missing:
        print(f"{path.name}: {missing} clips missing. Run: python src/clean.py --judge {split} (it resumes)")
        return None
    return judge


def run_judge(split: str, rows: list[dict], out_root: Path, model_id: str, batch_size: int, limit: int | None) -> None:
    """Transcribe every clip with Whisper (language forced to Basque) and record the probability
    Whisper's language detection gives to Spanish and to Basque. Appends as it goes, so a
    stopped run continues where it was."""
    import librosa
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    path = judge_path(out_root, split)
    done = {r["clip_id"]: r for r in read_csv(path)} if path.is_file() else {}
    todo = [r for r in rows if r["clip_id"] not in done][:limit]
    print(f"{split}: {len(done)} clips already judged, {len(todo)} to do now")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device=device, dtype=dtype).eval()
    lang_to_id = model.generation_config.lang_to_id  # {"<|en|>": 50259, ...}
    lang_tokens = sorted(lang_to_id, key=lang_to_id.get)
    lang_ids = torch.tensor([lang_to_id[t] for t in lang_tokens], device=device)
    es, eu = lang_tokens.index("<|es|>"), lang_tokens.index("<|eu|>")

    import time

    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.is_file()
    started = time.perf_counter()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JUDGE_COLUMNS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        for i in range(0, len(todo), batch_size):
            batch = todo[i : i + batch_size]
            audio = [librosa.load(out_root / r["path"], sr=16000)[0] for r in batch]
            features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device=device, dtype=dtype)
            with torch.no_grad():
                start = torch.full((len(batch), 1), model.config.decoder_start_token_id, device=device)
                logits = model(input_features=features, decoder_input_ids=start).logits[:, -1].float()
                probs = logits[:, lang_ids].softmax(dim=-1).cpu()
                tokens = model.generate(features, language="eu", task="transcribe", max_new_tokens=128)
            texts = processor.batch_decode(tokens, skip_special_tokens=True)
            for r, wave, text, p in zip(batch, audio, texts, probs):
                row = {
                    "clip_id": r["clip_id"],
                    "hypothesis": " ".join(text.split()),
                    "es_prob": f"{p[es].item():.4f}",
                    "eu_prob": f"{p[eu].item():.4f}",
                    "top_language": lang_tokens[int(p.argmax())].strip("<|>"),
                    "audio_seconds": f"{len(wave) / 16000:.2f}",
                }
                writer.writerow(row)
                done[r["clip_id"]] = row
            f.flush()
            per_clip = (time.perf_counter() - started) / (i + len(batch))
            print(f"\r{split}: {i + len(batch)}/{len(todo)}, {per_clip:.2f} s per clip", end="", flush=True)
    print()
    if todo:
        per_clip = (time.perf_counter() - started) / len(todo)
        print(f"{per_clip:.2f} s per clip at batch size {batch_size}. At this speed: "
              f"dev (1,397 clips) {1397 * per_clip / 60:.0f} min, train (10,515 clips) {10515 * per_clip / 3600:.1f} h")

    if all(r["clip_id"] in done for r in rows):  # finished: rewrite in manifest order
        write_csv(path, JUDGE_COLUMNS, [done[r["clip_id"]] for r in rows])
        settings = {
            "model": model_id,
            "revision": getattr(model.config, "_commit_hash", None),
            "language": "eu", "task": "transcribe", "decoding": "greedy, max_new_tokens=128",
            "precision": str(dtype).replace("torch.", ""),
            "hardware": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "batch_size": batch_size, "clips": len(rows),
        }
        path.with_suffix(".json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"{split}: complete, wrote {path.name} and {path.with_suffix('.json').name}")


# --- command line ------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", choices=["dev", "train"], help="run the Whisper judge on this split (GPU)")
    ap.add_argument("--batch-size", type=int, default=8, help="judge batch size (default 8)")
    ap.add_argument("--limit", type=int, default=None, help="judge only this many clips now (quick test)")
    ap.add_argument("--model", default=JUDGE_MODEL, help=argparse.SUPPRESS)
    ap.add_argument("--out-root", default=str(Path(__file__).resolve().parents[1]), help="repository root")
    ap.add_argument("--cache-dir", default=None, help="HuggingFace cache folder for the downloads")
    ap.add_argument("--local-repo", default=None, help="read the corpus from a local folder (used by the tests)")
    args = ap.parse_args(argv)
    out_root = Path(args.out_root)
    manifests = out_root / "data" / "raw" / "manifests"
    train, dev, test = (read_csv(manifests / name) for name in ("train_sample.csv", "dev.csv", "test.csv"))

    if args.judge:
        run_judge(args.judge, {"dev": dev, "train": train}[args.judge], out_root, args.model, args.batch_size, args.limit)
        return

    judge_dev, judge_train = load_judge(out_root, "dev", dev), load_judge(out_root, "train", train)
    if judge_dev is None or judge_train is None:
        kept, log, _ = clean(train, dev, test)
        counts = Counter(row["rule"] for row in log)
        print(f"\nPreview of rules 1-4 on {len(train)} clips (no file written):")
        for name, number in RULE_NUMBER.items():
            if number <= 4:
                print(f"  rule {number} {name}: {counts[name]}")
        print(f"  kept so far: {len(kept)} clips, {hours(kept):.2f} h")
        return

    kept, log, info = clean(train, dev, test, judge_train, judge_dev)
    authors_ids = authors_clean_ids(make_sample.fetch("metadata/train_clean.tsv", args.local_repo, args.cache_dir))
    leaked = {row["clip_id"] for row in log if RULE_NUMBER[row["rule"]] == 1}
    train_raw = [r for r in train if r["clip_id"] not in leaked]
    authors = [r for r in train_raw if r["clip_id"] in authors_ids]

    # Size matching: the three training sets get the hours of the smaller of ours and the authors'.
    target = min(hours(kept), hours(authors))
    outputs = {
        "train_raw": train_raw,
        "train_ours": kept if hours(kept) <= target else random_subset(kept, target),
        "train_authors": authors if hours(authors) <= target else random_subset(authors, target),
        "train_rawmatched": random_subset(train_raw, target),
        "dev": dev,
    }
    cleaned = out_root / "data" / "cleaned"
    for name in ("train_raw", "train_ours", "train_authors", "train_rawmatched"):
        write_csv(cleaned / f"{name}.csv", MANIFEST_COLUMNS, outputs[name])
    write_csv(cleaned / "removal_log.csv", LOG_COLUMNS, log)
    report = build_report(train, train_raw, kept, log, info, {r["clip_id"] for r in authors}, outputs)
    (cleaned / "cleaning_report.md").write_text(report, encoding="utf-8", newline="\n")
    print(report)


if __name__ == "__main__":
    main()
