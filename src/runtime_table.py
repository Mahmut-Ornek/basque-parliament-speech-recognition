#!/usr/bin/env python
"""Write results/runtime_table.md from the timing and run JSON files.

Reads
  results/timing/*_dev100.json   speed numbers from src/timing.py (same 100 dev clips for every model)
  results/predictions/*.json     hardware of the accuracy runs (shown only where it differs from the timing pass)
  results/runs/*.json            training logs of the fine-tuned runs (training time)

Nobody edits results/runtime_table.md by hand: run  python src/runtime_table.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Display order. Any run found on disk but not listed here is appended at the end of its part.
BASELINE_ORDER = [
    "whisper-small-zeroshot", "whisper-large-v3", "hitz-ctc", "hitz-transducer",
    "parakeet-v3-basque", "wav2vec2-bp1", "wav2vec2-cv-eu", "mhubert-basque",
]
# Key names tried, in order, because the run scripts may name them differently.
TRAIN_TIME_KEYS = ["training_time_seconds", "train_time_seconds", "training_seconds", "train_seconds",
                   "training_time_s", "training_time", "train_time"]
HARDWARE_KEYS = ["hardware", "gpu", "gpu_name"]


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"skipping {path}: {e}")
        return None


def first_key(d: dict, keys: list[str]):
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    return None


def fmt(x, nd=2, dash="n/a"):
    return dash if x is None else f"{x:.{nd}f}"


def run_name_from_file(path: Path) -> str:
    """whisper-large-v3_test.json -> whisper-large-v3"""
    stem = path.stem
    for suffix in ("_test", "_dev"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def collect_timing() -> dict[str, dict]:
    out = {}
    for p in sorted((RESULTS / "timing").glob("*_dev100.json")):
        rec = load_json(p)
        if rec:
            out[rec.get("name", p.stem.removesuffix("_dev100"))] = rec
    return out


def collect_test_hardware() -> dict[str, str]:
    hw = {}
    for p in sorted((RESULTS / "predictions").glob("*_test.json")):
        rec = load_json(p)
        if rec:
            value = first_key(rec, HARDWARE_KEYS)
            if value:
                hw[run_name_from_file(p)] = str(value)
    return hw


def collect_training() -> list[dict]:
    rows = []
    for p in sorted((RESULTS / "runs").glob("*.json")):
        rec = load_json(p)
        if not rec:
            continue
        seconds = first_key(rec, TRAIN_TIME_KEYS)
        if seconds is None and isinstance(rec.get("total_minutes"), (int, float)):  # src/finetune_whisper.py
            seconds = rec["total_minutes"] * 60
        rows.append({
            "run": p.stem,
            "seconds": float(seconds) if isinstance(seconds, (int, float)) else None,
            "gpu": first_key(rec, HARDWARE_KEYS),
            "best_step": rec.get("best_step"),
        })
    return rows


def ordered(names: list[str]) -> list[str]:
    known = [n for n in BASELINE_ORDER if n in names]
    return known + sorted(n for n in names if n not in BASELINE_ORDER)


def main() -> None:
    timing = collect_timing()
    test_hw = collect_test_hardware()
    training = collect_training()

    lines = [
        "# Hardware and runtime",
        "",
        "<!-- Written by src/runtime_table.py. Do not edit by hand. -->",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC.",
        "",
        "## Inference speed (pretrained models)",
        "",
        "Every model processed the same 100 dev clips at batch size 1, after 5 warm-up clips. "
        "The clock runs from \"audio is in memory\" to \"text is returned\"; model loading is excluded. "
        "RTFx = audio seconds / compute seconds (higher is faster). "
        "RTF = compute seconds / audio seconds (the inverse; lower is faster). "
        "Medians and 95th percentiles are seconds per clip.",
        "",
        "| Run | Model | Params (M) | Hardware | Precision | RTFx | RTF | Median s/clip | P95 s/clip |",
        "|---|---|---:|---|---|---:|---:|---:|---:|",
    ]
    if not timing:
        lines.append("| _no timing files found in results/timing/_ | | | | | | | | |")

    notes = []
    for name in ordered(list(timing)):
        r = timing[name]
        params = r.get("parameters")
        lines.append(
            f"| `{name}` | {r.get('model', 'n/a')} | {fmt(params / 1e6, 0) if params else 'n/a'} "
            f"| {r.get('gpu') or 'n/a'} | {r.get('precision') or 'n/a'} | {fmt(r.get('rtfx'))} "
            f"| {fmt(r.get('rtf'), 3)} | {fmt(r.get('median_seconds_per_clip'), 3)} "
            f"| {fmt(r.get('p95_seconds_per_clip'), 3)} |"
        )
        hw = test_hw.get(name)
        if hw and r.get("gpu") and hw.strip().lower() != str(r["gpu"]).strip().lower():
            notes.append(f"- `{name}`: the dev/test accuracy run used **{hw}**; the timing above is from {r['gpu']}.")
        if r.get("n_empty_outputs"):
            notes.append(f"- `{name}`: {r['n_empty_outputs']} of {r.get('n_timed', '?')} timed clips returned empty text.")
        if r.get("gpu") and "T4" not in str(r["gpu"]):
            notes.append(f"- `{name}`: timed on {r['gpu']}, not a Colab T4, so its speed is not comparable with the other rows.")

    missing = [n for n in BASELINE_ORDER if n not in timing]
    if missing:
        notes.append("- No timing yet for: " + ", ".join(f"`{n}`" for n in missing) + ".")
    if notes:
        lines += ["", *notes]

    lines += [
        "",
        "Speed measured here is for batch size 1 on one GPU model. It says how the models compare under "
        "identical conditions, not how fast any of them can go with batching.",
        "",
        "## Training time (our fine-tuned runs)",
        "",
        "| Run | GPU | Best step | Training time (min) |",
        "|---|---|---:|---:|",
    ]
    if not training:
        lines.append("| _no training logs found in results/runs/_ | | | |")
    for t in training:
        minutes = None if t["seconds"] is None else t["seconds"] / 60
        lines.append(f"| `{t['run']}` | {t['gpu'] or 'n/a'} | {t['best_step'] if t['best_step'] is not None else 'n/a'} "
                     f"| {fmt(minutes, 1)} |")
    lines += [
        "",
        "The fine-tuned runs were not timed separately: they have the architecture and decoding settings of "
        "`whisper-small-zeroshot`, but zero-shot Whisper-small loops on some clips (see `report/error_analysis.md`), "
        "which makes its row slower than the fine-tuned models would be.",
        "",
    ]

    out = RESULTS / "runtime_table.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(timing)} timed models, {len(training)} training logs)")


if __name__ == "__main__":
    main()
