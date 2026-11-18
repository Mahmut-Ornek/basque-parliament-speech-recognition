import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import nemo.collections.asr as nemo_asr

MODELS = {
    "hitz-ctc": "HiTZ/stt_eu_conformer_ctc_large",
    "hitz-transducer": "HiTZ/stt_eu_conformer_transducer_large",
    "parakeet-v3-basque": "itzune/parakeet-tdt-0.6b-v3-basque",
}


def load_manifest(manifest_path: str, limit: int = None):
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run NeMo ASR baselines for Basque.")
    parser.add_argument("--run", required=True, choices=list(MODELS.keys()), help="Run name")
    parser.add_argument("--split", required=True, choices=["dev", "test"], help="Dataset split")
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV")
    parser.add_argument("--limit", type=int, default=None, help="Limit clips for quick sanity check")
    parser.add_argument("--outdir", default="results/predictions", help="Output directory")
    args = parser.parse_args()

    model_id = MODELS[args.run]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_id} on {device}...")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
    model = model.to(device)
    model.eval()

    rows = load_manifest(args.manifest, args.limit)
    print(f"Loaded {len(rows)} clips from {args.manifest}")

    # Warm-up on first 5 clips (per timing rule)
    warmup_rows = rows[: min(5, len(rows))]
    for r in warmup_rows:
        _ = model.transcribe([r["path"]], batch_size=1, verbose=False)

    predictions = []
    clip_times = []
    total_audio_sec = 0.0

    for r in rows:
        audio_path = r["path"]  # Always read path column directly
        duration = float(r.get("duration", 0.0))
        total_audio_sec += duration

        start_t = time.perf_counter()
        out = model.transcribe([audio_path], batch_size=1, verbose=False)
        elapsed = time.perf_counter() - start_t
        clip_times.append(elapsed)

        # Handle NeMo return type (list of strings or list of Hypothesis objects)
        hyp = out[0] if isinstance(out, list) else out
        if isinstance(hyp, list):
            hyp = hyp[0]
        if hasattr(hyp, "text"):
            hyp = hyp.text

        predictions.append({"clip_id": r["clip_id"], "hypothesis": str(hyp)})

    os.makedirs(args.outdir, exist_ok=True)
    suffix = f"{args.run}_{args.split}" if not args.limit else f"{args.run}_{args.split}_sample{args.limit}"
    csv_out = Path(args.outdir) / f"{suffix}.csv"
    json_out = Path(args.outdir) / f"{suffix}.json"

    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "hypothesis"])
        writer.writeheader()
        writer.writerows(predictions)

    total_compute_sec = sum(clip_times)
    meta = {
        "run_name": args.run,
        "model_id": model_id,
        "split": args.split,
        "num_clips": len(rows),
        "language_hint": "eu",
        "decoding": "greedy",
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "precision": "fp32",
        "batch_size": 1,
        "total_audio_seconds": round(total_audio_sec, 3),
        "total_compute_seconds": round(total_compute_sec, 3),
        "rtfx": round(total_audio_sec / total_compute_sec, 3) if total_compute_sec > 0 else None,
    }

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {csv_out} and {json_out}")


if __name__ == "__main__":
    main()