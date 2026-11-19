import argparse
import csv
import inspect
import json
import os
import statistics
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from huggingface_hub import HfApi
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


def get_hf_revision(model_id: str) -> str:
    try:
        return HfApi().model_info(model_id).sha
    except Exception:
        return "unknown"


def transcribe_one(model, audio_input, supports_lhotse: bool):
    kwargs = {"batch_size": 1, "verbose": False}
    if supports_lhotse:
        kwargs["use_lhotse"] = False
    out = model.transcribe([audio_input], **kwargs)
    hyp = out[0] if isinstance(out, list) else out
    if isinstance(hyp, list):
        hyp = hyp[0]
    if hasattr(hyp, "text"):
        hyp = hyp.text
    return str(hyp)


def main():
    parser = argparse.ArgumentParser(description="Run NeMo ASR baselines for Basque.")
    parser.add_argument("--run", required=True, choices=list(MODELS.keys()), help="Run name")
    parser.add_argument("--split", required=True, choices=["dev", "test"], help="Dataset split")
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV")
    parser.add_argument("--limit", type=int, default=None, help="Limit clips for quick sanity check (dev only)")
    parser.add_argument("--outdir", default="results/predictions", help="Output directory")
    args = parser.parse_args()

    model_id = MODELS[args.run]
    revision = get_hf_revision(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_id} (rev: {revision[:8]}) on {device}...")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
    model = model.to(device)
    model.eval()

    sig = inspect.signature(model.transcribe)
    supports_lhotse = "use_lhotse" in sig.parameters

    rows = load_manifest(args.manifest, args.limit)
    print(f"Loaded {len(rows)} clips from {args.manifest}")

    # Test if model accepts in-memory numpy arrays (strict timing rule: audio loaded before timer)
    use_numpy = True
    first_wav, _ = librosa.load(rows[0]["path"], sr=16000, mono=True)
    try:
        _ = transcribe_one(model, first_wav, supports_lhotse)
    except Exception as e:
        print(f"Numpy input not supported ({e}), falling back to file paths.")
        use_numpy = False

    # Warm-up on first 5 clips
    print("Running 5 warm-up clips...")
    for r in rows[: min(5, len(rows))]:
        inp = librosa.load(r["path"], sr=16000, mono=True)[0] if use_numpy else r["path"]
        _ = transcribe_one(model, inp, supports_lhotse)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    predictions = []
    clip_times = []
    total_audio_sec = 0.0

    for idx, r in enumerate(rows, 1):
        audio_path = r["path"]
        duration = float(r.get("duration", 0.0))
        total_audio_sec += duration

        # Load audio into memory BEFORE starting timer
        inp = librosa.load(audio_path, sr=16000, mono=True)[0] if use_numpy else audio_path

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_t = time.perf_counter()
        hyp = transcribe_one(model, inp, supports_lhotse)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_t

        clip_times.append(elapsed)
        predictions.append({"clip_id": r["clip_id"], "hypothesis": hyp})

        if idx % 200 == 0 or idx == len(rows):
            print(f"[{args.run} | {args.split}] {idx}/{len(rows)} clips done")

    os.makedirs(args.outdir, exist_ok=True)
    suffix = f"{args.run}_{args.split}" if not args.limit else f"{args.run}_{args.split}_sample{args.limit}"
    csv_out = Path(args.outdir) / f"{suffix}.csv"
    json_out = Path(args.outdir) / f"{suffix}.json"

    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "hypothesis"])
        writer.writeheader()
        writer.writerows(predictions)

    total_compute_sec = sum(clip_times)
    rtfx = total_audio_sec / total_compute_sec if total_compute_sec > 0 else 0.0
    rtf = total_compute_sec / total_audio_sec if total_audio_sec > 0 else 0.0

    meta = {
        "run_name": args.run,
        "model_id": model_id,
        "revision": revision,
        "split": args.split,
        "num_clips": len(rows),
        "language_hint": "eu",
        "decoding": "greedy",
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "precision": "fp32",
        "batch_size": 1,
        "total_audio_seconds": round(total_audio_sec, 3),
        "total_compute_seconds": round(total_compute_sec, 3),
        "rtfx": round(rtfx, 3),
        "rtf": round(rtf, 5),
        "median_clip_seconds": round(statistics.median(clip_times), 5),
        "p95_clip_seconds": round(float(np.percentile(clip_times, 95)), 5),
    }

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {csv_out} and {json_out} (RTFx={meta['rtfx']})")


if __name__ == "__main__":
    main()