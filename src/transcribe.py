"""Transcribe one split with a pretrained or fine-tuned model, without any training.

Writes the raw model output, unchanged, to results/predictions/<name>_<split>.csv
(clip_id, hypothesis) and the settings, hardware and timing to results/predictions/<name>_<split>.json.
Scoring is done by the scoring script, not here; the WER printed at the end is only a quick check
that the run worked.

Run from the repository root:

    python src/transcribe.py --family whisper --model openai/whisper-small --name whisper-small-zeroshot --split dev
    python src/transcribe.py --family ctc --model stefan-it/wav2vec2-large-xlsr-53-basque --name wav2vec2-cv-eu --split dev
    python src/transcribe.py --family ctc --model gttsehu/wav2vec2-xls-r-300m-bp1-es_eu --name wav2vec2-bp1 --split dev
    python src/transcribe.py --family whisper --model models/ft-raw_seed1/best --name ft-raw_seed1 --split test

`--split` has no default on purpose. `test` is used once per model, for the final score; quick
checks and every decision use `dev`. An existing prediction file is never overwritten without --force.

`--family ctc` loads any CTC model (wav2vec 2.0, mHuBERT, ...) with AutoModelForCTC. Every Whisper run
uses greedy decoding, language forced to Basque, and max_new_tokens=128.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import jiwer

from clean import normalise, read_csv, write_csv

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000
MAX_NEW_TOKENS = 128  # clips are at most 10.4 s and 208 characters; the judge used the same limit
PREDICTION_COLUMNS = ["clip_id", "hypothesis"]
MANIFESTS = {"dev": "data/raw/manifests/dev.csv", "test": "data/raw/manifests/test.csv"}


def load_audio(paths: list[Path], workers: int = 8) -> list:
    """Decode MP3 clips to 16 kHz mono float arrays, several at a time."""
    import librosa

    def one(path: Path):
        return librosa.load(path, sr=SAMPLE_RATE, mono=True)[0]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, paths))


def model_revision(model_id: str) -> str | None:
    """Commit hash of a Hugging Face model, or None for a local folder."""
    if Path(model_id).exists():
        return None
    from huggingface_hub import model_info

    return model_info(model_id).sha


def hardware() -> str:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"cpu ({platform.processor()})"


def versions() -> dict:
    import torch
    import transformers

    return {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__}


class WhisperTranscriber:
    family = "whisper"

    def __init__(self, model_id: str, device: str, dtype, model=None, processor=None):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch, self.device, self.dtype = torch, device, dtype
        self.processor = processor or WhisperProcessor.from_pretrained(model_id)
        self.model = model or WhisperForConditionalGeneration.from_pretrained(model_id).to(device=device, dtype=dtype)
        self.decoding = f"greedy, language=eu (forced), task=transcribe, max_new_tokens={MAX_NEW_TOKENS}"

    def __call__(self, audio: list) -> tuple[list[str], int]:
        torch = self.torch
        features = self.processor.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        features = features.to(device=self.device, dtype=next(self.model.parameters()).dtype)
        with torch.no_grad(), torch.autocast(self.device, dtype=self.dtype, enabled=self.device == "cuda"):
            tokens = self.model.generate(features, language="eu", task="transcribe", max_new_tokens=MAX_NEW_TOKENS)
        texts = self.processor.batch_decode(tokens, skip_special_tokens=True)
        # A clip that used the whole token budget was cut off (usually a repetition loop).
        eos = self.model.generation_config.eos_token_id
        cut = sum(int(eos not in row.tolist()) for row in tokens)
        return texts, cut


class CTCTranscriber:
    family = "ctc"

    def __init__(self, model_id: str, device: str, dtype):
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        self.torch, self.device, self.dtype = torch, device, dtype
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCTC.from_pretrained(model_id).to(device=device, dtype=dtype).eval()
        self.decoding = "greedy CTC (argmax per frame), no language model"

    def __call__(self, audio: list) -> tuple[list[str], int]:
        torch = self.torch
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        values = inputs.input_values.to(device=self.device, dtype=self.dtype)
        mask = inputs.get("attention_mask")
        with torch.no_grad():
            logits = self.model(values, attention_mask=None if mask is None else mask.to(self.device)).logits
        return self.processor.batch_decode(logits.argmax(dim=-1)), 0


def transcribe(transcriber, rows: list[dict], batch_size: int, label: str = "") -> tuple[list[dict], dict]:
    """Run the transcriber over the manifest rows in order. Returns the predictions and timing."""
    out, cut, decode_s, model_s = [], 0, 0.0, 0.0
    started = time.perf_counter()
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        t0 = time.perf_counter()
        audio = load_audio([ROOT / r["path"] for r in batch])
        t1 = time.perf_counter()
        texts, n_cut = transcriber(audio)
        if transcriber.device == "cuda":
            transcriber.torch.cuda.synchronize()
        t2 = time.perf_counter()
        decode_s, model_s, cut = decode_s + t1 - t0, model_s + t2 - t1, cut + n_cut
        out += [{"clip_id": r["clip_id"], "hypothesis": " ".join(t.split())} for r, t in zip(batch, texts)]
        print(f"\r{label}{len(out)}/{len(rows)} clips", end="", flush=True)
    print()
    total = time.perf_counter() - started
    audio_s = sum(float(r["duration"]) for r in rows)
    timing = {
        "wall_seconds": round(total, 1),
        "audio_decoding_seconds": round(decode_s, 1),
        "model_seconds": round(model_s, 1),
        "seconds_per_clip": round(total / max(len(rows), 1), 4),
        "real_time_factor": round(total / audio_s, 4) if audio_s else None,
        "clips_hitting_token_limit": cut,
    }
    return out, timing


def quick_wer(rows: list[dict], predictions: list[dict]) -> float:
    """WER after clean.normalise (lowercase, no punctuation). A sanity check only: the official
    scores use the scoring policy in report/ and are written by the scoring script."""
    hyp = {p["clip_id"]: normalise(p["hypothesis"]) for p in predictions}
    return jiwer.wer([normalise(r["text"]) for r in rows], [hyp[r["clip_id"]] for r in rows])


def main(argv: list[str] | None = None) -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=["whisper", "ctc"], required=True, help="whisper, or any CTC model (AutoModelForCTC)")
    ap.add_argument("--model", required=True, help="Hugging Face model id or a local checkpoint folder")
    ap.add_argument("--name", required=True, help="model name, e.g. whisper-small-zeroshot; files are results/predictions/<name>_<split>.csv/.json")
    ap.add_argument("--split", choices=sorted(MANIFESTS), required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips (quick test; writes to --out-dir)")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "predictions"))
    ap.add_argument("--force", action="store_true", help="overwrite existing prediction files")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    csv_path, json_path = out_dir / f"{args.name}_{args.split}.csv", out_dir / f"{args.name}_{args.split}.json"
    if csv_path.exists() and not args.force:
        raise SystemExit(f"{csv_path} exists. Each model is scored on test once; pass --force to overwrite.")
    if args.split == "test" and args.limit:
        raise SystemExit("Quick checks run on dev only: --limit is not allowed with --split test.")

    rows = read_csv(ROOT / MANIFESTS[args.split])[: args.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cpu":
        print("Warning: no CUDA GPU found, running on CPU (slow)")

    load_started = time.perf_counter()
    cls = WhisperTranscriber if args.family == "whisper" else CTCTranscriber
    transcriber = cls(args.model, device, dtype)
    load_seconds = time.perf_counter() - load_started

    predictions, timing = transcribe(transcriber, rows, args.batch_size)
    write_csv(csv_path, PREDICTION_COLUMNS, predictions)
    settings = {
        "name": args.name,
        "model": args.model,
        "revision": model_revision(args.model),
        "family": args.family,
        "training": "none (pretrained model as published)" if not Path(args.model).exists() else "fine-tuned checkpoint",
        "split": args.split,
        "manifest": MANIFESTS[args.split],
        "clips": len(rows),
        "audio_hours": round(sum(float(r["duration"]) for r in rows) / 3600, 3),
        "decoding": transcriber.decoding,
        "max_new_tokens": MAX_NEW_TOKENS if args.family == "whisper" else None,
        "precision": str(dtype).replace("torch.", ""),
        "batch_size": args.batch_size,
        "hardware": hardware(),
        "versions": versions(),
        "model_load_seconds": round(load_seconds, 1),
        **timing,
        "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    json_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {csv_path.relative_to(ROOT) if csv_path.is_relative_to(ROOT) else csv_path} and the .json")
    print(f"{timing['wall_seconds']:.0f} s ({timing['seconds_per_clip']:.3f} s per clip, "
          f"real-time factor {timing['real_time_factor']}); {timing['clips_hitting_token_limit']} clips hit the token limit")
    print(f"Quick check, not the official score: WER {quick_wer(rows, predictions):.3f}")


if __name__ == "__main__":
    main()
