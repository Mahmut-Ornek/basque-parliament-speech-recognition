#!/usr/bin/env python
"""Timing pass: speed of every model on the same 100 dev clips.

Rules (one boundary for every model):
  * Colab T4, batch size 1.
  * The same 100 dev clips for every model, plus 5 warm-up clips that are not timed.
    Both sets are picked by a fixed hash of the clip id (seed 42), so every model
    gets exactly the same clips.
  * The clock runs from "audio is in memory as a float array" to "text is returned".
    Model loading, MP3 decoding and resampling are outside the timed part.
    Feature extraction and decoding are inside it.
  * The GPU is synchronised before and after every clip.

Run it from a clone of the repo (the manifests and audio come from the repo):

  python src/timing.py --family whisper --model openai/whisper-small --name whisper-small-zeroshot
  python src/timing.py --family whisper --model openai/whisper-large-v3 --name whisper-large-v3
  python src/timing.py --family ctc     --model gttsehu/wav2vec2-xls-r-300m-bp1-es_eu --name wav2vec2-bp1
  python src/timing.py --family ctc     --model stefan-it/wav2vec2-large-xlsr-53-basque --name wav2vec2-cv-eu
  python src/timing.py --family ctc     --model Ansu/mHubert-basque-ASR --name mhubert-basque
  # NeMo environment (requirements-nemo.txt):
  python src/timing.py --family nemo    --model HiTZ/stt_eu_conformer_ctc_large --name hitz-ctc
  python src/timing.py --family nemo    --model HiTZ/stt_eu_conformer_transducer_large --name hitz-transducer
  python src/timing.py --family nemo    --model itzune/parakeet-tdt-0.6b-v3-basque --name parakeet-v3-basque

Precision: the same as each model's accuracy run (default: float16 for Whisper and CTC models,
float32 for NeMo models). Whisper uses greedy decoding, language "eu" and max_new_tokens=128,
exactly as src/transcribe.py.

Output: results/timing/<name>_dev100.json. src/runtime_table.py reads these.

Other scripts can also import time_clips() and call it with their own transcribe function.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEV_MANIFEST = ROOT / "data" / "raw" / "manifests" / "dev.csv"
OUT_DIR = ROOT / "results" / "timing"

SEED = 42
N_WARMUP = 5
N_TIMED = 100
SAMPLE_RATE = 16000


# --------------------------------------------------------------------------- #
# Clip selection and loading
# --------------------------------------------------------------------------- #
def select_clips(manifest: Path = DEV_MANIFEST, n_warmup: int = N_WARMUP, n_timed: int = N_TIMED):
    """Return (warmup_rows, timed_rows). The same rows for every model, on every machine."""
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < n_warmup + n_timed:
        raise ValueError(f"{manifest} has only {len(rows)} clips")
    # Sorting by a hash of (seed, clip_id) gives a fixed pseudo-random order that does not
    # depend on the Python version or on the order of the rows in the file.
    rows.sort(key=lambda r: hashlib.sha256(f"{SEED}:{r['clip_id']}".encode()).hexdigest())
    chosen = rows[: n_warmup + n_timed]
    return chosen[:n_warmup], chosen[n_warmup:]


def load_audio(row: dict) -> np.ndarray:
    """Decode one clip to a mono float32 array at 16 kHz (not timed)."""
    import librosa

    path = Path(row["path"])
    if not path.is_absolute():
        path = ROOT / path
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


# --------------------------------------------------------------------------- #
# The timing function (importable)
# --------------------------------------------------------------------------- #
def _sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def time_clips(transcribe, clips, n_warmup: int = N_WARMUP) -> dict:
    """Time `transcribe(audio) -> str` clip by clip at batch size 1.

    clips: list of (clip_id, audio) with audio a float32 array at 16 kHz.
    The first n_warmup clips are run but not timed. Everything after them is timed.
    """
    if len(clips) <= n_warmup:
        raise ValueError("need more clips than warm-up clips")

    for _, audio in clips[:n_warmup]:
        transcribe(audio)
        _sync()

    ids, seconds, audio_seconds, hypotheses = [], [], [], []
    for clip_id, audio in clips[n_warmup:]:
        _sync()
        t0 = time.perf_counter()
        text = transcribe(audio)
        _sync()
        seconds.append(time.perf_counter() - t0)
        ids.append(clip_id)
        audio_seconds.append(len(audio) / SAMPLE_RATE)
        hypotheses.append(text)

    total_audio = float(sum(audio_seconds))
    total_compute = float(sum(seconds))
    return {
        "n_warmup": n_warmup,
        "n_timed": len(seconds),
        "total_audio_seconds": total_audio,
        "total_compute_seconds": total_compute,
        "rtfx": total_audio / total_compute,  # audio seconds per compute second; higher = faster
        "rtf": total_compute / total_audio,  # real-time factor, the inverse
        "median_seconds_per_clip": float(np.median(seconds)),
        "p95_seconds_per_clip": float(np.percentile(seconds, 95)),
        "n_empty_outputs": int(sum(1 for h in hypotheses if not h.strip())),
        "clip_ids": ids,
        "per_clip_seconds": seconds,
        "per_clip_audio_seconds": audio_seconds,
        "hypotheses": hypotheses,
    }


def hardware_info() -> dict:
    info = {"gpu": None, "torch": None, "cuda": None}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


# --------------------------------------------------------------------------- #
# Model loaders. Each returns (transcribe_fn, meta). Heavy imports stay inside.
# --------------------------------------------------------------------------- #
def _count_params(module) -> int | None:
    try:
        return int(sum(p.numel() for p in module.parameters()))
    except Exception:
        return None


def _device_and_dtype(precision: str):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and precision != "float32":
        print("WARNING: no GPU found, falling back to float32 on CPU. These numbers are not valid for the table.")
        precision = "float32"
    return device, getattr(torch, precision), precision


def load_whisper(args):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device, dtype, precision = _device_and_dtype(args.precision)
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    gen = dict(language="eu", task="transcribe", num_beams=1, max_new_tokens=128)  # same as src/transcribe.py

    def transcribe(audio):
        feats = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        feats = feats.to(device, dtype)
        with torch.inference_mode():
            ids = model.generate(feats, **gen)
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    return transcribe, {"parameters": _count_params(model), "precision": precision, "decoding": gen}


def load_ctc(args):
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    device, dtype, precision = _device_and_dtype(args.precision)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCTC.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    def transcribe(audio):
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        x = inputs.input_values.to(device, dtype)
        with torch.inference_mode():
            logits = model(x).logits
        return processor.batch_decode(logits.argmax(dim=-1))[0].strip()

    return transcribe, {"parameters": _count_params(model), "precision": precision, "decoding": "greedy CTC"}


def load_nemo(args):
    import nemo.collections.asr as nemo_asr

    import inspect

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model).to(device).eval()
    extra = {}
    if "use_lhotse" in inspect.signature(model.transcribe).parameters:
        extra["use_lhotse"] = False  # as in src/run_nemo.py: transcribe in-memory arrays directly

    def transcribe(audio):
        out = model.transcribe([audio], batch_size=1, verbose=False, **extra)
        if isinstance(out, tuple):  # older NeMo returns (texts, alt_texts) for transducers
            out = out[0]
        hyp = out[0]
        if isinstance(hyp, list):
            hyp = hyp[0]
        return (hyp.text if hasattr(hyp, "text") else str(hyp)).strip()

    return transcribe, {"parameters": _count_params(model), "precision": args.precision, "decoding": "NeMo default (greedy)"}


LOADERS = {"whisper": load_whisper, "ctc": load_ctc, "nemo": load_nemo}
DEFAULT_PRECISION = {"whisper": "float16", "ctc": "float16", "nemo": "float32"}


def model_revision(model_id: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id).sha
    except Exception:
        return None


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=LOADERS)
    ap.add_argument("--model", required=True, help="Hugging Face model id")
    ap.add_argument("--name", required=True, help="run name, e.g. whisper-large-v3")
    ap.add_argument("--precision", default=None, help="float16, float32 or bfloat16 (default depends on family)."
                    " Use the precision the accuracy run used.")
    ap.add_argument("--manifest", type=Path, default=DEV_MANIFEST)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.precision = args.precision or DEFAULT_PRECISION[args.family]

    warmup_rows, timed_rows = select_clips(args.manifest)
    clips = [(r["clip_id"], load_audio(r)) for r in warmup_rows + timed_rows]
    print(f"Loaded {len(clips)} clips ({N_WARMUP} warm-up + {N_TIMED} timed). Loading model {args.model} ...")

    transcribe, meta = LOADERS[args.family](args)
    result = time_clips(transcribe, clips, n_warmup=N_WARMUP)

    record = {
        "name": args.name,
        "model": args.model,
        "revision": model_revision(args.model),
        "family": args.family,
        "split": "dev",
        "clip_selection": f"{N_TIMED} dev clips (+{N_WARMUP} warm-up), sha256 order of '{SEED}:<clip_id>'",
        "batch_size": 1,
        "timed_from": "audio in memory",
        "timed_to": "text returned",
        "model_loading_excluded": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **hardware_info(),
        **meta,
        **result,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}_dev{N_TIMED}.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{args.name}: RTFx {record['rtfx']:.2f}, RTF {record['rtf']:.3f}, "
          f"median {record['median_seconds_per_clip']:.3f} s, p95 {record['p95_seconds_per_clip']:.3f} s, "
          f"empty outputs {record['n_empty_outputs']}/{N_TIMED}  ->  {out}")
    if record["n_empty_outputs"] > N_TIMED // 10:
        print("WARNING: many empty outputs. A model that returns nothing is fast for the wrong reason; check the setup.")


if __name__ == "__main__":
    main()
