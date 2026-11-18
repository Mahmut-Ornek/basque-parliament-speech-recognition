"""Fine-tune Whisper-small on one training manifest, choosing the checkpoint on `dev`.

One run = one condition and one seed. Every run (all conditions, all seeds) uses the same frozen
settings; only --condition and --seed change:

    python src/finetune_whisper.py --condition raw        --seed 1    # ft-raw_seed1
    python src/finetune_whisper.py --condition ours       --seed 1    # ft-ours_seed1
    python src/finetune_whisper.py --condition authors    --seed 1    # ft-authors_seed1
    python src/finetune_whisper.py --condition rawmatched --seed 1    # ft-rawmatched_seed1

Condition <c> trains on data/cleaned/train_<c>.csv; the run is called ft-<c>_seed<N>. Every
--eval-every optimizer steps the model transcribes all of `dev`; the dev WER and the evaluation time
are logged. The checkpoint with the lowest dev WER is kept in models/ft-<c>_seed<N>/best (not
committed). The log, results/runs/ft-<c>_seed<N>.json, is rewritten after every evaluation, so a
stopped run still has its curve.

This script never reads `test`. The best checkpoint is scored on test once, at the final scoring:

    python src/transcribe.py --family whisper --model models/ft-raw_seed1/best --name ft-raw_seed1 --split test
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from clean import read_csv
from transcribe import MAX_NEW_TOKENS, ROOT, MANIFESTS, WhisperTranscriber, hardware, load_audio, model_revision, quick_wer, transcribe, versions

BASE_MODEL = "openai/whisper-small"
CONDITIONS = ("raw", "ours", "authors", "rawmatched")  # data/cleaned/train_<condition>.csv


class Clips(Dataset):
    """Training clips as (log-Mel features, label token ids). Audio is decoded in the workers."""

    def __init__(self, rows: list[dict], feature_extractor, tokenizer):
        self.rows, self.fe, self.tok = rows, feature_extractor, tokenizer

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        audio = load_audio([ROOT / r["path"]], workers=1)[0]
        features = self.fe(audio, sampling_rate=16000, return_tensors="np").input_features[0]
        return features.astype(np.float32), self.tok(r["text"]).input_ids


class Collate:
    def __init__(self, decoder_start_token_id: int):
        self.start = decoder_start_token_id

    def __call__(self, batch):
        features = torch.from_numpy(np.stack([f for f, _ in batch]))
        longest = max(len(ids) for _, ids in batch)
        labels = torch.full((len(batch), longest), -100, dtype=torch.long)
        for i, (_, ids) in enumerate(batch):
            labels[i, : len(ids)] = torch.tensor(ids)
        # The model puts <|startoftranscript|> in front of the decoder input itself.
        if (labels[:, 0] == self.start).all():
            labels = labels[:, 1:]
        return features, labels


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batches_forever(loader: DataLoader):
    """Epoch after epoch; the seeded generator of the loader gives each epoch its own order."""
    while True:
        yield from loader


def evaluate(model, processor, dev: list[dict], batch_size: int) -> tuple[float, dict]:
    model.eval()
    transcriber = WhisperTranscriber(BASE_MODEL, "cuda", torch.float16, model=model, processor=processor)
    predictions, timing = transcribe(transcriber, dev, batch_size, label="  dev: ")
    model.train()
    return quick_wer(dev, predictions), timing


def main(argv: list[str] | None = None) -> None:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", choices=CONDITIONS, required=True, help="trains on data/cleaned/train_<condition>.csv")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=2500, help="optimizer steps (default 2500)")
    ap.add_argument("--eval-every", type=int, default=500, help="dev evaluation interval in steps (default 500)")
    ap.add_argument("--batch-size", type=int, default=8, help="clips per forward pass (default 8)")
    ap.add_argument("--grad-accum", type=int, default=2, help="forward passes per optimizer step (default 2: 16 clips per step)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=250, help="linear warm-up steps, then linear decay to 0")
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4, help="data loader processes")
    ap.add_argument("--dev-limit", type=int, default=None, help="evaluate on the first N dev clips only (quick test)")
    ap.add_argument("--runs-dir", default=str(ROOT / "results" / "runs"))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing log with this name")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("Fine-tuning needs a CUDA GPU.")
    name = f"ft-{args.condition}_seed{args.seed}"
    log_path = Path(args.runs_dir) / f"{name}.json"
    best_dir = Path(args.models_dir) / name / "best"
    if log_path.exists() and not args.force:
        raise SystemExit(f"{log_path} exists; pass --force to overwrite it.")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    train_path = ROOT / "data" / "cleaned" / f"train_{args.condition}.csv"
    train = read_csv(train_path)
    dev = read_csv(ROOT / MANIFESTS["dev"])[: args.dev_limit]
    clips_per_step = args.batch_size * args.grad_accum
    train_hours = sum(float(r["duration"]) for r in train) / 3600
    print(f"{name}: {len(train)} clips ({train_hours:.2f} h), {args.steps} steps of {clips_per_step} clips "
          f"= {args.steps * clips_per_step / len(train):.1f} epochs; dev {len(dev)} clips every {args.eval_every} steps")

    seed_everything(args.seed)
    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="basque", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL).to("cuda")
    model.generation_config.language, model.generation_config.task = "eu", "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        Clips(train, processor.feature_extractor, processor.tokenizer),
        batch_size=args.batch_size, shuffle=True, generator=generator, drop_last=True,
        collate_fn=Collate(model.config.decoder_start_token_id),
        num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup, args.steps)
    scaler = torch.amp.GradScaler("cuda")

    log = {
        "name": name,
        "condition": f"ft-{args.condition}",
        "base_model": BASE_MODEL,
        "base_revision": model_revision(BASE_MODEL),
        "train_manifest": train_path.relative_to(ROOT).as_posix(),
        "train_clips": len(train),
        "train_hours": round(train_hours, 3),
        "dev_clips": len(dev),
        "seed": args.seed,
        "settings": {
            "steps": args.steps, "eval_every": args.eval_every, "batch_size": args.batch_size,
            "grad_accum": args.grad_accum, "clips_per_step": clips_per_step,
            "epochs": round(args.steps * clips_per_step / len(train), 2),
            "optimizer": "AdamW (torch defaults: betas 0.9/0.999, eps 1e-8, weight decay 0.01)",
            "lr": args.lr, "schedule": f"linear warm-up {args.warmup} steps, linear decay to 0",
            "max_grad_norm": 1.0, "precision": "fp16 autocast, fp32 weights", "gradient_checkpointing": True,
            "language": "eu", "task": "transcribe",
            "dev_decoding": f"greedy, language=eu (forced), max_new_tokens={MAX_NEW_TOKENS}",
            "max_new_tokens": MAX_NEW_TOKENS,
            "dev_wer": "jiwer WER after clean.normalise (lowercase, no punctuation); used only to pick the checkpoint",
        },
        "hardware": hardware(),
        "versions": versions(),
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_loss": [],
        "evaluations": [],
        "best_step": None,
        "best_dev_wer": None,
        "checkpoint": best_dir.relative_to(ROOT).as_posix() if best_dir.is_relative_to(ROOT) else str(best_dir),
        "status": "running",
    }

    def save_log() -> None:
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    save_log()
    started = time.perf_counter()
    batches = batches_forever(loader)
    window, window_start = [], time.perf_counter()
    for step in range(1, args.steps + 1):
        step_loss = 0.0
        for _ in range(args.grad_accum):
            features, labels = next(batches)
            features, labels = features.to("cuda", non_blocking=True), labels.to("cuda", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_features=features, labels=labels).loss / args.grad_accum
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        window.append(step_loss)

        if step % 50 == 0 or step == args.steps:
            per_step = (time.perf_counter() - window_start) / len(window)
            log["train_loss"].append({"step": step, "loss": round(sum(window) / len(window), 4),
                                      "lr": scheduler.get_last_lr()[0], "seconds_per_step": round(per_step, 3)})
            left = (args.steps - step) * per_step / 60
            print(f"step {step}/{args.steps}  loss {sum(window) / len(window):.4f}  {per_step:.2f} s/step  "
                  f"~{left:.0f} min of training left  GPU mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
            window, window_start = [], time.perf_counter()

        if step % args.eval_every == 0 or step == args.steps:
            wer, timing = evaluate(model, processor, dev, args.eval_batch_size)
            entry = {"step": step, "epoch": round(step * clips_per_step / len(train), 2), "dev_wer": round(wer, 4),
                     "eval_seconds": timing["wall_seconds"], "clips_hitting_token_limit": timing["clips_hitting_token_limit"],
                     "elapsed_minutes": round((time.perf_counter() - started) / 60, 1)}
            log["evaluations"].append(entry)
            improved = log["best_dev_wer"] is None or wer < log["best_dev_wer"]
            if improved:
                log["best_step"], log["best_dev_wer"] = step, round(wer, 4)
                model.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)
                model.generation_config.save_pretrained(best_dir)
            print(f"  step {step}: dev WER {wer:.4f}{'  (best so far, saved)' if improved else ''}")
            save_log()
            window_start = time.perf_counter()  # evaluation time is not training time

    log["status"] = "finished"
    log["finished_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log["total_minutes"] = round((time.perf_counter() - started) / 60, 1)
    log["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    save_log()
    print("\nDev WER per evaluation:")
    for e in log["evaluations"]:
        print(f"  step {e['step']:>5}  epoch {e['epoch']:>5}  dev WER {e['dev_wer']:.4f}")
    print(f"Best: step {log['best_step']}, dev WER {log['best_dev_wer']:.4f}, saved in {log['checkpoint']}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
