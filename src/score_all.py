"""Score every run on one split and write the results tables. Nobody edits the outputs by hand.

    python src/score_all.py                # test: the official scores
    python src/score_all.py --split dev    # dev: quick checks and the dev-based choices

Reads
  data/raw/manifests/<split>.csv               references (column `text`)
  results/predictions/<run>_<split>.csv        raw model output (columns clip_id, hypothesis)

Writes (for --split dev the file names end in _dev instead)
  results/results_table.csv / .md              one row per baseline and per fine-tuning condition
  results/paired_comparisons.csv               the five fixed comparisons (see COMPARISONS)
  results/per_run_test.csv                     one row per scored run, every seed separately
  README.md                                    the same tables, between the results markers
                                               (test split and default --out-dir only)

Scoring (report/scoring_policy.md):
  * Reference and hypothesis go through src/normalize.py. Nothing else is changed.
  * WER and CER are corpus-level: all errors (substitutions + deletions + insertions) over all
    reference words (or characters, spaces included), computed from jiwer's alignment. Not the
    mean of per-clip rates, which would give a 3-word clip the same weight as a 30-word clip.
  * 95% confidence interval: bootstrap over clips, 2,000 resamples, seed 42, percentile method.
    The same resamples are used for every run, so paired comparisons line up clip by clip.
  * A fine-tuning condition (ft-raw, ...) with several seeds is scored on the mean error count of
    each clip over its seeds. Its corpus WER is then exactly the mean of the seeds' WERs; the SD
    over seeds is reported next to it.
  * Paired comparison A vs B: difference A - B (negative means A makes fewer errors), its 95%
    interval, and a two-sided p-value: 2 x the share of resamples on the other side of zero,
    at most 1. With 2,000 resamples the smallest value it can show is "< 0.001".

A run must have exactly one prediction for every clip of the manifest; otherwise it is skipped
with a message, never scored on a subset. A run name that is not in the tables below is ignored
with a message, and a model listed as unsupported (no Basque) is never scored.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESAMPLES = 2000
SEED = 42

# Baselines, in the order of the README table: (run name, Hugging Face model id).
BASELINES = [
    ("whisper-large-v3", "openai/whisper-large-v3"),
    ("whisper-small-zeroshot", "openai/whisper-small"),
    ("hitz-ctc", "HiTZ/stt_eu_conformer_ctc_large"),
    ("hitz-transducer", "HiTZ/stt_eu_conformer_transducer_large"),
    ("parakeet-v3-basque", "itzune/parakeet-tdt-0.6b-v3-basque"),
    ("wav2vec2-bp1", "gttsehu/wav2vec2-xls-r-300m-bp1-es_eu"),
    ("wav2vec2-cv-eu", "stefan-it/wav2vec2-large-xlsr-53-basque"),
    ("mhubert-basque", "Ansu/mHubert-basque-ASR"),
]

# Our contribution: Whisper-small fine-tuned on data/cleaned/train_<condition>.csv.
CONDITIONS = [
    ("ft-raw", "train_raw.csv: raw sample, leakage rule only (9.24 h)"),
    ("ft-rawmatched", "train_rawmatched.csv: random part of the raw sample, same hours as the cleaned sets"),
    ("ft-authors", "train_authors.csv: the corpus authors' clean split"),
    ("ft-ours", "train_ours.csv: our cleaning script"),
]

# Models without Basque support: listed in the README as not run, never scored.
NOT_RUN = {"canary-v2", "canary-1b-v2", "parakeet-v3", "parakeet-tdt-0.6b-v3", "qwen3-asr"}

# Fixed before any test score existed (task file, 17 November). "best-ft" is the fine-tuning
# condition with the lowest mean dev WER; it is chosen on dev, never on test.
COMPARISONS = [
    ("ft-ours", "ft-rawmatched", "Does cleaning help beyond having less data?"),
    ("ft-ours", "ft-authors", "Does our cleaning match the authors' cleaning?"),
    ("ft-authors", "ft-rawmatched", "Does the authors' cleaning help beyond having less data?"),
    ("ft-raw", "ft-rawmatched", "Do more but noisier hours win?"),
    ("best-ft", "whisper-small-zeroshot", "How much does fine-tuning help?"),
]

SEED_RUN = re.compile(r"^(ft-[a-z]+)_seed(\d+)$")

AUDIT = ROOT / "report" / "baseline_audit.md"
README = ROOT / "README.md"
README_START = "<!-- results:start (written by src/score_all.py, do not edit by hand) -->"
README_END = "<!-- results:end -->"


def heard_test_speakers(audit: Path = AUDIT) -> dict[str, str]:
    """Run name -> the "Heard our test speakers?" cell of report/baseline_audit.md (Tugra's audit)."""
    if not audit.exists():
        return {}
    header, out = None, {}
    for line in audit.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None and "Heard our test speakers?" in cells:
            header = cells
        elif header and len(cells) == len(header) and cells[0].startswith("`"):
            out[cells[0].strip("`")] = cells[header.index("Heard our test speakers?")].split(" (")[0]
    return out


# ---------------------------------------------------------------- counting errors

@dataclass
class Counts:
    """Per-clip error counts and lengths, in manifest order."""
    word_errors: list[float]
    words: list[int]
    char_errors: list[float]
    chars: list[int]

    @property
    def wer(self) -> float:
        return sum(self.word_errors) / sum(self.words)

    @property
    def cer(self) -> float:
        return sum(self.char_errors) / sum(self.chars)


def _errors_per_sentence(output) -> list[int]:
    """Substitutions + deletions + insertions for each sentence of a jiwer output."""
    errors = []
    for chunks in output.alignments:
        n = 0
        for c in chunks:
            if c.type in ("substitute", "delete"):
                n += c.ref_end_idx - c.ref_start_idx
            elif c.type == "insert":
                n += c.hyp_end_idx - c.hyp_start_idx
        errors.append(n)
    return errors


def count_errors(references: list[str], hypotheses: list[str]) -> Counts:
    """Normalise both sides, align with jiwer and count errors per clip."""
    refs = [normalize(t) for t in references]
    hyps = [normalize(t) for t in hypotheses]
    empty = [i for i, r in enumerate(refs) if not r]
    if empty:
        raise ValueError(f"{len(empty)} reference(s) are empty after normalisation (first: row {empty[0]})")
    words = jiwer.process_words(refs, hyps)
    chars = jiwer.process_characters(refs, hyps)
    return Counts(
        word_errors=_errors_per_sentence(words),
        words=[len(r.split()) for r in refs],
        char_errors=_errors_per_sentence(chars),
        chars=[len(r) for r in refs],
    )


def mean_counts(runs: list[Counts]) -> Counts:
    """Per-clip mean error counts over seeds (lengths are the same for every seed)."""
    k = len(runs)
    return Counts(
        word_errors=[sum(v) / k for v in zip(*(r.word_errors for r in runs))],
        words=runs[0].words,
        char_errors=[sum(v) / k for v in zip(*(r.char_errors for r in runs))],
        chars=runs[0].chars,
    )


# ---------------------------------------------------------------- bootstrap

def make_resamples(n_clips: int, n_resamples: int = RESAMPLES, seed: int = SEED) -> list[list[int]]:
    """Clip indices for every resample, drawn with replacement. Generated once per split."""
    rng = random.Random(seed)
    population = range(n_clips)
    return [rng.choices(population, k=n_clips) for _ in range(n_resamples)]


def percentile(sorted_values: list[float], q: float) -> float:
    """q-th percentile (0-100) with linear interpolation (numpy's default method)."""
    pos = (len(sorted_values) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def _resampled_rates(errors: list[float], lengths: list[int], resamples: list[list[int]]) -> list[float]:
    e, n = errors.__getitem__, lengths.__getitem__
    return [sum(map(e, idx)) / sum(map(n, idx)) for idx in resamples]


def bootstrap_ci(errors: list[float], lengths: list[int], resamples: list[list[int]]) -> tuple[float, float]:
    rates = sorted(_resampled_rates(errors, lengths, resamples))
    return percentile(rates, 2.5), percentile(rates, 97.5)


def paired_bootstrap(errors_a: list[float], errors_b: list[float], lengths: list[int],
                     resamples: list[list[int]]) -> dict:
    """Difference of corpus rates A - B, its 95% interval and a two-sided p-value."""
    observed = (sum(errors_a) - sum(errors_b)) / sum(lengths)
    a, b, n = errors_a.__getitem__, errors_b.__getitem__, lengths.__getitem__
    diffs = sorted((sum(map(a, idx)) - sum(map(b, idx))) / sum(map(n, idx)) for idx in resamples)
    at_or_below = sum(d <= 0 for d in diffs) / len(diffs)
    at_or_above = sum(d >= 0 for d in diffs) / len(diffs)
    return {
        "diff": observed,
        "ci_low": percentile(diffs, 2.5),
        "ci_high": percentile(diffs, 97.5),
        "p_value": min(1.0, 2 * min(at_or_below, at_or_above)),
    }


# ---------------------------------------------------------------- files

def read_manifest(path: Path) -> tuple[list[str], list[str]]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["clip_id"] for r in rows], [r["text"] for r in rows]


def read_predictions(path: Path, clip_ids: list[str]) -> list[str]:
    """Hypotheses in manifest order. Raises ValueError unless every clip has exactly one."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"clip_id", "hypothesis"} <= set(reader.fieldnames):
            raise ValueError(f"{path.name}: columns must include clip_id, hypothesis")
        rows = list(reader)
    hyps: dict[str, str] = {}
    for r in rows:
        if r["clip_id"] in hyps:
            raise ValueError(f"{path.name}: clip {r['clip_id']} appears twice")
        hyps[r["clip_id"]] = r["hypothesis"] or ""
    missing = [c for c in clip_ids if c not in hyps]
    extra = set(hyps) - set(clip_ids)
    if missing or extra:
        raise ValueError(f"{path.name}: {len(missing)} manifest clip(s) missing, {len(extra)} unknown clip(s)")
    return [hyps[c] for c in clip_ids]


def find_runs(pred_dir: Path, split: str) -> dict[str, Path]:
    suffix = f"_{split}.csv"
    return {p.name[: -len(suffix)]: p for p in sorted(pred_dir.glob(f"*{suffix}"))}


# ---------------------------------------------------------------- scoring

@dataclass
class Scored:
    name: str
    counts: Counts
    wer_ci: tuple[float, float]
    cer_ci: tuple[float, float]
    seeds: dict[int, Counts] = field(default_factory=dict)  # fine-tuning conditions only


def score_split(split: str, pred_dir: Path, manifest_dir: Path, resamples_n: int = RESAMPLES,
                seed: int = SEED, log=print) -> tuple[dict[str, Scored], dict[str, Counts], list[int]]:
    """Score every recognised run. Returns (baselines + conditions by name, every run by name, lengths)."""
    clip_ids, references = read_manifest(manifest_dir / f"{split}.csv")
    resamples = make_resamples(len(clip_ids), resamples_n, seed)
    baseline_names = {name for name, _ in BASELINES}
    condition_names = {name for name, _ in CONDITIONS}

    per_run: dict[str, Counts] = {}
    for run, path in find_runs(pred_dir, split).items():
        m = SEED_RUN.match(run)
        if run in NOT_RUN:
            log(f"skip {run}: model without Basque support, never scored")
            continue
        if run not in baseline_names and not (m and m.group(1) in condition_names):
            log(f"skip {run}: not a known run name")
            continue
        try:
            per_run[run] = count_errors(references, read_predictions(path, clip_ids))
        except ValueError as e:
            log(f"skip {run}: {e}")

    scored: dict[str, Scored] = {}
    for name in baseline_names & set(per_run):
        c = per_run[name]
        scored[name] = Scored(name, c, bootstrap_ci(c.word_errors, c.words, resamples),
                              bootstrap_ci(c.char_errors, c.chars, resamples))
    for name in condition_names:
        seeds = {int(SEED_RUN.match(r).group(2)): c for r, c in per_run.items()
                 if SEED_RUN.match(r) and SEED_RUN.match(r).group(1) == name}
        if seeds:
            c = mean_counts([seeds[s] for s in sorted(seeds)])
            scored[name] = Scored(name, c, bootstrap_ci(c.word_errors, c.words, resamples),
                                  bootstrap_ci(c.char_errors, c.chars, resamples), dict(sorted(seeds.items())))
    return scored, per_run, resamples


def best_condition_on_dev(pred_dir: Path, manifest_dir: Path) -> str | None:
    """The fine-tuning condition with the lowest mean dev WER over its seeds (no bootstrap)."""
    try:
        clip_ids, references = read_manifest(manifest_dir / "dev.csv")
    except FileNotFoundError:
        return None
    means = {}
    for name, _ in CONDITIONS:
        wers = []
        for run, path in find_runs(pred_dir, "dev").items():
            m = SEED_RUN.match(run)
            if m and m.group(1) == name:
                try:
                    wers.append(count_errors(references, read_predictions(path, clip_ids)).wer)
                except ValueError:
                    pass
        if wers:
            means[name] = statistics.mean(wers)
    return min(means, key=means.get) if means else None


def compare(scored: dict[str, Scored], resamples: list[list[int]], best_ft: str | None) -> list[dict]:
    rows = []
    for a, b, question in COMPARISONS:
        a_name = best_ft if a == "best-ft" else a
        row = {"a": a_name or a, "b": b, "question": question}
        if a == "best-ft":
            row["note"] = "A chosen by mean dev WER" if best_ft else "no fine-tuned dev predictions yet"
        if a_name in scored and b in scored:
            A, B = scored[a_name].counts, scored[b].counts
            for metric, ea, eb, n in (("wer", A.word_errors, B.word_errors, A.words),
                                      ("cer", A.char_errors, B.char_errors, A.chars)):
                for k, v in paired_bootstrap(ea, eb, n, resamples).items():
                    row[f"{metric}_{k}"] = v
        rows.append(row)
    return rows


# ---------------------------------------------------------------- output

def _f(x: float | None, digits: int = 4) -> str:
    return "" if x is None else f"{x:.{digits}f}"


def _p(p: float | None) -> str:
    return "" if p is None else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def table_rows(scored: dict[str, Scored], heard: dict[str, str] | None = None) -> list[dict]:
    heard = heard or {}
    rows = []
    for name, model in BASELINES:
        s = scored.get(name)
        rows.append({
            "part": "Baselines", "run": name, "trained_on": model, "seeds": "",
            "heard_test_speakers": heard.get(name, ""),
            "wer": s and s.counts.wer, "wer_ci_low": s and s.wer_ci[0], "wer_ci_high": s and s.wer_ci[1],
            "wer_sd": None, "cer": s and s.counts.cer, "cer_ci_low": s and s.cer_ci[0],
            "cer_ci_high": s and s.cer_ci[1], "cer_sd": None, "per_seed_wer": "",
            "status": "scored" if s else "no prediction file",
        })
    for name, data in CONDITIONS:
        s = scored.get(name)
        seed_wers = [c.wer for c in s.seeds.values()] if s else []
        seed_cers = [c.cer for c in s.seeds.values()] if s else []
        rows.append({
            "part": "Our contribution", "run": name, "trained_on": data,
            "heard_test_speakers": "No",
            "seeds": " ".join(str(k) for k in s.seeds) if s else "",
            "wer": s and s.counts.wer, "wer_ci_low": s and s.wer_ci[0], "wer_ci_high": s and s.wer_ci[1],
            "wer_sd": statistics.stdev(seed_wers) if len(seed_wers) > 1 else None,
            "cer": s and s.counts.cer, "cer_ci_low": s and s.cer_ci[0], "cer_ci_high": s and s.cer_ci[1],
            "cer_sd": statistics.stdev(seed_cers) if len(seed_cers) > 1 else None,
            "per_seed_wer": " ".join(f"{w:.4f}" for w in seed_wers),
            "status": "scored" if s else "no prediction file",
        })
    return rows


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (_f(v, 6) if isinstance(v, float) else ("" if v is None else v)) for k, v in r.items()})


def results_markdown(split: str, rows: list[dict], comparisons: list[dict], n_clips: int) -> str:
    def ci(r, m):
        return f"[{_f(r[m + '_ci_low'])}, {_f(r[m + '_ci_high'])}]" if r[m] is not None else ""

    def sd(x):
        return f" ± {_f(x)}" if x is not None else ""

    out = [
        f"# Results ({split} split, {n_clips} clips)", "",
        "Written by `src/score_all.py`. Do not edit by hand.", "",
        "Scoring policy: `report/scoring_policy.md`. WER and CER are corpus-level, after "
        "`src/normalize.py`. 95% intervals: bootstrap over clips, "
        f"{RESAMPLES:,} resamples, seed {SEED}.", "",
        "## Baselines (pretrained models, no training by us)", "",
        "Heard our test speakers: from `report/baseline_audit.md`. A model that heard them was "
        "trained on this corpus, so the comparison is in its favour.", "",
        "| Run | Model | Heard our test speakers? | WER | WER 95% CI | CER | CER 95% CI |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["part"] == "Baselines":
            if r["status"] == "scored":
                out.append(f"| `{r['run']}` | `{r['trained_on']}` | {r['heard_test_speakers']} | {_f(r['wer'])} | "
                           f"{ci(r, 'wer')} | {_f(r['cer'])} | {ci(r, 'cer')} |")
            else:
                out.append(f"| `{r['run']}` | `{r['trained_on']}` | {r['heard_test_speakers']} | not scored yet | | | |")
    out += [
        "", "## Our contribution (Whisper-small fine-tuned)", "",
        "Mean over seeds ± SD over seeds. The interval is the bootstrap over clips of the per-clip "
        "mean over seeds. None of our training sets contains a `dev` or `test` speaker.", "",
        "| Run | Training set | Seeds | WER | WER 95% CI | CER | CER 95% CI | WER per seed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["part"] == "Our contribution":
            if r["status"] == "scored":
                out.append(f"| `{r['run']}` | {r['trained_on']} | {r['seeds']} | {_f(r['wer'])}{sd(r['wer_sd'])} | "
                           f"{ci(r, 'wer')} | {_f(r['cer'])}{sd(r['cer_sd'])} | {ci(r, 'cer')} | {r['per_seed_wer']} |")
            else:
                out.append(f"| `{r['run']}` | {r['trained_on']} | | not scored yet | | | | |")
    out += [
        "", "## Paired comparisons (fixed before any test score)", "",
        "Difference = A − B; negative means A makes fewer errors. Paired bootstrap on the same clips; "
        "two-sided p-value.", "",
        "| A | B | Question | WER diff | 95% CI | p | CER diff | 95% CI | p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in comparisons:
        note = f" ({c['note']})" if c.get("note") else ""
        if "wer_diff" in c:
            out.append(f"| `{c['a']}` | `{c['b']}` | {c['question']}{note} | {c['wer_diff']:+.4f} | "
                       f"[{c['wer_ci_low']:+.4f}, {c['wer_ci_high']:+.4f}] | {_p(c['wer_p_value'])} | "
                       f"{c['cer_diff']:+.4f} | [{c['cer_ci_low']:+.4f}, {c['cer_ci_high']:+.4f}] | {_p(c['cer_p_value'])} |")
        else:
            out.append(f"| `{c['a']}` | `{c['b']}` | {c['question']}{note} | not scored yet | | | | | |")
    return "\n".join(out) + "\n"


def update_readme(readme: Path, markdown: str) -> bool:
    """Put the results between the README's markers, one heading level down. False if no markers."""
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return False
    body = markdown.split("\n", 2)[2]  # drop the "# Results" title line and the blank line after it
    body = "\n".join("#" + line if line.startswith("## ") else line for line in body.splitlines())
    head, rest = text.split(README_START, 1)
    tail = rest.split(README_END, 1)[1]
    readme.write_text(f"{head}{README_START}\n{body.strip()}\n{README_END}{tail}", encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["test", "dev"], default="test")
    ap.add_argument("--pred-dir", default=str(ROOT / "results" / "predictions"))
    ap.add_argument("--manifest-dir", default=str(ROOT / "data" / "raw" / "manifests"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--readme", default=None,
                    help="README to update between its results markers (default: README.md, only for the test "
                         "split and the default --out-dir)")
    args = ap.parse_args(argv)

    pred_dir, manifest_dir, out_dir = Path(args.pred_dir), Path(args.manifest_dir), Path(args.out_dir)
    scored, per_run, resamples = score_split(args.split, pred_dir, manifest_dir)
    best_ft = best_condition_on_dev(pred_dir, manifest_dir)
    comparisons = compare(scored, resamples, best_ft)
    rows = table_rows(scored, heard_test_speakers())

    tag = "" if args.split == "test" else f"_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / f"results_table{tag}.csv", rows, list(rows[0]))
    markdown = results_markdown(args.split, rows, comparisons, len(resamples[0]))
    (out_dir / f"results_table{tag}.md").write_text(markdown, encoding="utf-8")
    readme = Path(args.readme) if args.readme else (
        README if args.split == "test" and out_dir.resolve() == (ROOT / "results").resolve() else None)
    if readme and readme.exists() and update_readme(readme, markdown):
        print(f"Results written into {readme}")
    cmp_cols = ["a", "b", "question", "note"] + [f"{m}_{k}" for m in ("wer", "cer")
                                                  for k in ("diff", "ci_low", "ci_high", "p_value")]
    write_csv(out_dir / f"paired_comparisons{tag}.csv", comparisons, cmp_cols)
    write_csv(out_dir / f"per_run_{args.split}.csv",
              [{"run": r, "wer": c.wer, "cer": c.cer, "word_errors": sum(c.word_errors), "words": sum(c.words),
                "char_errors": sum(c.char_errors), "chars": sum(c.chars)} for r, c in sorted(per_run.items())],
              ["run", "wer", "cer", "word_errors", "words", "char_errors", "chars"])
    print(f"Scored {len(per_run)} run(s) on {args.split}; tables in {out_dir}")


if __name__ == "__main__":
    main()
