"""Draw the test WER of every scored run with its 95% confidence interval.

    python src/plot_results.py

Reads results/results_table.csv (written by src/score_all.py) and writes
results/figures/wer_ci.png. Run it after score_all.py; nothing is typed by hand.

Each run is a dot at its WER with a horizontal line from the low to the high end of its 95%
bootstrap interval, with end caps. Left panel: every scored run on a logarithmic WER axis, because
the scores run from about 0.04 to above 1 and on a straight axis the intervals of the best models
would be hidden under their dots. Right panel: the four fine-tuning conditions only, zoomed in on a
straight axis, because their differences are too small to see next to the baselines.
Colour and marker shape say whether a model heard our test speakers (from report/baseline_audit.md):
the shape repeats the colour's meaning, so the figure also reads in grey.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GROUPS = {  # group -> (label, colour, marker)
    "ours": ("Ours: Whisper-small fine-tuned, never heard the test speakers", "#2a78d6", "o"),
    "heard": ("Baseline that heard our test speakers", "#eb6834", "s"),
    "not_heard": ("Baseline that did not, or not documented", "#1baf7a", "D"),
}


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "scored"]
    for r in rows:
        for k in ("wer", "wer_ci_low", "wer_ci_high"):
            r[k] = float(r[k])
        r["group"] = ("ours" if r["part"] == "Our contribution"
                      else "heard" if r["heard_test_speakers"].lower() == "yes" else "not_heard")
    return rows


def draw(ax, rows: list[dict], label_fmt: str) -> None:
    for y, r in enumerate(rows):
        _, colour, marker = GROUPS[r["group"]]
        ax.hlines(y, r["wer_ci_low"], r["wer_ci_high"], color=colour, linewidth=2, zorder=2)
        ax.vlines([r["wer_ci_low"], r["wer_ci_high"]], y - 0.22, y + 0.22, color=colour, linewidth=1.5, zorder=2)
        ax.plot(r["wer"], y, marker=marker, markersize=6, color=colour,
                markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)
        ax.annotate(label_fmt.format(r["wer"]), (r["wer_ci_high"], y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [r["run"] for r in rows], fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_facecolor(SURFACE)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default=str(ROOT / "results" / "results_table.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figures" / "wer_ci.png"))
    args = ap.parse_args(argv)

    rows = sorted(load(Path(args.table)), key=lambda r: r["wer"])
    ours = [r for r in rows if r["group"] == "ours"]
    if not rows:
        raise SystemExit("no scored runs in the results table yet")

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor(SURFACE)

    draw(left, rows, "{:.3f}")
    left.set_xscale("log")
    left.set_xlim(min(r["wer_ci_low"] for r in rows) * 0.8, max(r["wer_ci_high"] for r in rows) * 1.6)
    ticks = [t for t in (0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0) if left.get_xlim()[0] <= t <= left.get_xlim()[1]]
    left.set_xticks(ticks, [f"{t:g}" for t in ticks])
    left.minorticks_off()
    left.set_xlabel("Test WER, logarithmic axis (lower is better)", fontsize=10, color=MUTED)
    left.set_title("All scored models", loc="left", fontsize=12, color=INK, fontweight="bold")

    if ours:
        draw(right, ours, "{:.4f}")
        lo = min(r["wer_ci_low"] for r in ours)
        hi = max(r["wer_ci_high"] for r in ours)
        pad = (hi - lo) * 0.25
        right.set_xlim(lo - pad, hi + pad * 2)
        right.set_xlabel("Test WER (lower is better), zoomed", fontsize=10, color=MUTED)
        right.set_title("Our four training sets", loc="left", fontsize=12, color=INK, fontweight="bold")
    else:
        right.axis("off")

    handles = [plt.Line2D([], [], color=c, marker=m, linewidth=2, markersize=8, label=l)
               for l, c, m in GROUPS.values()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.01), labelcolor=INK)
    fig.text(0.01, 0.985, "Dot: WER on the 1,521 test clips. Line: 95% confidence interval "
             "(bootstrap over clips, 2,000 resamples, seed 42).", fontsize=9.5, color=MUTED, va="top")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
