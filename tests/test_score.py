"""Tests for src/score_all.py: error counts, corpus-level rates, bootstrap and the tables."""
import csv
import statistics
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import score_all  # noqa: E402
from score_all import count_errors, make_resamples, bootstrap_ci, paired_bootstrap, mean_counts  # noqa: E402


def test_lecture_example():
    """Reference 'the cat sat on the mat', prediction 'the cat sit on the':
    1 substitution, 1 deletion, 0 insertions, WER 0.3333, CER 0.2273 (5 of 22 characters)."""
    c = count_errors(["the cat sat on the mat"], ["the cat sit on the"])
    assert c.word_errors == [2] and c.words == [6]
    assert c.wer == pytest.approx(0.3333, abs=1e-4)
    assert c.char_errors == [5] and c.chars == [22]
    assert c.cer == pytest.approx(0.2273, abs=1e-4)


def test_normalisation_applied_before_scoring():
    c = count_errors(["kaixo mundua"], ["Kaixo, Mundua!"])
    assert c.wer == 0 and c.cer == 0


def test_corpus_level_not_mean_of_clips():
    # Clip 1: 1 error in 1 word (100%); clip 2: 0 errors in 9 words. Corpus WER = 1/10, not 50%.
    c = count_errors(["a", "b c d e f g h i j"], ["x", "b c d e f g h i j"])
    assert c.wer == pytest.approx(0.1)


def test_insertions_make_wer_above_one():
    # A looping model: WER has no upper bound.
    c = count_errors(["bai"], ["bai bai bai bai"])
    assert c.word_errors == [3] and c.wer == pytest.approx(3.0)


def test_empty_hypothesis_is_all_deletions():
    c = count_errors(["bai eta ez"], [""])
    assert c.word_errors == [3] and c.wer == pytest.approx(1.0)


def test_empty_reference_rejected():
    with pytest.raises(ValueError):
        count_errors(["..."], ["bai"])


def test_resamples_reproducible():
    assert make_resamples(50, 20, 42) == make_resamples(50, 20, 42)
    assert make_resamples(50, 20, 42) != make_resamples(50, 20, 43)


def test_bootstrap_ci_contains_estimate():
    refs = [f"hitz {i} bat bi hiru" for i in range(200)]
    hyps = [r if i % 4 else "hitz okerra" for i, r in enumerate(refs)]
    c = count_errors(refs, hyps)
    low, high = bootstrap_ci(c.word_errors, c.words, make_resamples(len(refs), 500))
    assert low <= c.wer <= high and high - low > 0


def test_paired_identical_runs():
    refs = [f"a b c {i}" for i in range(100)]
    c = count_errors(refs, refs)
    r = paired_bootstrap(c.word_errors, c.word_errors, c.words, make_resamples(100, 200))
    assert r["diff"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0 and r["p_value"] == 1.0


def test_paired_clear_difference():
    refs = [f"a b c d {i}" for i in range(300)]
    good = count_errors(refs, refs)
    bad = count_errors(refs, ["a b x d " + str(i) for i in range(300)])
    r = paired_bootstrap(good.word_errors, bad.word_errors, good.words, make_resamples(300, 500))
    assert r["diff"] == pytest.approx(-0.2)
    assert r["ci_high"] < 0 and r["p_value"] < 0.01


def test_mean_over_seeds_equals_mean_of_wers():
    refs = ["a b c d", "e f g h i j"]
    s1 = count_errors(refs, ["a b c d", "e f x h i j"])
    s2 = count_errors(refs, ["a x c d", "e f x h x j"])
    assert mean_counts([s1, s2]).wer == pytest.approx(statistics.mean([s1.wer, s2.wer]))


# ---------------------------------------------------------------- end to end on a tiny repo

def _write(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def tiny(tmp_path):
    refs = [(f"c{i}", f"kaixo mundua {i} bat") for i in range(40)]
    manifests, preds = tmp_path / "manifests", tmp_path / "predictions"
    for split in ("dev", "test"):
        _write(manifests / f"{split}.csv", ["clip_id", "path", "text", "duration", "speaker_id"],
               [[c, f"{c}.mp3", t, "3.0", "1"] for c, t in refs])

    def pred(run, split, wrong_every):
        _write(preds / f"{run}_{split}.csv", ["clip_id", "hypothesis"],
               [[c, "okerra" if i % wrong_every == 0 else t.upper() + "."] for i, (c, t) in enumerate(refs)])

    for split in ("dev", "test"):
        pred("whisper-small-zeroshot", split, 1)       # every clip wrong
        pred("ft-ours_seed1", split, 10)
        pred("ft-ours_seed2", split, 8)
        pred("ft-rawmatched_seed1", split, 4)
        pred("parakeet-v3", split, 2)                  # unsupported: must never be scored
        pred("my-experiment", split, 2)                # unknown name: ignored
    _write(preds / "hitz-ctc_test.csv", ["clip_id", "hypothesis"], [[c, t] for c, t in refs[:-1]])  # one clip missing
    return tmp_path


def test_end_to_end(tiny):
    out = tiny / "results"
    score_all.main(["--pred-dir", str(tiny / "predictions"), "--manifest-dir", str(tiny / "manifests"),
                    "--out-dir", str(out)])
    with open(out / "results_table.csv", encoding="utf-8") as f:
        rows = {r["run"]: r for r in csv.DictReader(f)}
    # Two separate parts, all eight baselines and all four conditions listed.
    assert {r["part"] for r in rows.values()} == {"Baselines", "Our contribution"}
    assert len(rows) == len(score_all.BASELINES) + len(score_all.CONDITIONS)
    assert "parakeet-v3" not in rows and "my-experiment" not in rows
    assert rows["hitz-ctc"]["status"] == "no prediction file"  # incomplete file is skipped
    assert float(rows["whisper-small-zeroshot"]["wer"]) == pytest.approx(1.0)
    ours = rows["ft-ours"]
    assert ours["seeds"] == "1 2" and ours["wer_sd"] != ""
    assert float(ours["wer"]) == pytest.approx(statistics.mean(float(w) for w in ours["per_seed_wer"].split()), abs=1e-5)

    with open(out / "paired_comparisons.csv", encoding="utf-8") as f:
        comps = list(csv.DictReader(f))
    assert len(comps) == len(score_all.COMPARISONS)
    first = comps[0]  # ft-ours vs ft-rawmatched: ours makes fewer errors
    assert first["a"] == "ft-ours" and float(first["wer_diff"]) < 0
    last = comps[-1]  # best fine-tuned condition, chosen on dev, vs zero-shot
    assert last["a"] == "ft-ours" and float(last["wer_diff"]) < 0
    assert (out / "results_table.md").read_text(encoding="utf-8").count("not scored yet") >= 1


def test_dev_outputs_have_their_own_names(tiny):
    out = tiny / "results"
    score_all.main(["--split", "dev", "--pred-dir", str(tiny / "predictions"),
                    "--manifest-dir", str(tiny / "manifests"), "--out-dir", str(out)])
    assert (out / "results_table_dev.csv").exists() and not (out / "results_table.csv").exists()


def test_heard_test_speakers_read_from_audit():
    heard = score_all.heard_test_speakers()
    if not score_all.AUDIT.exists():
        pytest.skip("no audit yet")
    # Every baseline has a row in the audit, and the audit's answers are kept as written.
    assert set(heard) >= {name for name, _ in score_all.BASELINES}
    assert heard["wav2vec2-cv-eu"] == "No" and heard["hitz-ctc"] == "Yes"


def test_readme_updated_between_markers(tiny):
    readme = tiny / "README.md"
    readme.write_text(f"# Title\n\nBefore.\n\n{score_all.README_START}\nold numbers\n{score_all.README_END}\n\nAfter.\n",
                      encoding="utf-8")
    score_all.main(["--pred-dir", str(tiny / "predictions"), "--manifest-dir", str(tiny / "manifests"),
                    "--out-dir", str(tiny / "results"), "--readme", str(readme)])
    text = readme.read_text(encoding="utf-8")
    assert "old numbers" not in text and "Before." in text and "After." in text
    assert "### Baselines" in text and "# Results" not in text
    assert text.count(score_all.README_START) == 1 and text.count(score_all.README_END) == 1


def test_readme_without_markers_untouched(tiny):
    readme = tiny / "README.md"
    readme.write_text("# Title\n", encoding="utf-8")
    score_all.main(["--pred-dir", str(tiny / "predictions"), "--manifest-dir", str(tiny / "manifests"),
                    "--out-dir", str(tiny / "results"), "--readme", str(readme)])
    assert readme.read_text(encoding="utf-8") == "# Title\n"


def test_speed_table_copied_from_runtime_table(tmp_path):
    runtime = tmp_path / "runtime_table.md"
    runtime.write_text("# Hardware\n\n## Inference speed (pretrained models)\n\nText.\n\n| Run | RTFx |\n|---|---:|\n"
                       "| `a` | 1.0 |\n\n- note\n\n## Training time\n\n| x | y |\n", encoding="utf-8")
    assert score_all.speed_table(runtime) == "| Run | RTFx |\n|---|---:|\n| `a` | 1.0 |"
    readme = tmp_path / "README.md"
    readme.write_text(f"A\n{score_all.SPEED_START}\nold\n{score_all.SPEED_END}\nB\n", encoding="utf-8")
    assert score_all.replace_block(readme, score_all.SPEED_START, score_all.SPEED_END, "| new |")
    assert readme.read_text(encoding="utf-8") == f"A\n{score_all.SPEED_START}\n| new |\n{score_all.SPEED_END}\nB\n"
