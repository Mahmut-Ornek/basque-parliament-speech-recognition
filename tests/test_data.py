"""Tests for the data pipeline.

The tests of `make_sample.py` run on a tiny synthetic corpus with the same layout as the real one
(`metadata/*.tsv` + `audio/*.tar`), so they need no download. The last tests check the real
manifests in `data/raw/manifests/` and are skipped until those files exist.
"""
import csv
import io
import zlib
import random
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import make_sample  # noqa: E402

TRAIN_SHARDS = 3


def _write_split(corpus: Path, split: str, n_clips: int, n_shards: int, rng: random.Random) -> None:
    rows = []
    first_year = {"train": 2013, "dev": 2022, "test": 2022}[split]
    for i in range(n_clips):
        language = rng.choice(["es", "es", "eu", "eu", "bi"])
        # Same shape as the corpus: <session>/<session>_<start>_<end>.mp3, sessions in date order.
        session = f"{split[:2]}-{i // 30:03d}_{first_year + i // 100}0211_01"
        start, length = (i % 30) * 12.0, rng.uniform(3, 10)
        rows.append(
            {
                "path": f"{session}/{session}_{start:.2f}_{start + length:.2f}.mp3",
                "language": language,
                "speaker_id": str(rng.randrange(12)),
                "PRR": f"{rng.uniform(50, 100):.2f}",
                "length": f"{length:.2f}",
                "sentence": f'{language} esaldia {i}, "komatxoekin"',
            }
        )
    (corpus / "metadata").mkdir(parents=True, exist_ok=True)
    with open(corpus / "metadata" / f"{split}.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
        writer.writeheader()
        writer.writerows(rows)
    (corpus / "audio").mkdir(parents=True, exist_ok=True)
    per_shard = -(-n_clips // n_shards)
    for shard in range(n_shards):
        with tarfile.open(corpus / "audio" / f"{split}_{shard}.tar", "w") as tar:
            for row in rows[shard * per_shard : (shard + 1) * per_shard]:
                data = f"fake mp3 bytes of {row['path']}".encode()
                info = tarfile.TarInfo(row["path"])
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    corpus = tmp_path_factory.mktemp("corpus")
    rng = random.Random(0)
    _write_split(corpus, "train", 900, TRAIN_SHARDS, rng)
    _write_split(corpus, "dev", 40, 1, rng)
    _write_split(corpus, "test", 40, 1, rng)
    return corpus


def run(corpus: Path, out: Path, hours: float, *extra: str) -> None:
    make_sample.main(
        ["--local-repo", str(corpus), "--out-root", str(out), "--hours", str(hours),
         "--train-shards", str(TRAIN_SHARDS), *extra]
    )


def manifest(out: Path, name: str) -> list[dict]:
    with open(out / "data" / "raw" / "manifests" / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def corpus_rows(corpus: Path, split: str) -> dict[str, dict]:
    with open(corpus / "metadata" / f"{split}.tsv", encoding="utf-8", newline="") as f:
        return {Path(r["path"]).stem: r for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}


def tree(out: Path) -> dict[str, bytes]:
    return {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}


def test_running_twice_gives_identical_files(corpus, tmp_path):
    run(corpus, tmp_path / "a", 0.2)
    run(corpus, tmp_path / "b", 0.2)
    run(corpus, tmp_path / "b", 0.2)  # a second run on the same folder changes nothing
    assert tree(tmp_path / "a") == tree(tmp_path / "b")


def test_manifest_format_and_basque_only(corpus, tmp_path):
    run(corpus, tmp_path, 0.2)
    for name, split in [("train_sample.csv", "train"), ("dev.csv", "dev"), ("test.csv", "test")]:
        source = corpus_rows(corpus, split)
        rows = manifest(tmp_path, name)
        assert list(rows[0]) == ["clip_id", "path", "text", "duration", "speaker_id"]  # no PRR
        for r in rows:
            src = source[r["clip_id"]]
            assert src["language"] == "eu"
            assert r["text"] == src["sentence"]  # transcript unchanged
            assert r["duration"] == src["length"] and r["speaker_id"] == src["speaker_id"]
    n_basque_dev = sum(r["language"] == "eu" for r in corpus_rows(corpus, "dev").values())
    assert len(manifest(tmp_path, "dev.csv")) == n_basque_dev  # dev and test are exported in full


def test_every_row_points_to_unchanged_audio(corpus, tmp_path):
    run(corpus, tmp_path, 0.2)
    for name in ("train_sample.csv", "dev.csv", "test.csv"):
        for r in manifest(tmp_path, name):
            assert r["path"].startswith("data/raw/audio/") and "\\" not in r["path"]
            assert b"fake mp3 bytes of" in (tmp_path / r["path"]).read_bytes()
            assert r["clip_id"].encode() in (tmp_path / r["path"]).read_bytes()


def test_sample_reaches_the_hours_and_is_nested(corpus, tmp_path):
    run(corpus, tmp_path / "small", 0.1)
    run(corpus, tmp_path / "large", 0.2)
    small, large = manifest(tmp_path / "small", "train_sample.csv"), manifest(tmp_path / "large", "train_sample.csv")
    assert large[: len(small)] == small  # same clips, same order, same paths
    seconds = [float(r["duration"]) for r in large]
    assert sum(seconds) >= 0.2 * 3600 > sum(seconds[:-1])  # stops at the first clip that reaches the target


def test_no_clip_id_shared_between_splits(corpus, tmp_path):
    run(corpus, tmp_path, 0.2)
    ids = [r["clip_id"] for name in ("train_sample.csv", "dev.csv", "test.csv") for r in manifest(tmp_path, name)]
    assert len(ids) == len(set(ids))


def test_folders_stay_small(corpus, tmp_path, monkeypatch):
    monkeypatch.setattr(make_sample, "CLIPS_PER_FOLDER", 7)
    run(corpus, tmp_path, 0.2)
    folders = [d for d in (tmp_path / "data" / "raw" / "audio").rglob("*") if d.is_dir() and d.name.isdigit()]
    assert len(folders) > 3 and all(len(list(d.iterdir())) <= 7 for d in folders)


def test_prune_removes_clips_of_a_larger_sample(corpus, tmp_path):
    run(corpus, tmp_path, 0.2)
    run(corpus, tmp_path, 0.1, "--prune")
    on_disk = {p.relative_to(tmp_path).as_posix() for p in (tmp_path / "data" / "raw" / "audio" / "train").rglob("*.mp3")}
    assert on_disk == {r["path"] for r in manifest(tmp_path, "train_sample.csv")}


def test_archive_subset_uses_only_those_archives(corpus, tmp_path):
    run(corpus, tmp_path, 0.1, "--archives", "1")
    (chosen,) = make_sample.choose_archives("1", TRAIN_SHARDS)
    with tarfile.open(corpus / "audio" / f"train_{chosen}.tar") as tar:
        members = {Path(name).stem for name in tar.getnames()}
    assert {r["clip_id"] for r in manifest(tmp_path, "train_sample.csv")} <= members


def test_session_and_year_come_from_the_path():
    path = "12-077_20220211_01/12-077_20220211_01_382.04_392.24.mp3"
    assert make_sample.session_and_year(path) == ("12-077_20220211_01", "2022")


def test_archive_choice():
    assert make_sample.choose_archives("all", 11) == list(range(11))
    assert make_sample.choose_archives("0,5,10", 11) == [0, 5, 10]
    assert make_sample.choose_archives("3", 11) == make_sample.choose_archives("3", 11)  # seeded
    with pytest.raises(SystemExit):
        make_sample.choose_archives("0,11", 11)


def test_unknown_speaker_is_not_counted_as_a_speaker():
    rows = [{"speaker_id": s} for s in ["0", "416", "0", "417", "416"]]
    assert make_sample.count_speakers(rows) == (2, 2)


def test_too_few_hours_in_the_archives_is_an_error(corpus, tmp_path):
    with pytest.raises(SystemExit, match="Use more archives"):
        run(corpus, tmp_path, 50)


# --- the real data in this repository -------------------------------------------------------

REAL = REPO / "data" / "raw" / "manifests"
needs_real_data = pytest.mark.skipif(
    not all((REAL / n).is_file() for n in ("train_sample.csv", "dev.csv", "test.csv")),
    reason="raw manifests are not in the repo yet",
)


@needs_real_data
def test_real_manifest_rows_point_to_audio():
    for name in ("train_sample.csv", "dev.csv", "test.csv"):
        missing = [r["path"] for r in manifest(REPO, name) if not (REPO / r["path"]).is_file()]
        assert not missing, f"{name}: {len(missing)} rows without audio, e.g. {missing[:3]}"


@needs_real_data
def test_real_splits_share_no_clip_id():
    train = {r["clip_id"] for r in manifest(REPO, "train_sample.csv")}
    held_out = {r["clip_id"] for name in ("dev.csv", "test.csv") for r in manifest(REPO, name)}
    assert not train & held_out


@needs_real_data
def test_real_dev_and_test_are_complete():
    assert len(manifest(REPO, "dev.csv")) == 1397
    assert len(manifest(REPO, "test.csv")) == 1521


# === clean.py ================================================================================

import clean  # noqa: E402

WORDS = ("eta da ez bat hau hori gure legebiltzar talde sailburu andrea jauna eskerrik asko egun on "
         "guztioi lehendakari proposamen eztabaida puntu gai herritar osasun hezkuntza aurrekontu").split()


def _row(session: str, start: float, text: str, duration: float | None = None, speaker: str = "7") -> dict:
    if duration is None:  # speakers differ: 10 to 18 characters per second, fixed per text
        duration = round(len(text) / (10 + zlib.crc32(text.encode()) % 800 / 100), 2)
    clip_id = f"{session}_{start:.2f}_{start + duration:.2f}"
    return {"clip_id": clip_id, "path": f"data/raw/audio/x/000/{clip_id}.mp3", "text": text,
            "duration": f"{duration:.2f}", "speaker_id": speaker}


def _sentences(n: int, rng: random.Random) -> list[str]:
    out = set()
    while len(out) < n:
        out.add(" ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 14))))
    return sorted(out)


def _noisy(text: str) -> str:
    """What a decent recogniser returns: the text with a wrong letter now and then."""
    return "".join("x" if i % 17 == 5 and c != " " else c for i, c in enumerate(text))


def _judge_row(row: dict, hypothesis: str | None = None, es_prob: float = 0.02, audio_seconds: float | None = None) -> dict:
    seconds = float(row["duration"]) + 0.03 if audio_seconds is None else audio_seconds
    return {"clip_id": row["clip_id"], "hypothesis": _noisy(row["text"]) if hypothesis is None else hypothesis,
            "es_prob": f"{es_prob:.4f}", "eu_prob": f"{1 - es_prob:.4f}", "top_language": "eu",
            "audio_seconds": f"{seconds:.2f}"}


@pytest.fixture()
def cleaning_repo(tmp_path) -> dict:
    """A repo with manifests, judge outputs and a local corpus folder. Bad clips are planted in
    the training sample; `expected` says which rule must remove each of them."""
    rng = random.Random(1)
    texts = _sentences(480, rng)
    dev = [_row("12-077_20220211_01", 20.0 * i, t) for i, t in enumerate(texts[:120])]
    test = [_row("12-078_20220218_01", 20.0 * i, t) for i, t in enumerate(texts[120:180])]
    test[0] = _row("12-078_20220218_01", 0.0, "eskerrik asko", duration=1.0)
    train = [_row(f"11-{i // 50:03d}_2019{1 + i // 50:02d}15_01", 20.0 * (i % 50), t) for i, t in enumerate(texts[180:480])]

    judge = {r["clip_id"]: _judge_row(r) for r in train}
    expected = {}

    def plant(row: dict, rule: str, **judge_args) -> None:
        train.insert(rng.randrange(len(train)), row)
        judge[row["clip_id"]] = _judge_row(row, **judge_args)
        expected[row["clip_id"]] = rule

    s = "11-900_20201020_01"
    plant(_row(s, 0, "Eskerrik asko.", duration=1.0), "leakage_text")
    plant(_row(s, 20, test[5]["text"]), "leakage_text")
    plant(_row("12-078_20220218_01", 400.5, "gure talde hau ez da hori bat"), "leakage_audio_overlap")  # inside a test clip's time
    plant(_row(s, 40, "aurrekontu 2022 da gure gai hau"), "broken_text")
    plant(_row(s, 60, " ", duration=4.0), "broken_text")
    plant(_row(s, 80, " ".join(WORDS), duration=1.5), "speaking_rate")
    plant(_row(s, 100, "bat eta bat", duration=9.5), "speaking_rate")
    bad_name = _row(s, 120, "gure legebiltzar talde hau da eta ez da hori")
    bad_name["duration"] = "9.90"  # the file name says about 3 seconds
    bad_name["text"] = "gure legebiltzar talde hau da eta ez da hori gure legebiltzar talde hau da eta ez da hori gure legebiltzar talde hau da"
    plant(bad_name, "duration_mismatch")
    plant(_row(s, 140, "sailburu andrea eta jauna gure talde hau da"), "duration_mismatch", audio_seconds=1.0)
    formula = "egun on guztioi lehendakari andrea"
    for k in range(5):
        row = _row(s, 200 + 20 * k, formula)
        train.append(row)
        judge[row["clip_id"]] = _judge_row(row)
        if k >= 3:
            expected[row["clip_id"]] = "repeated_text"
    first = _row(s, 400, "proposamen hau gure talde osasun eta hezkuntza gai da")
    train.append(first)
    judge[first["clip_id"]] = _judge_row(first)
    late = _row(s, 402, "osasun eta hezkuntza gai da eta ez da bat")
    train.append(late)
    judge[late["clip_id"]] = _judge_row(late)
    expected[late["clip_id"]] = "repeated_audio_overlap"
    for k in range(8):
        row = _row(s, 500 + 20 * k, texts[180 + k] + " eta " + texts[300 + k])
        plant(row, "judge_cer", hypothesis=_noisy(texts[400 + k]))  # the audio says something else
    for k in range(3):
        plant(_row(s, 800 + 20 * k, texts[200 + k] + " hau da"), "wrong_language", es_prob=0.97)

    out = tmp_path / "repo"
    for name, rows in (("train_sample", train), ("dev", dev), ("test", test)):
        clean.write_csv(out / "data" / "raw" / "manifests" / f"{name}.csv", clean.MANIFEST_COLUMNS, rows)
    clean.write_csv(clean.judge_path(out, "train"), clean.JUDGE_COLUMNS, [judge[r["clip_id"]] for r in train])
    clean.write_csv(clean.judge_path(out, "dev"), clean.JUDGE_COLUMNS, [_judge_row(r) for r in dev])

    # The authors' clean split: everything except the clips whose audio says something else
    # and forty good clips.
    dropped = {c for c, rule in expected.items() if rule == "judge_cer"} | {r["clip_id"] for r in train[:40]}
    corpus = tmp_path / "corpus" / "metadata"
    corpus.mkdir(parents=True)
    with open(corpus / "train_clean.tsv", "w", encoding="utf-8", newline="") as f:
        f.write("path\tlanguage\tspeaker_id\tPRR\tlength\tsentence\n")
        for r in train:
            if r["clip_id"] not in dropped:
                session = clean.clip_times(r["clip_id"])[0]
                f.write(f"{session}/{r['clip_id']}.mp3\teu\t7\t99.00\t{r['duration']}\t{r['text'] or 'x'}\n")
    return {"out": out, "corpus": tmp_path / "corpus", "train": train, "dev": dev, "test": test,
            "expected": expected, "authors_dropped": dropped}


def run_clean(repo: dict) -> None:
    clean.main(["--out-root", str(repo["out"]), "--local-repo", str(repo["corpus"])])


def cleaned(repo: dict, name: str) -> list[dict]:
    return clean.read_csv(repo["out"] / "data" / "cleaned" / name)


def test_normalise():
    assert clean.normalise("  Eskerrik   ASKO, jauna!  ") == "eskerrik asko jauna"
    assert clean.normalise("ñ") == "ñ"  # NFC
    assert clean.clip_times("12-077_20220211_01_382.04_392.24") == ("12-077_20220211_01", 382.04, 392.24)


def test_judge_ignores_digits_and_symbols():
    assert clean.judge_text("Bozketa: 75 alde, 0 kontra (%100)!") == "bozketa alde kontra"
    assert clean.has_digit("2022 urtean") and not clean.has_digit("bi mila hogeita bi")


def test_speaking_rate_bounds_ignore_one_odd_dev_clip(cleaning_repo):
    dev = list(cleaning_repo["dev"])
    odd = _row("12-077_20220211_01", 9000.0, "bat eta bat", duration=9.5)  # 1 char per second
    th_without = clean.timing_thresholds(dev, None)
    th_with = clean.timing_thresholds(dev + [odd], None)
    assert abs(th_with["rate_min"] - th_without["rate_min"]) < 0.5


def test_report_splits_judge_removals_by_digits(cleaning_repo):
    run_clean(cleaning_repo)
    report = (cleaning_repo["out"] / "data" / "cleaned" / "cleaning_report.md").read_text(encoding="utf-8")
    assert "| with digits |" in report and "| without digits |" in report


def test_each_rule_removes_exactly_the_planted_clips(cleaning_repo):
    run_clean(cleaning_repo)
    log = {r["clip_id"]: r["rule"] for r in cleaned(cleaning_repo, "removal_log.csv")}
    assert log == cleaning_repo["expected"]
    assert set(log.values()) == set(clean.RULE_NUMBER)  # every rule is exercised
    assert list(cleaned(cleaning_repo, "removal_log.csv")[0]) == ["clip_id", "rule", "value", "threshold"]


def test_rules_keep_three_copies_and_the_earlier_overlapping_clip(cleaning_repo):
    train = cleaning_repo["train"]
    removed = clean.find_repeats(train)
    formula = [r["clip_id"] for r in train if r["text"] == "egun on guztioi lehendakari andrea"]
    assert [c in removed for c in formula] == [False, False, False, True, True]


def test_judge_threshold_separates_broken_from_untouched(cleaning_repo):
    judge_dev = {r["clip_id"]: r for r in cleaned(cleaning_repo, "judge_dev.csv")}
    c = clean.calibrate_cer(cleaning_repo["dev"], judge_dev)
    assert c["false_removal_rate"] == 0 and c["catch_rate"] > 0.95
    assert c["median_cer_untouched"] < c["threshold"] < 0.5
    assert all(rate > 0.9 for rate in c["catch_rate_by_kind"].values())


def test_cleaned_sets_have_equal_hours_and_come_from_the_sample(cleaning_repo):
    run_clean(cleaning_repo)
    sample = {r["clip_id"]: r for r in cleaning_repo["train"]}
    removed = {r["clip_id"] for r in cleaned(cleaning_repo, "removal_log.csv")}
    sets = {name: cleaned(cleaning_repo, f"train_{name}.csv") for name in ("ours", "authors", "rawmatched")}
    for rows in sets.values():
        assert all(r == sample[r["clip_id"]] for r in rows)  # rows unchanged, transcripts never rewritten
    assert not {r["clip_id"] for r in sets["ours"]} & removed
    assert not {r["clip_id"] for r in sets["authors"]} & cleaning_repo["authors_dropped"]
    totals = [clean.hours(rows) * 3600 for rows in sets.values()]
    assert max(totals) - min(totals) < 10  # within one clip


def test_cleaned_train_shares_no_transcript_with_dev_or_test(cleaning_repo):
    run_clean(cleaning_repo)
    held_out = {clean.normalise(r["text"]) for r in cleaning_repo["dev"] + cleaning_repo["test"]}
    assert not {clean.normalise(r["text"]) for r in cleaned(cleaning_repo, "train_ours.csv")} & held_out


def test_clean_twice_gives_identical_files_and_leaves_manifests_alone(cleaning_repo):
    before = tree(cleaning_repo["out"] / "data" / "raw")
    run_clean(cleaning_repo)
    first = tree(cleaning_repo["out"])
    run_clean(cleaning_repo)
    assert tree(cleaning_repo["out"]) == first
    assert tree(cleaning_repo["out"] / "data" / "raw") == before  # dev and test are never touched


def test_without_judge_outputs_nothing_is_written(cleaning_repo, capsys):
    clean.judge_path(cleaning_repo["out"], "train").unlink()
    run_clean(cleaning_repo)
    assert not (cleaning_repo["out"] / "data" / "cleaned" / "train_ours.csv").exists()
    assert "Preview of rules 1-4" in capsys.readouterr().out


# --- the real cleaned data in this repository --------------------------------------------------

CLEANED = REPO / "data" / "cleaned"
needs_cleaned_data = pytest.mark.skipif(
    not (CLEANED / "train_ours.csv").is_file(), reason="cleaned manifests are not in the repo yet"
)


@needs_cleaned_data
def test_real_cleaned_train_shares_nothing_with_dev_or_test():
    held_out = [r for name in ("dev.csv", "test.csv") for r in manifest(REPO, name)]
    for name in ("train_ours.csv", "train_authors.csv", "train_rawmatched.csv"):
        rows = clean.read_csv(CLEANED / name)
        assert not {r["clip_id"] for r in rows} & {r["clip_id"] for r in held_out}
    ours = {clean.normalise(r["text"]) for r in clean.read_csv(CLEANED / "train_ours.csv")}
    assert not ours & {clean.normalise(r["text"]) for r in held_out}


@needs_cleaned_data
def test_real_cleaned_sets_are_size_matched_subsets_of_the_sample():
    sample = {r["clip_id"]: r for r in manifest(REPO, "train_sample.csv")}
    removed = {r["clip_id"] for r in clean.read_csv(CLEANED / "removal_log.csv")}
    totals = []
    for name in ("train_ours.csv", "train_authors.csv", "train_rawmatched.csv"):
        rows = clean.read_csv(CLEANED / name)
        assert all(r == sample[r["clip_id"]] for r in rows)
        totals.append(clean.hours(rows) * 3600)
    assert max(totals) - min(totals) < 10
    assert not {r["clip_id"] for r in clean.read_csv(CLEANED / "train_ours.csv")} & removed
