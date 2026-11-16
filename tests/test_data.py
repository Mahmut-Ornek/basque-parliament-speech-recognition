"""Tests for the data pipeline.

The tests of `make_sample.py` run on a tiny synthetic corpus with the same layout as the real one
(`metadata/*.tsv` + `audio/*.tar`), so they need no download. The last tests check the real
manifests in `data/raw/manifests/` and are skipped until those files exist.
"""
import csv
import io
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
