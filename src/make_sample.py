"""Build the raw data for the project from the Basque Parliament Speech Corpus 1.0.

It writes, under the repository root:

    data/raw/audio/{train,dev,test}/<NNN>/<clip_id>.mp3   clips, bytes unchanged from the corpus
    data/raw/manifests/{train_sample,dev,test}.csv        clip_id, path, text, duration, speaker_id

Train: Basque clips in a seeded random order (seed 42) until the requested hours are reached.
The order is nested: the clips of a 12-hour sample are the first rows of the 20-hour sample.
Dev and test: every Basque clip of the official splits.

The corpus has no index that says which clip is in which archive, so the script lists the
members of each archive it uses. `--archives all` samples from the whole train split
(11 archives, about 16.5 GB, downloaded once into the HuggingFace cache). `--archives N`
samples only from N archives chosen with the same seed, and `--archives 0,5,10` from exactly
those archives.

Usage:
    python src/make_sample.py --hours 20 --archives all
    python src/make_sample.py --hours 12 --archives all --prune    # shrink an existing sample
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

REPO_ID = "gttsehu/basque_parliament_1"
REVISION = "46208b3001c500aa53ddb5f4a9da81493bbb8067"  # dataset commit, fixed for reproducibility
LANGUAGE = "eu"
SEED = 42
TRAIN_SHARDS = 11
CLIPS_PER_FOLDER = 500  # GitHub's web view cuts a folder off at 1,000 files
EXPECTED_CLIPS = {"dev": 1397, "test": 1521}
UNKNOWN_SPEAKER = "0"  # not a person: clips without an identified speaker

# The corpus quality score (PRR) is deliberately not in this list: it never leaves read_rows().
TSV_COLUMNS = ("path", "language", "speaker_id", "length", "sentence")
MANIFEST_COLUMNS = ("clip_id", "path", "text", "duration", "speaker_id")
MANIFEST_NAMES = {"train": "train_sample.csv", "dev": "dev.csv", "test": "test.csv"}


def fetch(filename: str, local_repo: str | None, cache_dir: str | None) -> Path:
    """Return a local path to one file of the dataset repo (downloaded once, then cached)."""
    if local_repo is not None:
        path = Path(local_repo) / filename
        if not path.is_file():
            raise SystemExit(f"{path} does not exist")
        return path
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(REPO_ID, filename, repo_type="dataset", revision=REVISION, cache_dir=cache_dir)
    )


def read_rows(tsv_path: Path) -> list[dict]:
    """Read one metadata file and keep the Basque rows, in file order."""
    with open(tsv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        missing = set(TSV_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"{tsv_path.name}: columns {sorted(missing)} not found. Found: {reader.fieldnames}"
            )
        rows, seen = [], set()
        for row in reader:
            if row["language"] != LANGUAGE or row["path"] in seen:
                continue
            seen.add(row["path"])
            rows.append({c: row[c] for c in TSV_COLUMNS})
    return rows


def assign_clip_ids(rows_by_split: dict[str, list[dict]]) -> None:
    """Add a clip_id to every row: the file name without extension, or the whole path if
    file names are not unique across the Basque clips of all splits."""
    paths = [r["path"] for rows in rows_by_split.values() for r in rows]
    stems = [PurePosixPath(p).stem for p in paths]
    stems_unique = len(set(stems)) == len(stems)
    for rows in rows_by_split.values():
        for r in rows:
            p = PurePosixPath(r["path"])
            r["clip_id"] = p.stem if stems_unique else str(p.with_suffix("")).replace("/", "__")


def sample_order(train_rows: list[dict]) -> list[dict]:
    """Seeded random order of the train rows. It does not depend on the order of the file."""
    rows = sorted(train_rows, key=lambda r: r["path"])
    random.Random(SEED).shuffle(rows)
    return rows


def take_hours(rows: list[dict], hours: float) -> list[dict]:
    """The shortest prefix of `rows` that reaches `hours` of audio."""
    target, total, taken = hours * 3600.0, 0.0, []
    for r in rows:
        if total >= target:
            break
        taken.append(r)
        total += float(r["length"])
    if total < target:
        raise SystemExit(
            f"Only {total / 3600:.2f} h of Basque audio in the chosen archives, {hours} h requested. "
            "Use more archives."
        )
    return taken


def choose_archives(spec: str, n_shards: int) -> list[int]:
    if spec == "all":
        return list(range(n_shards))
    if "," in spec:
        ids = sorted({int(x) for x in spec.split(",") if x.strip()})
        if not ids or ids[0] < 0 or ids[-1] >= n_shards:
            raise SystemExit(f"--archives: archive numbers go from 0 to {n_shards - 1}")
        return ids
    return sorted(random.Random(SEED).sample(range(n_shards), int(spec)))


def session_and_year(path: str) -> tuple[str, str]:
    """Corpus paths look like 12-077_20220211_01/12-077_20220211_01_382.04_392.24.mp3:
    the folder is the session and it carries the date."""
    session = PurePosixPath(path).parts[0]
    match = re.search(r"_(\d{4})\d{4}(?:_|$)", session)
    return session, match.group(1) if match else "unknown"


def count_speakers(rows: list[dict]) -> tuple[int, int]:
    """Number of identified speakers, and number of clips whose speaker is unknown."""
    speakers = {r["speaker_id"] for r in rows} - {UNKNOWN_SPEAKER}
    return len(speakers), sum(r["speaker_id"] == UNKNOWN_SPEAKER for r in rows)


def member_name(name: str) -> str:
    return name[2:] if name.startswith("./") else name


def list_members(tar_path: Path) -> list[str]:
    with tarfile.open(tar_path, "r:") as tar:
        return [member_name(m.name) for m in tar if m.isfile()]


def extract(tar_path: Path, wanted: dict[str, Path]) -> None:
    """Copy the wanted members of one archive to their destination, bytes unchanged."""
    with tarfile.open(tar_path, "r:") as tar:
        for m in tar:
            dest = wanted.get(member_name(m.name))
            if dest is None or not m.isfile():
                continue
            if dest.is_file() and dest.stat().st_size == m.size:
                continue  # already extracted by an earlier run
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tar.extractfile(m).read())


def build_split(split: str, rows: list[dict], tar_paths: list[Path], out_root: Path, prune: bool) -> dict:
    """Extract the clips of one split, write its manifest, and return summary numbers."""
    audio_dir = out_root / "data" / "raw" / "audio" / split
    wanted, manifest = {}, []
    for i, r in enumerate(rows):
        suffix = PurePosixPath(r["path"]).suffix
        rel = f"data/raw/audio/{split}/{i // CLIPS_PER_FOLDER:03d}/{r['clip_id']}{suffix}"
        wanted[r["path"]] = out_root / rel
        manifest.append((r["clip_id"], rel, r["sentence"], r["length"], r["speaker_id"]))

    for tar_path in tar_paths:
        extract(tar_path, wanted)
    missing = [p for p, dest in wanted.items() if not dest.is_file()]
    if missing:
        raise SystemExit(f"{split}: {len(missing)} clips not found in the archives, e.g. {missing[:3]}")

    manifest_path = out_root / "data" / "raw" / "manifests" / MANIFEST_NAMES[split]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerows(manifest)

    # Clips left over from a larger, earlier sample are not in the manifest any more.
    keep = set(wanted.values())
    extra = [p for p in audio_dir.rglob("*") if p.is_file() and p.name != ".gitkeep" and p not in keep]
    if extra and prune:
        for p in extra:
            p.unlink()
        for d in sorted((d for d in audio_dir.rglob("*") if d.is_dir()), reverse=True):
            if not any(d.iterdir()):
                d.rmdir()
    elif extra:
        print(f"WARNING {split}: {len(extra)} audio files are not in the manifest. Rerun with --prune.")

    return {
        "clips": len(rows),
        "hours": sum(float(r["length"]) for r in rows) / 3600,
        "speakers": count_speakers(rows)[0],
        "unknown": count_speakers(rows)[1],
        "megabytes": sum(p.stat().st_size for p in keep) / 1e6,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=20.0, help="hours of Basque train audio (default 20)")
    ap.add_argument("--archives", default="all", help="'all', a number N (N archives chosen with the seed), or a list such as 0,5,10")
    ap.add_argument("--prune", action="store_true", help="delete audio files that are not in the new manifests")
    ap.add_argument("--out-root", default=str(Path(__file__).resolve().parents[1]), help="repository root")
    ap.add_argument("--cache-dir", default=None, help="HuggingFace cache folder for the downloads")
    ap.add_argument("--local-repo", default=None, help="read the corpus from a local folder (used by the tests)")
    ap.add_argument("--train-shards", type=int, default=TRAIN_SHARDS, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    out_root = Path(args.out_root)

    def get(filename: str) -> Path:
        return fetch(filename, args.local_repo, args.cache_dir)

    rows = {split: read_rows(get(f"metadata/{split}.tsv")) for split in ("train", "dev", "test")}
    assign_clip_ids(rows)
    for split, expected in EXPECTED_CLIPS.items():
        if args.local_repo is None and len(rows[split]) != expected:
            print(f"WARNING {split}: {len(rows[split])} Basque clips, expected {expected}")

    # Train: find out which clips the chosen archives hold, then take the first clips of the
    # seeded order that are in them.
    archives = choose_archives(args.archives, args.train_shards)
    train_tars = {i: get(f"audio/train_{i}.tar") for i in archives}
    archive_of = {name: i for i, tar in train_tars.items() for name in list_members(tar)}
    candidates = [r for r in sample_order(rows["train"]) if r["path"] in archive_of]
    if not candidates:
        raise SystemExit(
            "No train.tsv path matches an archive member. "
            f"Archive members look like {list(archive_of)[:3]}, paths like {[r['path'] for r in rows['train'][:3]]}"
        )
    sample = take_hours(candidates, args.hours)

    summary = {"train": build_split("train", sample, list(train_tars.values()), out_root, args.prune)}
    for split in ("dev", "test"):
        summary[split] = build_split(split, rows[split], [get(f"audio/{split}_0.tar")], out_root, args.prune)

    print(f"\nDataset commit {REVISION}, seed {SEED}, train archives used: {archives}")
    print(f"Basque train clips in these archives: {len(candidates)} of {len(rows['train'])}")
    per_archive = Counter(archive_of[r["path"]] for r in sample)
    print("Sample clips per archive: " + ", ".join(f"train_{i}: {per_archive[i]}" for i in archives))
    print("\n| Split | Clips | Hours | Speakers | Clips with unknown speaker | Size (MB) |\n|---|---|---|---|---|---|")
    for split, s in summary.items():
        print(f"| {split} | {s['clips']} | {s['hours']:.2f} | {s['speakers']} | {s['unknown']} | {s['megabytes']:.1f} |")
    print(f"| total | | | | | {sum(s['megabytes'] for s in summary.values()):.1f} |")
    years = Counter(session_and_year(r["path"])[1] for r in sample)
    print("\nTrain sample by year: " + ", ".join(f"{y}: {n}" for y, n in sorted(years.items())))
    print(f"Sessions in the train sample: {len({session_and_year(r['path'])[0] for r in sample})}")
    for h in (10, 12):
        if h < args.hours:
            print(f"The first {h} hours of the sample are its first {len(take_hours(sample, h))} clips.")


if __name__ == "__main__":
    main()
