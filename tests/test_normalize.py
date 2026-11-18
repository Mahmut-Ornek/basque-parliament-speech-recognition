"""Tests for src/normalize.py, the scoring policy as code (report/scoring_policy.md)."""
import csv
import sys
import unicodedata
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from normalize import normalize  # noqa: E402

DEV = REPO / "data" / "raw" / "manifests" / "dev.csv"
PREDICTIONS = REPO / "results" / "predictions"


def test_lowercase():
    assert normalize("Kaixo Legebiltzarkideok") == "kaixo legebiltzarkideok"


def test_punctuation_becomes_space():
    assert normalize("bai, eta ez.") == "bai eta ez"
    assert normalize("¿Qué tal? ¡Bien!") == "qué tal bien"
    assert normalize("«Eusko Jaurlaritza»") == "eusko jaurlaritza"


def test_hyphen_and_apostrophe_split_words():
    assert normalize("lehendakari-ordea") == "lehendakari ordea"
    assert normalize("d'Euskadi") == "d euskadi"


def test_whitespace_collapsed():
    assert normalize("  bai \t\n  ez  ") == "bai ez"


def test_letters_of_both_alphabets_kept():
    assert normalize("Ñ Ü Ç Á É Í Ó Ú") == "ñ ü ç á é í ó ú"


def test_nfc_composes_accents():
    decomposed = "a\u0301"  # "a" + combining acute accent
    out = normalize(decomposed)
    assert out == "á" and len(out) == 1
    assert unicodedata.is_normalized("NFC", out)


def test_digits_and_symbols_kept():
    # Numbers are not rewritten; they count as errors against the word form in the reference.
    assert normalize("2022an 15€ + 3") == "2022an 15€ + 3"
    # "%" is punctuation in Unicode (category Po), so it becomes a space like "." or ",".
    assert normalize("%15") == "15"


def test_empty_and_punctuation_only():
    assert normalize("") == ""
    assert normalize(" ... !? ") == ""


@pytest.mark.parametrize("text", ["Kaixo, Mundua!", "lehendakari-ordea", "a\u0301 B  c", "¿Qué?"])
def test_idempotent(text):
    assert normalize(normalize(text)) == normalize(text)


@pytest.mark.skipif(not DEV.exists(), reason="dev manifest not in the repo yet")
def test_dev_references_already_normalised():
    """The references are lowercase, without punctuation and with numbers as words, so
    normalising them changes nothing. This is why only normalised WER can be reported."""
    with open(DEV, encoding="utf-8", newline="") as f:
        refs = [r["text"] for r in csv.DictReader(f)]
    assert refs
    assert all(normalize(t) == t for t in refs)
    assert not any(c.isdigit() for t in refs for c in t)


def _texts_to_compare() -> list[str]:
    texts = []
    if DEV.exists():
        with open(DEV, encoding="utf-8", newline="") as f:
            texts += [r["text"] for r in csv.DictReader(f)]
    for path in sorted(PREDICTIONS.glob("*_dev.csv")):
        with open(path, encoding="utf-8", newline="") as f:
            texts += [r["hypothesis"] or "" for r in csv.DictReader(f)]
    return texts


def test_same_as_clean_normalise():
    """Checkpoints are chosen with clean.normalise on dev; final scores use normalize.py.
    Both must give the same text, on hand-made cases and on every dev reference and prediction."""
    clean = pytest.importorskip("clean")
    cases = ["Kaixo, Mundua!", "lehendakari-ordea", "a\u0301", "2022an 15€ %3", "  x  "] + _texts_to_compare()
    assert all(clean.normalise(t) == normalize(t) for t in cases)
