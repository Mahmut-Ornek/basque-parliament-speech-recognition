"""Scoring normalisation: the policy in report/scoring_policy.md, as code.

    from normalize import normalize
    normalize("Kaixo, Legebiltzarkideok!")   # -> "kaixo legebiltzarkideok"

Applied to the reference and to the model output (the hypothesis) before WER and CER are
computed. Prediction files keep the raw model output; only the scorer normalises.

Steps, in this order:
  1. Unicode NFC, so that a letter with an accent is one code point ("á", not "a" + accent).
  2. Lowercase.
  3. Every punctuation character (Unicode category P: . , ; : ! ? ¿ ¡ ' " - ( ) and the like)
     becomes a space. A space and not nothing, so "lehendakari-ordea" and "lehendakari ordea"
     are scored the same way and two words joined by a comma do not merge into one.
  4. Runs of whitespace become one space; leading and trailing spaces are removed.

Nothing else changes. In particular:
  * Letters are kept as they are, accents included (á é í ó ú ñ ü ç): they are part of the
    Basque and Spanish spelling, and the references use them.
  * Digits are kept. The references write numbers as words ("bi mila hogeita bi"); a model that
    writes "2022" is counted as wrong on that number. Nobody in the group can check Basque number
    words, so we do not rewrite them. Number errors are reported as their own error type in
    report/error_analysis.md.
  * Symbols that are not punctuation in Unicode (€ + = and the like) are kept and count as
    errors. "%" is punctuation in Unicode (category Po), so it becomes a space.

The same steps are used by clean.normalise, which picks the fine-tuning checkpoints on dev;
tests/test_normalize.py checks that the two give the same text.
"""
from __future__ import annotations

import unicodedata


def normalize(text: str) -> str:
    """Return `text` normalised for scoring (see the module docstring)."""
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return " ".join(text.split())
