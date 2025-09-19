"""Validation utilities for crossword generation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Iterable, Sequence, TYPE_CHECKING

from utils.logging_config import get_logger

logger = get_logger("validators")

if TYPE_CHECKING:  # pragma: no cover - used only for typing
    from utils.llm_generator import WordClue

__all__ = [
    "CharacterValidationError",
    "DuplicateWordError",
    "LengthValidationError",
    "WordValidationError",
    "WordValidationIssue",
    "get_last_validation_issues",
    "validate_word_clues",
    "validate_word_list",
]


class WordValidationError(Exception):
    """Base class for validation errors describing why a word is rejected."""

    code = "invalid"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class CharacterValidationError(WordValidationError):
    """Raised when a word contains unsupported characters."""

    code = "characters"


class LengthValidationError(WordValidationError):
    """Raised when a word is shorter or longer than allowed."""

    code = "length"


class DuplicateWordError(WordValidationError):
    """Raised when the word duplicates an already accepted entry."""

    code = "duplicate"


@dataclass(slots=True)
class WordValidationIssue:
    """Represents a rejected word alongside the reason."""

    clue: "WordClue"
    word: str
    canonical_word: str
    error: WordValidationError


_LANGUAGE_PATTERNS: dict[str, re.Pattern[str]] = {
    "en": re.compile(r"^[A-Z]+$"),
    "ru": re.compile(r"^[А-ЯЁ]+$"),
    "it": re.compile(r"^[A-ZÀÈÉÌÒÙ]+$"),
    "es": re.compile(r"^[A-ZÁÉÍÑÓÚÜ]+$"),
}

_GENERIC_LETTERS = re.compile(r"^[^\W\d_]+$", re.UNICODE)

_MIN_WORD_LENGTH = 3
_MAX_WORD_LENGTH = 12

_LAST_VALIDATION_ISSUES: list[WordValidationIssue] = []


def _normalise_language(language: str) -> str:
    return language.lower()


def _pattern_for_language(language: str) -> re.Pattern[str]:
    normalised = _normalise_language(language)
    if normalised in _LANGUAGE_PATTERNS:
        return _LANGUAGE_PATTERNS[normalised]

    logger.debug("No dedicated alphabet pattern for %s, using generic unicode letters", language)
    return _GENERIC_LETTERS


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _canonicalise_word(word: str) -> str:
    """Normalise a word for duplicate checks (case, accents, apostrophes, ё/е)."""

    cleaned = _strip_accents(word)
    cleaned = cleaned.replace("’", "'").replace("`", "'").replace("´", "'")
    cleaned = cleaned.replace("Ё", "Е").replace("ё", "е")
    cleaned = cleaned.replace("'", "")
    return cleaned.upper()


def _prepare_word(word: str) -> str:
    return unicodedata.normalize("NFC", word or "").strip()


def _validate_length(word: str) -> None:
    if not (_MIN_WORD_LENGTH <= len(word) <= _MAX_WORD_LENGTH):
        raise LengthValidationError(
            f"Word length must be between {_MIN_WORD_LENGTH} and {_MAX_WORD_LENGTH} characters",
        )


def _validate_characters(word: str, pattern: re.Pattern[str]) -> None:
    if not pattern.fullmatch(word):
        raise CharacterValidationError(
            "Word must consist only of alphabetic characters for the selected language",
        )


def _normalise_for_output(word: str) -> str:
    return unicodedata.normalize("NFC", word).upper()


def get_last_validation_issues() -> list[WordValidationIssue]:
    """Return the list of issues collected during the last validation call."""

    return list(_LAST_VALIDATION_ISSUES)


def validate_word_list(
    language: str, words: Iterable["WordClue"], *, deduplicate: bool = True
) -> list["WordClue"]:
    """Validate crossword entries and return only the acceptable ones.

    Args:
        language: Target crossword language (determines alphabet rules).
        words: Iterable of :class:`WordClue` instances to validate.
        deduplicate: When ``True`` (default) repeated words are rejected.

    The function records detailed information about every rejected word and stores it
    in :func:`get_last_validation_issues` for diagnostic purposes.
    """

    try:
        from utils.llm_generator import WordClue  # local import to avoid circular dependency
    except ImportError:  # pragma: no cover - defensive guard
        logger.exception("Unable to import WordClue for validation")
        raise

    pattern = _pattern_for_language(language)
    accepted: list[WordClue] = []
    issues: list[WordValidationIssue] = []
    seen: set[str] = set()

    for clue in words:
        raw_word = _prepare_word(clue.word)
        canonical_word = _canonicalise_word(raw_word)

        try:
            _validate_length(raw_word)
            _validate_characters(_normalise_for_output(raw_word), pattern)
            if deduplicate and canonical_word in seen:
                raise DuplicateWordError("Word duplicates a previously accepted entry")
        except WordValidationError as exc:
            issue = WordValidationIssue(clue=clue, word=raw_word, canonical_word=canonical_word, error=exc)
            issues.append(issue)
            logger.debug("Rejected word %r (%s): %s", raw_word, exc.code, exc)
            continue

        if deduplicate:
            seen.add(canonical_word)
        accepted.append(replace(clue, word=_normalise_for_output(raw_word)))

    _LAST_VALIDATION_ISSUES.clear()
    _LAST_VALIDATION_ISSUES.extend(issues)

    if issues:
        logger.info(
            "Validation rejected %s words for language %s (accepted=%s)",
            len(issues),
            language,
            len(accepted),
        )
    else:
        logger.info("Validation accepted %s words for language %s", len(accepted), language)

    return accepted


def validate_word_clues(
    word_clues: Sequence["WordClue"], language: str, *, deduplicate: bool = True
) -> list["WordClue"]:
    """Backward compatible wrapper around :func:`validate_word_list`."""

    return validate_word_list(language, word_clues, deduplicate=deduplicate)
