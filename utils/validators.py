"""Validation utilities for crossword generation."""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Sequence

logger = logging.getLogger(__name__)


_LANGUAGE_PATTERNS: dict[str, re.Pattern[str]] = {
    "en": re.compile(r"^[A-Z]{3,12}$"),
    "ru": re.compile(r"^[А-ЯЁ]{3,12}$"),
    "it": re.compile(r"^[A-ZÀÈÉÌÒÙ]{3,12}$"),
    "es": re.compile(r"^[A-ZÁÉÍÑÓÚÜ]{3,12}$"),
}


def _normalise_language(language: str) -> str:
    return language.lower()


def _pattern_for_language(language: str) -> re.Pattern[str]:
    normalised = _normalise_language(language)
    if normalised in _LANGUAGE_PATTERNS:
        return _LANGUAGE_PATTERNS[normalised]

    logger.debug("No dedicated alphabet pattern for %s, using generic unicode letters", language)
    return re.compile(r"^[^\W\d_]{3,12}$", re.UNICODE)


def validate_word_clues(
    word_clues: Sequence["WordClue"], language: str, *, deduplicate: bool = True
) -> list["WordClue"]:
    """Validate that generated clues meet the crossword requirements."""

    try:
        from utils.llm_generator import WordClue  # local import to avoid circular dependency
    except ImportError:  # pragma: no cover - defensive guard
        logger.exception("Unable to import WordClue for validation")
        raise

    pattern = _pattern_for_language(language)
    valid_clues: list[WordClue] = []
    seen: set[str] = set()

    for clue in word_clues:
        word = clue.word
        if not pattern.match(word):
            logger.debug("Skipping word %s – does not match language constraints", word)
            continue

        if deduplicate and word in seen:
            logger.debug("Skipping duplicated word %s", word)
            continue

        seen.add(word)
        valid_clues.append(replace(clue))

    return valid_clues
