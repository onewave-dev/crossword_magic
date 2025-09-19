"""LLM powered generator for crossword word clues."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Iterable

from langchain_core.output_parsers import OutputParserException, PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from utils.validators import validate_word_list

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WordClue:
    """Representation of a crossword word with its clue and desired direction."""

    word: str
    clue: str
    direction_preference: str | None = None


class _WordClueSchema(BaseModel):
    word: str = Field(..., description="Crossword answer word without spaces")
    clue: str = Field(..., description="Question or hint for the player")
    direction_preference: str | None = Field(
        None,
        description="Preferred placement direction (across, down or any)",
    )


class _ClueListSchema(BaseModel):
    clues: list[_WordClueSchema] = Field(..., description="Collection of generated clues")


_PARSER = PydanticOutputParser(pydantic_object=_ClueListSchema)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant that prepares crossword material. "
            "Always respond with a strict JSON object following the provided schema. "
            "Avoid commentary and do not wrap the response in code fences.\n"
            "{format_instructions}",
        ),
        (
            "human",
            "Theme: {theme}\n"
            "Language: {language}\n"
            "Generate at least 60 unique single-word crossword answers that fit the theme. "
            "Each answer must be between 3 and 12 characters long, consist only of letters of the target language, "
            "and must not contain spaces, hyphens or digits. Provide a concise clue for every word and an optional "
            "direction preference (across, down, any).",
        ),
    ]
).partial(format_instructions=_PARSER.get_format_instructions())

_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_MAX_ATTEMPTS = 3


def _get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not configured")

    return ChatOpenAI(
        api_key=api_key,
        model=_DEFAULT_MODEL,
        temperature=0.2,
    )


def _strip_code_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```") and content.endswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    if content.startswith("```json"):
        return content[len("```json") :].strip("`\n ")
    return content


def _parse_response(raw_content: str) -> _ClueListSchema:
    try:
        return _PARSER.parse(raw_content)
    except OutputParserException as exc:
        logger.warning("Primary parsing failed: %s", exc)

    cleaned = _strip_code_fence(raw_content)
    try:
        loaded = json.loads(cleaned)
    except json.JSONDecodeError as json_exc:
        logger.error("JSON decoding failed: %s", json_exc)
        raise ValueError("Unable to parse LLM response as JSON") from json_exc

    try:
        return _ClueListSchema.parse_obj(loaded)
    except ValidationError as validation_exc:
        logger.error("Parsed JSON does not match expected schema: %s", validation_exc)
        raise ValueError("Invalid JSON structure from LLM") from validation_exc


def _normalise_word(word: str) -> str:
    return "".join(word.split()).upper()


def _normalise_clue(clue: str) -> str:
    return " ".join(clue.split()).strip()


def _normalise_direction(direction: str | None) -> str | None:
    if not direction:
        return None
    cleaned = direction.strip().lower()
    if cleaned in {"across", "down", "any"}:
        return cleaned
    return None


def _normalise_payload(clues: Iterable[_WordClueSchema]) -> list[WordClue]:
    normalised: list[WordClue] = []
    for clue_schema in clues:
        word = _normalise_word(clue_schema.word)
        if not word:
            continue
        normalised.append(
            WordClue(
                word=word,
                clue=_normalise_clue(clue_schema.clue),
                direction_preference=_normalise_direction(clue_schema.direction_preference),
            )
        )
    return normalised


def generate_clues(theme: str, language: str) -> list[WordClue]:
    """Generate crossword clues for the provided theme and language."""

    if not theme or not language:
        raise ValueError("Theme and language must be provided")

    llm = _get_llm()
    best_partial: list[WordClue] = []

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            messages = _PROMPT.format_messages(theme=theme, language=language)
            logger.debug("Requesting clues from LLM (attempt %s)", attempt)
            response = llm.invoke(messages)
            raw_content = response.content if hasattr(response, "content") else str(response)
            parsed = _parse_response(raw_content)
            normalised = _normalise_payload(parsed.clues)
            validated = validate_word_list(language, normalised)

            if len(validated) >= 60:
                logger.info("Generated %s validated clues", len(validated))
                return validated

            if len(validated) > len(best_partial):
                best_partial = validated
            logger.warning(
                "Received only %s valid clues on attempt %s (expected >= 60)",
                len(validated),
                attempt,
            )
        except Exception:  # noqa: BLE001 - logging and retry strategy
            logger.exception("LLM generation attempt %s failed", attempt)
            if attempt == _MAX_ATTEMPTS and best_partial:
                logger.warning("Returning best partial result with %s clues", len(best_partial))
                return best_partial
            if attempt == _MAX_ATTEMPTS:
                raise

    if best_partial:
        logger.warning("Returning partial validated clues (%s) after retries", len(best_partial))
        return best_partial

    raise RuntimeError("Failed to generate clues with the language model")
