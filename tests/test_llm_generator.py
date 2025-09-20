"""Tests for the LLM clue generation helper."""

from __future__ import annotations

import json
import string
import sys
import types
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

try:  # pragma: no cover - exercised when optional dependency is available
    from utils.llm_generator import WordClue, generate_clues
except ModuleNotFoundError as exc:  # pragma: no cover - fallback for optional deps
    if exc.name != "langchain_openai":
        raise

    fake_langchain_openai = types.ModuleType("langchain_openai")

    class ChatOpenAI:  # type: ignore[redefinition]
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401, ANN002, ANN003 - test stub
            """No-op stub used when langchain_openai is unavailable."""

        def invoke(self, messages):  # type: ignore[override]  # noqa: ANN001
            raise RuntimeError("ChatOpenAI stub should not be invoked in tests")

    fake_langchain_openai.ChatOpenAI = ChatOpenAI
    sys.modules["langchain_openai"] = fake_langchain_openai

    from utils.llm_generator import WordClue, generate_clues


@dataclass
class _DummyLLM:
    """Simple stub that returns a predefined JSON payload."""

    payload: str

    def invoke(self, messages):  # type: ignore[override]
        return SimpleNamespace(content=self.payload)


def _serialise_words(words: list[str]) -> str:
    entries = [
        {"word": word, "clue": f"Clue for {word}", "direction_preference": None}
        for word in words
    ]
    return json.dumps({"clues": entries})


def _alphabetical_words(count: int) -> list[str]:
    letters = string.ascii_uppercase
    results: list[str] = []
    for index in range(count):
        value = index
        suffix = []
        while True:
            suffix.append(letters[value % 26])
            value //= 26
            if value == 0:
                break
        results.append("theme" + "".join(suffix))
    return results


class GenerateCluesTests(unittest.TestCase):
    def test_generate_clues_returns_minimum_set(self) -> None:
        words = [
            "alpha",
            "bravo",
            "charlie",
            "delta",
            "echo",
            "foxtrot",
            "golf",
            "hotel",
            "india",
            "juliet",
        ]
        payload = _serialise_words(words)

        with patch("utils.llm_generator._get_llm", return_value=_DummyLLM(payload)):
            generated = generate_clues(theme="Navigation", language="en")

        self.assertEqual(10, len(generated))
        self.assertTrue(all(isinstance(item, WordClue) for item in generated))
        self.assertListEqual([word.upper() for word in words], [item.word for item in generated])

    def test_generate_clues_trims_to_maximum(self) -> None:
        words = _alphabetical_words(45)
        payload = _serialise_words(words)

        with patch("utils.llm_generator._get_llm", return_value=_DummyLLM(payload)):
            generated = generate_clues(theme="Navigation", language="en")

        self.assertEqual(40, len(generated))
        self.assertListEqual([word.upper() for word in words[:40]], [item.word for item in generated])


if __name__ == "__main__":
    unittest.main()