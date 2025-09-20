import sys
import types
import unittest
from dataclasses import dataclass

try:  # pragma: no cover - exercised implicitly when dependencies are present
    from utils.llm_generator import WordClue
except Exception:  # pragma: no cover - fallback when optional deps are missing
    fake_llm_generator = types.ModuleType("utils.llm_generator")

    @dataclass(slots=True)
    class WordClue:  # type: ignore[redefinition]
        word: str
        clue: str
        direction_preference: str | None = None

    fake_llm_generator.WordClue = WordClue
    sys.modules["utils.llm_generator"] = fake_llm_generator

from utils.validators import (
    LengthValidationError,
    get_last_validation_issues,
    validate_word_list,
)


class ValidatorMaxLengthTests(unittest.TestCase):
    def test_accepts_fifteen_letter_words_for_supported_languages(self) -> None:
        cases = {
            "en": "abcdefghijklmno",
            "ru": "о" * 15,
            "it": "àèéìòùabcdefghi",
            "es": "áéíóúüñabcdeñop",
        }

        for language, word in cases.items():
            with self.subTest(language=language):
                accepted = validate_word_list(language, [WordClue(word=word, clue="test clue")])
                self.assertEqual(1, len(accepted))
                self.assertEqual(word.upper(), accepted[0].word)
                self.assertEqual([], get_last_validation_issues())

    def test_rejects_words_longer_than_fifteen_characters(self) -> None:
        word = "abcdefghijklmnop"  # 16 characters
        accepted = validate_word_list("en", [WordClue(word=word, clue="test clue")])

        self.assertEqual([], accepted)
        issues = get_last_validation_issues()
        self.assertEqual(1, len(issues))
        self.assertIsInstance(issues[0].error, LengthValidationError)
        self.assertIn("3 and 15", str(issues[0].error))


if __name__ == "__main__":
    unittest.main()
