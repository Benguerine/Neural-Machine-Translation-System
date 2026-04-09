

import re
import pytest
from ai_translator.languages import (
    LANGUAGE_CODES,
    SPEECH_LANG_CODES,
    SUPPORTED_LANGUAGES,
    get_flores_code,
    get_speech_code,
)

FLORES_PATTERN = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")

class TestLanguageCodes:

    def test_all_flores_codes_have_correct_format(self):
        for lang, code in LANGUAGE_CODES.items():
            assert FLORES_PATTERN.match(code), \
                f"{lang} → '{code}' does not match FLORES-200 format (e.g. eng_Latn)"
    
    def test_supported_languages_matches_dict_keys(self):
        assert SUPPORTED_LANGUAGES == list(LANGUAGE_CODES.keys())

    def test_contains_core_languages(self):
        for lang in ["English", "French", "Arabic", "Spanish", "German"]:
            assert lang in LANGUAGE_CODES, f"Missing: {lang}"
    minimum_lang = 50
    def test_has_minimum_languages(self):
        assert len(LANGUAGE_CODES) >= self.minimum_lang

    def test_no_duplicate_flores_codes(self):
        codes = list(LANGUAGE_CODES.values())
        assert len(codes) == len(set(codes)), "Duplicate FLORES-200 codes found"

    def test_english_code_is_correct(self):
        assert LANGUAGE_CODES["English"] == "eng_Latn"

    def test_arabic_code_is_correct(self):
        assert LANGUAGE_CODES["Arabic"] == "arb_Arab"


class TestGetFloresCode:

    def test_known_language(self):
        assert get_flores_code("English") == "eng_Latn"
        assert get_flores_code("French")  == "fra_Latn"
        assert get_flores_code("Arabic")  == "arb_Arab"

    def test_unknown_language_returns_default_fallback(self):
        assert get_flores_code("Klingon") == "eng_Latn"

    def test_unknown_language_returns_custom_fallback(self):
        assert get_flores_code("Klingon", "fra_Latn") == "fra_Latn"


class TestGetSpeechCode:

    def test_known_language(self):
        assert get_speech_code("English") == "en-US"
        assert get_speech_code("French")  == "fr-FR"
        assert get_speech_code("Arabic")  == "ar-SA"

    def test_language_without_speech_support_returns_fallback(self):
        # Swahili has FLORES code but no Google SR support
        assert get_speech_code("Swahili") == "en-US"

    def test_custom_fallback(self):
        assert get_speech_code("Swahili", "fr-FR") == "fr-FR"

            