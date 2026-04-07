

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

    def test_contains_core_Languages(self):
        for lang in ["English", "French", "Arabic", "Spanish", "German"]:
            assert lang in LANGUAGE_CODES, f"Missing: {lang}"
    
    def test_has_50_languages(self):
        assert len(LANGUAGE_CODES) == 50

    def test_no_duplicate_flores_codes(self):
        codes = list(LANGUAGE_CODES.values())
        assert len(codes) == len(set(codes)), "Duplicate FLORES-200 codes found"
            