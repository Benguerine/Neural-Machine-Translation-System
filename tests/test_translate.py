"""
Integration tests for the translation functionality
"""



import pytest
from ai_translator.translate import translate_text, batch_translate
from unittest.mock import patch, MagicMock


class TestTranslateText:

    

    def test_english_to_french_returns_string(self):
        with patch("ai_translator.translate.model") as mock_model:
            mock_model.generate.return_value = MagicMock()
            with patch("ai_translator.translate.tokenizer") as mock_tok:
                mock_tok.decode.return_value = "Bonjour Mohammed, comment allez-vous?"
            
        result = translate_text("Hello Mohammed, how are you?", "English", "French")
            
        assert isinstance(result, str)
        assert len(result) > 0
        assert not result.startswith("Translation error")

    import pytest

    @pytest.mark.integration
    def test_english_to_french_real_model(self):
        result = translate_text("Hello Mohammed, how are you?", "English", "French")
        assert isinstance(result, str)
        assert len(result) > 0
        assert not result.startswith("Translation error")

    def test_english_to_arabic(self):
        result = translate_text("My name is Mohammed.", "English", "Arabic")
        assert isinstance(result, str)
        assert len(result) > 0
        assert not result.startswith("Translation error")

    def test_empty_string(self):
        assert translate_text("", "English", "French") == ""

    def test_whitespace_only_returns_empty(self):
        assert translate_text("   ", "English", "French") == ""

    def test_unknown_language_does_not_crash(self):
        # Falls back to default FLORES code, should not raise
        result = translate_text("Hello", "Klingon", "French")
        assert isinstance(result, str)
        assert not result.startswith("Translation error")