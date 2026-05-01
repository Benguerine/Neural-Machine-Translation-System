"""

"""

import pytest
from unittest.mock import patch, MagicMock

from ai_translator.translate import translate_text, batch_translate



def _make_mock_get_model(decoded: str = "Mocked translation"):
    """Return a patch-ready mock of get_model() for a given decoded string."""
    mock_model     = MagicMock()
    mock_tokenizer = MagicMock()

    # tokenizer() returns a dict-like object that supports .to(device)
    mock_inputs = MagicMock()
    mock_inputs.__iter__ = MagicMock(return_value=iter([]))
    mock_tokenizer.return_value = mock_inputs
    mock_inputs.to.return_value = mock_inputs

    # model.generate() returns a tensor-like list
    mock_model.generate.return_value = [[1, 2, 3]]

    # batch_decode returns the decoded string(s)
    mock_tokenizer.batch_decode.return_value = [decoded]

    
    

    return MagicMock(return_value=(mock_model, mock_tokenizer))




class TestTranslateText:

    def test_english_to_french_returns_string(self):
        
        with patch(
            "ai_translator.translate.get_model",
            _make_mock_get_model("Bonjour Mohammed, comment allez-vous?"),
        ):
            result = translate_text("Hello Mohammed, how are you?", "English", "French")

        assert isinstance(result, str)
        assert len(result) > 0
        assert not result.startswith("Translation error")

    def test_english_to_arabic_returns_string(self):
        
        with patch("ai_translator.translate.get_model", _make_mock_get_model("اسمي محمد.")):
            result = translate_text("My name is Mohammed.", "English", "Arabic")

        assert isinstance(result, str)
        assert len(result) > 0
        assert not result.startswith("Translation error")

    def test_empty_string_returns_empty(self):
        
        assert translate_text("", "English", "French") == ""

    def test_whitespace_only_returns_empty(self):
        
        assert translate_text("   ", "English", "French") == ""

    def test_unknown_language_does_not_crash(self):
        
        with patch("ai_translator.translate.get_model", _make_mock_get_model("Bonjour")):
            result = translate_text("Hello", "Klingon", "French")

        assert isinstance(result, str)
        assert not result.startswith("Translation error")


class TestBatchTranslate:

    def test_empty_list_returns_empty(self):
        assert batch_translate([], "English", "French") == ""

    def test_whitespace_only_list_returns_empty(self):
        assert batch_translate(["  ", "\t"], "English", "French") == ""

    def test_batch_returns_numbered_lines(self):
        mock = _make_mock_get_model("Bonjour")
        # Override batch_decode to return two items
        mock_model, mock_tokenizer = mock.return_value
        mock_tokenizer.batch_decode.return_value = ["Bonjour", "Au revoir"]

        with patch("ai_translator.translate.get_model", mock):
            result = batch_translate(["Hello", "Goodbye"], "English", "French")

        lines = result.strip().splitlines()
        assert lines[0].startswith("1.")
        assert lines[1].startswith("2.")




@pytest.mark.integration
class TestTranslateTextIntegration:

    def test_english_to_french_real_model(self):
        result = translate_text("Hello Mohammed, how are you?", "English", "French")
        assert isinstance(result, str)
        assert len(result) > 0
        assert not result.startswith("Translation error")

    def test_english_to_arabic_real_model(self):
        result = translate_text("My name is Mohammed.", "English", "Arabic")
        assert isinstance(result, str)
        assert len(result) > 0
        assert not result.startswith("Translation error")
