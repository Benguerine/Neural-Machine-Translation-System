"""
Public API of the NMT package.

Usage:
    from src import translate_text, calculate_bleu
    from src import SUPPORTED_LANGUAGES
"""

from .languages import LANGUAGE_CODES, SUPPORTED_LANGUAGES, get_flores_code, get_speech_code
from .translate import translate_text, batch_translate
from .speech    import speech_to_text
from .evaluate  import calculate_bleu, corpus_bleu

__all__ = [
    "LANGUAGE_CODES",
    "SUPPORTED_LANGUAGES",
    "get_flores_code",
    "get_speech_code",
    "translate_text",
    "batch_translate",
    "speech_to_text",
    "calculate_bleu",
    "corpus_bleu",
]
