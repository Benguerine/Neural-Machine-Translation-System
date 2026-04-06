"""
Language mapping utilities for translation and speech recognition.

Components:
- LANGUAGE_CODES: Maps language names to FLORES-200 codes (used by Meta's NLLB model).
- SPEECH_LANG_CODES: Maps language names to BCP-47 codes for Google Speech Recognition.
- SUPPORTED_LANGUAGES: List of all supported language names (keys from LANGUAGE_CODES).
- get_flores_code: Returns the FLORES-200 code for a given language name.
- get_speech_code: Returns the Google SR BCP-47 code for a given language name.

FLORES-200 code format: <language>_<Script>
  e.g. eng_Latn = English in Latin script
       arb_Arab = Arabic in Arabic script
       zho_Hans = Chinese in Simplified Han script
"""

LANGUAGE_CODES: dict[str, str] = {
    "English":               "eng_Latn",
    "French":                "fra_Latn",
    "Arabic":                "arb_Arab",
    "Spanish":               "spa_Latn",
    "German":                "deu_Latn",
    "Chinese (Simplified)":  "zho_Hans",
    "Chinese (Traditional)": "zho_Hant",
    "Japanese":              "jpn_Jpan",
    "Korean":                "kor_Hang",
    "Russian":               "rus_Cyrl",
    "Portuguese":            "por_Latn",
    "Italian":               "ita_Latn",
    "Dutch":                 "nld_Latn",
    "Turkish":               "tur_Latn",
    "Polish":                "pol_Latn",
    "Hindi":                 "hin_Deva",
    "Bengali":               "ben_Beng",
    "Urdu":                  "urd_Arab",
    "Vietnamese":            "vie_Latn",
    "Thai":                  "tha_Thai",
    "Indonesian":            "ind_Latn",
    "Malay":                 "zsm_Latn",
    "Swahili":               "swh_Latn",
    "Greek":                 "ell_Grek",
    "Hebrew":                "heb_Hebr",
    "Persian":               "pes_Arab",
    "Ukrainian":             "ukr_Cyrl",
    "Czech":                 "ces_Latn",
    "Swedish":               "swe_Latn",
    "Danish":                "dan_Latn",
    "Finnish":               "fin_Latn",
    "Norwegian":             "nob_Latn",
    "Hungarian":             "hun_Latn",
    "Romanian":              "ron_Latn",
    "Bulgarian":             "bul_Cyrl",
    "Croatian":              "hrv_Latn",
    "Serbian":               "srp_Cyrl",
    "Slovak":                "slk_Latn",
    "Lithuanian":            "lit_Latn",
    "Latvian":               "lvs_Latn",
    "Estonian":              "est_Latn",
    "Slovenian":             "slv_Latn",
    "Catalan":               "cat_Latn",
    "Tagalog":               "tgl_Latn",
    "Tamil":                 "tam_Taml",
    "Telugu":                "tel_Telu",
    "Kannada":               "kan_Knda",
    "Malayalam":             "mal_Mlym",
    "Marathi":               "mar_Deva",
    "Gujarati":              "guj_Gujr",
}

# ISO 639-1 codes for Google Speech Recognition (subset of supported languages)
SPEECH_LANG_CODES: dict[str, str] = {
    "English":              "en-US",
    "French":               "fr-FR",
    "Arabic":               "ar-SA",
    "Spanish":              "es-ES",
    "German":               "de-DE",
    "Chinese (Simplified)": "zh-CN",
    "Japanese":             "ja-JP",
    "Korean":               "ko-KR",
    "Russian":              "ru-RU",
    "Portuguese":           "pt-PT",
    "Italian":              "it-IT",
}


SUPPORTED_LANGUAGES: list[str] = list(LANGUAGE_CODES.keys())

def get_flores_code(language: str, fallback: str = "eng_Latn") -> str:
    """Return the floress-200 code for a language name."""
    return LANGUAGE_CODES.get(language, fallback)


def get_speech_code(language: str, fallback: str = "en_US") -> str:
    """Return the GOOGLE SR ISO code for a language name."""
    return SPEECH_LANG_CODES.get(language, fallback)



if __name__ == "__main__":
    print("Supported Languages:")
    for lang in SUPPORTED_LANGUAGES:
        print(f" {lang} and flores code: {get_flores_code(lang)} and speech code: {get_speech_code(lang)}")