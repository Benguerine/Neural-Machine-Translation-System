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
    "Spanish":               "spa_Latn",
    "Portuguese":            "por_Latn",
    "Italian":               "ita_Latn",
    "German":                "deu_Latn",
    "Dutch":                 "nld_Latn",
    "Polish":                "pol_Latn",
    "Romanian":              "ron_Latn",
    "Czech":                 "ces_Latn",
    "Slovak":                "slk_Latn",
    "Hungarian":             "hun_Latn",
    "Swedish":               "swe_Latn",
    "Danish":                "dan_Latn",
    "Finnish":               "fin_Latn",
    "Norwegian":             "nob_Latn",
    "Croatian":              "hrv_Latn",
    "Lithuanian":            "lit_Latn",
    "Latvian":               "lvs_Latn",
    "Estonian":              "est_Latn",
    "Slovenian":             "slv_Latn",
    "Catalan":               "cat_Latn",
    "Galician":              "glg_Latn",
    "Basque":                "eus_Latn",
    "Icelandic":             "isl_Latn",
    "Albanian":              "als_Latn",
    "Bosnian":               "bos_Latn",
    "Welsh":                 "cym_Latn",
    "Irish":                 "gle_Latn",
    "Maltese":               "mlt_Latn",
    "Luxembourgish":         "ltz_Latn",
    "Afrikaans":             "afr_Latn",
    "Russian":               "rus_Cyrl",
    "Ukrainian":             "ukr_Cyrl",
    "Bulgarian":             "bul_Cyrl",
    "Serbian":               "srp_Cyrl",
    "Belarusian":            "bel_Cyrl",
    "Macedonian":            "mkd_Cyrl",
    "Kazakh":                "kaz_Cyrl",
    "Kyrgyz":                "kir_Cyrl",
    "Tajik":                 "tgk_Cyrl",
    "Mongolian":             "khk_Cyrl",
    "Greek":                 "ell_Grek",
    "Georgian":              "kat_Geor",
    "Armenian":              "hye_Armn",
    "Arabic":                "arb_Arab",
    "Persian":               "pes_Arab",
    "Urdu":                  "urd_Arab",
    "Pashto":                "pbt_Arab",
    "Sindhi":                "snd_Arab",
    "Hebrew":                "heb_Hebr",
    "Turkish":               "tur_Latn",
    "Azerbaijani":           "azj_Latn",
    "Uzbek":                 "uzn_Latn",
    "Turkmen":               "tuk_Latn",
    "Kurdish":               "kmr_Latn",
    "Hindi":                 "hin_Deva",
    "Bengali":               "ben_Beng",
    "Marathi":               "mar_Deva",
    "Nepali":                "npi_Deva",
    "Gujarati":              "guj_Gujr",
    "Punjabi":               "pan_Guru",
    "Tamil":                 "tam_Taml",
    "Telugu":                "tel_Telu",
    "Kannada":               "kan_Knda",
    "Malayalam":             "mal_Mlym",
    "Sinhala":               "sin_Sinh",
    "Odia":                  "ory_Orya",
    "Chinese (Simplified)":  "zho_Hans",
    "Chinese (Traditional)": "zho_Hant",
    "Japanese":              "jpn_Jpan",
    "Korean":                "kor_Hang",
    "Vietnamese":            "vie_Latn",
    "Thai":                  "tha_Thai",
    "Indonesian":            "ind_Latn",
    "Malay":                 "zsm_Latn",
    "Tagalog":               "tgl_Latn",
    "Khmer":                 "khm_Khmr",
    "Lao":                   "lao_Laoo",
    "Burmese":               "mya_Mymr",
    "Javanese":              "jav_Latn",
    "Sundanese":             "sun_Latn",
    "Cebuano":               "ceb_Latn",
    "Tibetan":               "bod_Tibt",
    "Swahili":               "swh_Latn",
    "Amharic":               "amh_Ethi",
    "Hausa":                 "hau_Latn",
    "Yoruba":                "yor_Latn",
    "Igbo":                  "ibo_Latn",
    "Zulu":                  "zul_Latn",
    "Xhosa":                 "xho_Latn",
    "Somali":                "som_Latn",
    "Oromo":                 "gaz_Latn",
    "Wolof":                 "wol_Latn",
    "Kinyarwanda":           "kin_Latn",
    "Lingala":               "lin_Latn",
    "Bambara":               "bam_Latn",
    "Quechua":               "quy_Latn",
    "Guarani":               "grn_Latn",
    "Aymara":                "ayr_Latn",
}

assert len(LANGUAGE_CODES) == 100, (
    f"Expected 100 languages, got {len(LANGUAGE_CODES)}"
)

# BCP-47 codes for Google Speech Recognition (subset of supported languages)
SPEECH_LANG_CODES: dict[str, str] = {
    "English":               "en-US",
    "French":                "fr-FR",
    "Spanish":               "es-ES",
    "Portuguese":            "pt-PT",
    "Italian":               "it-IT",
    "German":                "de-DE",
    "Dutch":                 "nl-NL",
    "Polish":                "pl-PL",
    "Romanian":              "ro-RO",
    "Czech":                 "cs-CZ",
    "Hungarian":             "hu-HU",
    "Swedish":               "sv-SE",
    "Finnish":               "fi-FI",
    "Norwegian":             "nb-NO",
    "Croatian":              "hr-HR",
    "Slovak":                "sk-SK",
    "Russian":               "ru-RU",
    "Ukrainian":             "uk-UA",
    "Greek":                 "el-GR",
    "Georgian":              "ka-GE",
    "Armenian":              "hy-AM",
    "Arabic":                "ar-SA",
    "Hebrew":                "he-IL",
    "Persian":               "fa-IR",
    "Turkish":               "tr-TR",
    "Azerbaijani":           "az-AZ",
    "Uzbek":                 "uz-UZ",
    "Hindi":                 "hi-IN",
    "Bengali":               "bn-BD",
    "Gujarati":              "gu-IN",
    "Punjabi":               "pa-IN",
    "Tamil":                 "ta-IN",
    "Telugu":                "te-IN",
    "Kannada":               "kn-IN",
    "Malayalam":             "ml-IN",
    "Marathi":               "mr-IN",
    "Nepali":                "ne-NP",
    "Urdu":                  "ur-PK",
    "Sinhala":               "si-LK",
    "Chinese (Simplified)":  "zh-CN",
    "Chinese (Traditional)": "zh-TW",
    "Japanese":              "ja-JP",
    "Korean":                "ko-KR",
    "Vietnamese":            "vi-VN",
    "Thai":                  "th-TH",
    "Indonesian":            "id-ID",
    "Malay":                 "ms-MY",
    "Tagalog":               "fil-PH",
    "Khmer":                 "km-KH",
    "Burmese":               "my-MM",
    "Mongolian":             "mn-MN",
    # Swahili removed: Google SR does not support sw-KE reliably;
    # get_speech_code("Swahili") will return the fallback (en-US by default).
    "Amharic":               "am-ET",
    "Zulu":                  "zu-ZA",
    "Afrikaans":             "af-ZA",
    "Catalan":               "ca-ES",
    "Welsh":                 "cy-GB",
    "Irish":                 "ga-IE",
    "Maltese":               "mt-MT",
}


SUPPORTED_LANGUAGES: list[str] = list(LANGUAGE_CODES.keys())


def get_flores_code(language: str, fallback: str = "eng_Latn") -> str:
    """Return the FLORES-200 code for a language name."""
    return LANGUAGE_CODES.get(language, fallback)


def get_speech_code(language: str, fallback: str = "en-US") -> str:
    """Return the Google SR BCP-47 code for a language name."""
    return SPEECH_LANG_CODES.get(language, fallback)


if __name__ == "__main__":
    print(f"Total supported languages: {len(SUPPORTED_LANGUAGES)}\n")
    for lang in SUPPORTED_LANGUAGES:
        flores = get_flores_code(lang)
        speech = SPEECH_LANG_CODES.get(lang, "—")
        print(f"  {lang:<25} flores: {flores:<12} speech: {speech}")