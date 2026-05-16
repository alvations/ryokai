"""Languages supported by ryokai.

Coverage is the intersection of the languages handled by the default
backbones:
  * `wietsedv/xlm-roberta-base-ft-udpos28` (the POS-heuristic SRL)
  * `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
    (the default sentence-similarity backbone)

The 13 MEANT 2.0 languages are guaranteed first-class (curated stopwords
in `data/stopwords.yaml`). Other languages work but rely on the embedded
punctuation-only filter for content-word filtering since we ship no
hand-curated stopwords for them.
"""

# Original 13 MEANT 2.0 languages (curated stopwords in data/stopwords.yaml)
MEANT_LANGS: dict[str, str] = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "cs": "Czech",
    "fi": "Finnish",
    "hi": "Hindi",
    "lv": "Latvian",
    "pl": "Polish",
    "ro": "Romanian",
    "ru": "Russian",
    "tr": "Turkish",
    "zh": "Chinese",
}

# Additional UDPOS28-supported languages (no curated stopwords)
_EXTRA_LANGS: dict[str, str] = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "da": "Danish",
    "el": "Greek",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "he": "Hebrew",
    "hr": "Croatian",
    "hu": "Hungarian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "lt": "Lithuanian",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "pt": "Portuguese",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
}

SUPPORTED_LANGS: dict[str, str] = {**MEANT_LANGS, **_EXTRA_LANGS}
