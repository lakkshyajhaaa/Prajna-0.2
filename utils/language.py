"""
utils/language.py — Prajñā 0.2
Language utilities extracted from llm_utils.py for modularity.
llm_utils.py is UNCHANGED — this module provides the same dict for new code.
"""

# 22 Scheduled Languages of India + English
LANGUAGES = {
    "English":   "English",
    "Assamese":  "অসমীয়া",
    "Bengali":   "বাংলা",
    "Bodo":      "बर'",
    "Dogri":     "डोगरी",
    "Gujarati":  "ગુજારાતી",
    "Hindi":     "हिन्दी",
    "Kannada":   "ಕನ್ನಡ",
    "Kashmiri":  "کٲشُر",
    "Konkani":   "कोंकणी",
    "Maithili":  "मैथिली",
    "Malayalam": "മലയാളം",
    "Manipuri":  "মৈতেইললোন",
    "Marathi":   "मराठी",
    "Nepali":    "नेपाली",
    "Odia":      "ଓଡ଼ିଆ",
    "Punjabi":   "ਪੰਜਾਬੀ",
    "Sanskrit":  "संस्कृतम्",
    "Santali":   "संताली",
    "Sindhi":    "سنڌي",
    "Tamil":     "தமிழ்",
    "Telugu":    "తెలుగు",
    "Urdu":      "اردو",
}


def get_native_languages() -> dict:
    return LANGUAGES
