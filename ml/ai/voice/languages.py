"""
Languages the voice interface supports.

Charlie is meant to be usable by clinicians whose first language is not
English, so speech recognition, speech synthesis and the assistant's replies
all have to follow the same language — and, critically, agree on which one it
is. Three different components each guessing separately is how you end up with
a Vietnamese question, an English answer, and a Chinese voice reading it.

Each entry carries three codes because the three subsystems want different
formats:

``whisper``  ISO-639-1, what faster-whisper expects (``vi``)
``bcp47``    what the browser's SpeechRecognition and SpeechSynthesis want
             (``vi-VN``) — a bare ``vi`` is rejected by some engines
``label``    the language's own name, because someone looking for Vietnamese
             is looking for "Tiếng Việt", not "Vietnamese"
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    whisper: str
    bcp47: str
    label: str
    english_name: str
    #: Whether the assistant is instructed to reply in this language.
    reply: bool = True


#: Ordered for the picker: auto-detect first, then by expected use.
LANGUAGES: tuple[Language, ...] = (
    Language("", "en-US", "Auto-detect", "Auto-detect", reply=False),
    Language("en", "en-US", "English", "English"),
    Language("vi", "vi-VN", "Tiếng Việt", "Vietnamese"),
    Language("zh", "zh-CN", "中文", "Chinese"),
    Language("ja", "ja-JP", "日本語", "Japanese"),
    Language("ko", "ko-KR", "한국어", "Korean"),
    Language("es", "es-ES", "Español", "Spanish"),
    Language("fr", "fr-FR", "Français", "French"),
    Language("de", "de-DE", "Deutsch", "German"),
    Language("pt", "pt-BR", "Português", "Portuguese"),
    Language("hi", "hi-IN", "हिन्दी", "Hindi"),
    Language("ar", "ar-SA", "العربية", "Arabic"),
)

BY_CODE: dict[str, Language] = {lang.whisper: lang for lang in LANGUAGES if lang.whisper}


def resolve(code: str) -> Language:
    """
    Look up a language by any reasonable spelling of its code.

    Accepts ``vi``, ``VI``, ``vi-VN`` and ``vi_VN`` — clients disagree about
    the format and rejecting a valid language over a hyphen would be a
    pointless failure.
    """
    normalised = (code or "").strip().lower().replace("_", "-")
    if not normalised:
        return LANGUAGES[0]
    if normalised in BY_CODE:
        return BY_CODE[normalised]
    base = normalised.split("-")[0]
    return BY_CODE.get(base, LANGUAGES[0])


def reply_instruction(code: str) -> str:
    """
    The line added to the prompt so the answer comes back in the same language
    the question was asked in.

    Empty for auto-detect: with no reliable signal, instructing a language
    would be guessing, and an answer confidently delivered in the wrong
    language is worse than one that simply mirrors the question.
    """
    language = resolve(code)
    if not language.reply or not language.whisper:
        return ""
    return (
        f"LANGUAGE: reply entirely in {language.english_name} "
        f"({language.label}). Keep clinical and anatomical terms in their "
        f"standard form — a surgeon searching for a structure needs the term "
        f"they would find in the literature, so give the accepted "
        f"{language.english_name} term and put the English in parentheses on "
        f"first use where the two differ."
    )


def to_dict(language: Language) -> dict:
    return {
        "code": language.whisper,
        "bcp47": language.bcp47,
        "label": language.label,
        "english_name": language.english_name,
    }


def catalogue() -> list[dict]:
    return [to_dict(language) for language in LANGUAGES]
