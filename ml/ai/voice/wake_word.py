"""
Wake-word gating and ambient-speech rejection.

An operating room is a continuously noisy speech environment: circulating
staff, anaesthesia, phone calls, music. An always-listening assistant that acts
on everything it hears is worse than useless — it interrupts, and it acts on
instructions that were never addressed to it.

Two gates, in order:

1. **Wake word.** The utterance must be addressed to the assistant, by name,
   near the start. Everything else is discarded without being routed anywhere.
2. **Ambient rejection.** Even after the wake word, fragments that carry no
   command ("charlie... uh"), filler-only speech, and known OR chatter patterns
   are dropped rather than sent to an agent that will guess at them.

Matching is fuzzy on purpose. Speech recognisers routinely render "Charlie" as
"O'Ryan", "or ryan", "aryan", or "iron" depending on accent and microphone, and
a surgeon should not have to enunciate for the assistant's benefit.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("charlie.wake_word")

#: Recognised renderings of the wake word. Ordered longest-first so
#: "hey charlie" is stripped whole rather than leaving "hey".
WAKE_VARIANTS = [
    "hey charlie", "ok charlie", "okay charlie", "hi charlie",
    "charlie", "o'ryan", "o ryan", "oryan", "aryan", "ariane",
    "oh ryan", "iron", "o'brien",
]

_WAKE_RE = re.compile(
    r"^\W*(?:" + "|".join(re.escape(v) for v in WAKE_VARIANTS) + r")\b[\s,.:;!?-]*",
    re.IGNORECASE,
)

#: Filler-only utterances that survive the wake word but say nothing.
_FILLER_RE = re.compile(
    r"^(?:uh|um|er|ah|hmm+|mm+|yeah|yep|ok|okay|right|so|well|and|the|a)?[\s.,]*$",
    re.IGNORECASE,
)

#: Ambient OR speech that is clearly not addressed to an assistant. These are
#: rejected even if a wake-word variant was spuriously matched.
_AMBIENT_RE = re.compile(
    r"\b(suction|sponge count|count is correct|can i have|pass me|"
    r"blood pressure is|sats are|heart rate|how much longer|"
    r"music|turn (it|the music)|phone|page \w+|"
    r"scrub (nurse|tech)|anaesthe|anesthe)\b",
    re.IGNORECASE,
)

#: A command needs at least this many characters after the wake word.
MIN_COMMAND_CHARS = 6


@dataclass
class WakeResult:
    """Outcome of gating one utterance."""

    triggered: bool
    command: str = ""
    reason: str = ""
    matched_variant: str = ""

    def to_dict(self) -> dict:
        return {
            "triggered": self.triggered,
            "command": self.command,
            "reason": self.reason,
            "matched_variant": self.matched_variant,
        }


def detect(transcript: str, require_wake_word: bool = True) -> WakeResult:
    """
    Gate one transcript.

    >>> detect("Charlie, what's at risk here?").command
    "what's at risk here?"
    >>> detect("can I have the suction please").triggered
    False
    >>> detect("what phase is this", require_wake_word=False).triggered
    True
    """
    text = (transcript or "").strip()
    if not text:
        return WakeResult(False, reason="empty")

    match = _WAKE_RE.match(text)

    if require_wake_word:
        if not match:
            return WakeResult(False, reason="no wake word")
        command = text[match.end() :].strip()
        variant = match.group(0).strip(" ,.:;!?-").lower()
    else:
        # Push-to-talk mode: the button press *is* the addressing signal, so
        # a wake word is optional but still stripped if present.
        command = text[match.end() :].strip() if match else text
        variant = match.group(0).strip(" ,.:;!?-").lower() if match else ""

    if _AMBIENT_RE.search(command):
        return WakeResult(False, reason="ambient OR speech", matched_variant=variant)

    if _FILLER_RE.match(command) or len(command) < MIN_COMMAND_CHARS:
        return WakeResult(False, reason="no command after wake word", matched_variant=variant)

    return WakeResult(True, command=command, matched_variant=variant)


def config() -> dict:
    """Wake-word configuration, surfaced in the UI and console."""
    return {
        "wake_variants": WAKE_VARIANTS,
        "primary": "Charlie",
        "min_command_chars": MIN_COMMAND_CHARS,
        "rejects": [
            "utterances without the wake word",
            "filler-only speech after the wake word",
            "recognised ambient OR chatter",
        ],
        "note": (
            "Fuzzy matching covers common mis-recognitions of 'Charlie'. "
            "Push-to-talk bypasses the wake word — the button press is the "
            "addressing signal."
        ),
    }
