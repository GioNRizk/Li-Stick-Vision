"""Manually exercise the configured TTS backend without starting vision."""

from __future__ import annotations

import config
from decision_engine import GuidanceDecision, PRIORITY
from voice_manager import VoiceManager


TEST_PHRASES = (
    "Li-Stick ready",
    "Stop",
    "Move slightly left",
    "Move slightly right",
    "Head obstacle",
    "Hold for emergency",
    "Emergency cancelled",
    "Sending emergency alert",
    "Emergency alert sent with location",
    "Emergency alert sent. Location unavailable",
    "No connection. Retrying emergency alert",
    "Emergency delivery failed. Retrying",
    "Fall detected",
    "Silent mode on",
    "AI guidance paused",
    "Cane paused",
)


def main() -> int:
    voice = VoiceManager(enabled=config.ENABLE_VOICE)
    try:
        if not voice.enabled:
            return 1

        for index, phrase in enumerate(TEST_PHRASES, start=1):
            decision = GuidanceDecision(
                code=f"TTS_TEST_{index}",
                message=phrase,
                priority=PRIORITY["INFO"],
                risk_level="info",
                source="tts_test",
            )
            if not voice.speak_immediate(decision):
                return 1
    finally:
        voice.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
