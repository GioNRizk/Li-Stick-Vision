from __future__ import annotations

from dataclasses import dataclass

from decision_engine import GuidanceDecision


MODE_EVENT_CODES = {
    "AI_PAUSE_ON",
    "AI_PAUSE_OFF",
    "SILENT_MODE_ON",
    "SILENT_MODE_OFF",
    "FULL_PAUSE_ON",
    "FULL_PAUSE_OFF",
}

AI_NAVIGATION_CODES = {
    "PERSON_AHEAD",
    "OBJECT_AHEAD",
    "WALL_AHEAD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "STOP",
    "CENTER_DANGER",
    "CROWDED_AREA",
    "HEAD_OBSTACLE",
}

ALWAYS_SPEAK_ESP32_CODES = {
    "FALL_DETECTED",
    "SOS_SENT",
    "HEAD_SENSOR_ALERT",
    "BATTERY_LOW",
}


@dataclass
class RuntimeState:
    ai_guidance_enabled: bool = True
    silent_mode_enabled: bool = False
    full_pause_enabled: bool = False

    def apply_esp32_decision(self, decision: GuidanceDecision) -> bool:
        """
        Apply ESP32 mode events to runtime flags.

        Returns True when the decision changed or confirmed a runtime mode. The
        caller should still pass the event to VoiceManager once so the user hears
        the mode transition.
        """
        code = decision.code
        if code == "AI_PAUSE_ON":
            self.ai_guidance_enabled = False
            return True
        if code == "AI_PAUSE_OFF":
            self.ai_guidance_enabled = True
            return True
        if code == "SILENT_MODE_ON":
            self.silent_mode_enabled = True
            return True
        if code == "SILENT_MODE_OFF":
            self.silent_mode_enabled = False
            return True
        if code == "FULL_PAUSE_ON":
            self.full_pause_enabled = True
            return True
        if code == "FULL_PAUSE_OFF":
            self.full_pause_enabled = False
            return True
        return False

    def allows_ai_guidance(self) -> bool:
        return (
            self.ai_guidance_enabled
            and not self.silent_mode_enabled
            and not self.full_pause_enabled
        )

    def allows_ai_decision(self, decision: GuidanceDecision) -> bool:
        if not decision.should_speak:
            return False
        return self.allows_ai_guidance()

    def allows_esp32_decision(self, decision: GuidanceDecision) -> bool:
        if not decision.should_speak:
            return False
        if decision.code in ALWAYS_SPEAK_ESP32_CODES:
            return True
        return True

    def mode_summary(self) -> dict[str, bool]:
        return {
            "ai_guidance_enabled": self.ai_guidance_enabled,
            "silent_mode_enabled": self.silent_mode_enabled,
            "full_pause_enabled": self.full_pause_enabled,
        }
