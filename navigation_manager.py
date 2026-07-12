from __future__ import annotations

import time
from collections.abc import Callable

import config
from decision_engine import GuidanceDecision


BYPASS_STABILIZATION_CODES = {
    "STOP",
    "CENTER_DANGER",
    "HEAD_OBSTACLE",
    "HEAD_SENSOR_ALERT",
    "SOS",
    "SOS_SENT",
    "SOS_HOLD_STARTED",
    "FALL",
    "FALL_DETECTED",
}

LOCKING_CODES = {
    "STOP",
    "CENTER_DANGER",
    "HEAD_OBSTACLE",
    "HEAD_SENSOR_ALERT",
    "SOS",
    "SOS_SENT",
    "FALL",
    "FALL_DETECTED",
}


class NavigationManager:
    """
    Smooths frame-by-frame guidance before anything reaches text-to-speech.

    DecisionEngine still owns the per-frame decision. NavigationManager owns
    temporal behavior: stable-frame confirmation, short command locks, repeated
    command suppression, and priority-aware interruption.
    """

    def __init__(
        self,
        stable_frames: int = config.NAV_STABLE_FRAMES,
        command_lock_seconds: float = config.NAV_COMMAND_LOCK_SECONDS,
        repeat_seconds: float = config.NAV_REPEAT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.stable_frames = max(1, int(stable_frames))
        self.command_lock_seconds = max(0.0, float(command_lock_seconds))
        self.repeat_seconds = max(0.0, float(repeat_seconds))
        self._clock = clock

        self._candidate_code: str | None = None
        self._candidate_frames = 0
        self._last_command: GuidanceDecision | None = None
        self._last_spoken_at: dict[str, float] = {}
        self._locked_decision: GuidanceDecision | None = None
        self._lock_until = 0.0

    @property
    def last_command_code(self) -> str | None:
        """Return the last command released for speech, if any."""
        return self._last_command.code if self._last_command is not None else None

    def update(self, decision: GuidanceDecision) -> GuidanceDecision | None:
        """
        Return a decision only when VoiceManager should speak it.

        Silent decisions reset stabilization state. AI navigation commands must
        remain stable for ``stable_frames`` unless they are safety-critical.
        Runtime and ESP32 events are not frame streams, so they bypass
        stabilization while still using repeat and priority protection.
        """
        now = self._clock()
        self._clear_expired_lock(now)

        if not decision.should_speak:
            self._reset_candidate()
            return None

        if self._is_blocked_by_lock(decision):
            return None

        if self._requires_stabilization(decision):
            stable_count = self._record_candidate(decision)
            if stable_count < self.stable_frames:
                return None
        else:
            self._record_candidate(decision)

        if self._is_repeat_suppressed(decision, now):
            return None

        return self._release(decision, now)

    def _requires_stabilization(self, decision: GuidanceDecision) -> bool:
        if self.stable_frames <= 1:
            return False
        if decision.source != "ai":
            return False
        if decision.code in BYPASS_STABILIZATION_CODES or decision.is_emergency:
            return False
        return True

    def _record_candidate(self, decision: GuidanceDecision) -> int:
        if decision.code == self._candidate_code:
            self._candidate_frames += 1
        else:
            self._candidate_code = decision.code
            self._candidate_frames = 1
        return self._candidate_frames

    def _reset_candidate(self):
        self._candidate_code = None
        self._candidate_frames = 0

    def _is_repeat_suppressed(self, decision: GuidanceDecision, now: float) -> bool:
        if self.repeat_seconds <= 0.0:
            return False

        last_spoken = self._last_spoken_at.get(decision.code)
        if last_spoken is None:
            return False

        return now - last_spoken < self.repeat_seconds

    def _is_blocked_by_lock(self, decision: GuidanceDecision) -> bool:
        locked = self._locked_decision
        if locked is None:
            return False
        if not self._is_lock_sensitive(decision):
            return False
        return decision.priority <= locked.priority

    def _is_lock_sensitive(self, decision: GuidanceDecision) -> bool:
        return decision.source == "ai" or decision.code in LOCKING_CODES

    def _release(
        self,
        decision: GuidanceDecision,
        now: float,
    ) -> GuidanceDecision:
        self._last_command = decision
        self._last_spoken_at[decision.code] = now

        if self._starts_lock(decision):
            self._locked_decision = decision
            self._lock_until = now + self.command_lock_seconds

        return decision

    def _starts_lock(self, decision: GuidanceDecision) -> bool:
        if self.command_lock_seconds <= 0.0:
            return False
        return decision.source == "ai" or decision.code in LOCKING_CODES

    def _clear_expired_lock(self, now: float):
        if self._locked_decision is not None and now >= self._lock_until:
            self._locked_decision = None
            self._lock_until = 0.0
