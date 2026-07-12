from __future__ import annotations

import importlib
import sys
import time
import types

from decision_engine import DecisionEngine, GuidanceDecision, PRIORITY
from navigation_manager import NavigationManager
from runtime_state import RuntimeState
from uart_bridge import UartBridge
from voice_manager import VoiceManager


EVENTS = [
    "AI_PAUSE_ON",
    "AI_PAUSE_OFF",
    "SILENT_MODE_ON",
    "SILENT_MODE_OFF",
    "FULL_PAUSE_ON",
    "FULL_PAUSE_OFF",
    "SOS_SENT",
    "FALL_DETECTED",
    "HEAD_SENSOR_ALERT",
]


def _sample_ai_decision() -> GuidanceDecision:
    return GuidanceDecision(
        code="PERSON_AHEAD",
        message="Person ahead",
        priority=PRIORITY["PERSON_AHEAD"],
        risk_level="warning",
        source="ai",
    )


def _status_decision(code: str, message: str) -> GuidanceDecision:
    return GuidanceDecision(
        code=code,
        message=message,
        priority=PRIORITY[code],
        risk_level="info",
        source="runtime",
    )


def _obstacle(
    label: str,
    zone: str = "center",
    proximity: float = 0.4,
    bbox: list[int] | None = None,
) -> dict:
    return {
        "label": label,
        "confidence": 0.9,
        "proximity": proximity,
        "zone": zone,
        "bbox": bbox or [260, 260, 380, 460],
    }


def _detector_result(*obstacles: dict, risk_level: str = "warning") -> dict:
    return {
        "risk_level": risk_level,
        "frame_width": 640,
        "frame_height": 480,
        "smooth_proximity": max(
            (obstacle.get("proximity", 0.0) for obstacle in obstacles),
            default=0.0,
        ),
        "obstacles": list(obstacles),
    }


def _expect(label: str, actual, expected):
    result = "PASS" if actual == expected else "FAIL"
    print(f"{result:<4} {label:<42} expected={expected!r} actual={actual!r}")
    assert actual == expected


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _load_ai_release_guard():
    """Import main's release guard without requiring camera/AI packages."""
    if "main" not in sys.modules:
        cv2_stub = types.ModuleType("cv2")
        camera_stub = types.ModuleType("camera_source")
        detector_stub = types.ModuleType("detector")

        class CameraOpenError(Exception):
            pass

        camera_stub.CameraOpenError = CameraOpenError
        camera_stub.open_camera = lambda: None
        detector_stub.ObstacleDetector = object

        sys.modules.setdefault("cv2", cv2_stub)
        sys.modules.setdefault("camera_source", camera_stub)
        sys.modules.setdefault("detector", detector_stub)

    return importlib.import_module("main")._release_ai_decision


class _RecordingVoice:
    def __init__(self):
        self.spoken: list[GuidanceDecision] = []
        self.cancel_count = 0

    def speak_decision(self, decision: GuidanceDecision) -> bool:
        self.spoken.append(decision)
        return True

    def cancel_current_and_pending(self):
        self.cancel_count += 1


class _RecordingNavigation(NavigationManager):
    def __init__(self, stable_frames: int = 2):
        super().__init__(
            stable_frames=stable_frames,
            command_lock_seconds=0.0,
            repeat_seconds=0.0,
        )
        self.received: list[GuidanceDecision] = []
        self.reset_count = 0

    def update(self, decision: GuidanceDecision) -> GuidanceDecision | None:
        self.received.append(decision)
        return super().update(decision)

    def reset(self):
        self.reset_count += 1
        super().reset()


class _SubprocessVoiceManager(VoiceManager):
    def __init__(self):
        self.started_messages: list[str] = []
        super().__init__(
            enabled=True,
            cooldown_seconds=0.0,
            emergency_cooldown_seconds=0.0,
            backend="test_subprocess",
            timeout_seconds=10.0,
        )

    def _speak_backend(self, message: str) -> bool:
        self.started_messages.append(message)
        return self._run_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
        )


def simulate_ai_decisions():
    engine = DecisionEngine()

    print("AI decision simulation")
    print("----------------------")

    ai_ready = _status_decision("AI_READY", "AI guidance ready")
    _expect("startup confirmation code", ai_ready.code, "AI_READY")
    _expect("startup confirmation message", ai_ready.message, "AI guidance ready")

    vehicle = engine.decide(
        _detector_result(_obstacle("car", zone="center", proximity=0.4))
    )
    _expect("vehicle centered nearby", vehicle.code, "VEHICLE_NEARBY")
    _expect("vehicle nearby message", vehicle.message, "Vehicle nearby")

    vehicle_stop = engine.decide(
        _detector_result(_obstacle("truck", zone="center", proximity=0.86))
    )
    _expect("vehicle centered very close", vehicle_stop.code, "STOP")

    stairs = engine.decide(
        _detector_result(_obstacle("stairs", zone="center", proximity=0.22))
    )
    _expect("stairs label placeholder", stairs.code, "STAIRS_AHEAD")
    _expect("stairs message", stairs.message, "Stairs ahead")

    pole = engine.decide(
        _detector_result(_obstacle("post", zone="center", proximity=0.22))
    )
    _expect("pole/post label placeholder", pole.code, "POLE_AHEAD")
    _expect("pole message", pole.message, "Pole ahead")

    upper_pole = engine.decide(
        _detector_result(
            _obstacle("pole", zone="center", proximity=0.3, bbox=[260, 20, 380, 170])
        )
    )
    _expect("upper pole keeps pole context", upper_pole.code, "POLE_AHEAD")

    pet = engine.decide(
        _detector_result(_obstacle("dog", zone="center", proximity=0.6))
    )
    _expect("pet below danger avoids stop", pet.code, "OBJECT_AHEAD")
    _expect("pet uses generic obstacle wording", pet.message, "Obstacle ahead")

    side_pet = engine.decide(
        _detector_result(_obstacle("cat", zone="left", proximity=0.5))
    )
    _expect("side pet avoids directional noise", side_pet.code, "SAFE")
    print()


def simulate_runtime_events():
    state = RuntimeState()
    parser = UartBridge(enabled=False)

    print("Runtime event simulation")
    print("------------------------")
    print(f"initial: {state.mode_summary()}")
    print()

    for event in EVENTS:
        decision = parser._parse_line(f"{event}\n".encode("utf-8"))
        if decision is None:
            print(f"{event}: parser returned None")
            continue

        state.apply_esp32_decision(decision)

        ai_decision = _sample_ai_decision()
        esp32_allowed = state.allows_esp32_decision(decision)
        ai_allowed = state.allows_ai_decision(ai_decision)

        esp32_result = "allowed" if esp32_allowed else "suppressed"
        ai_result = "allowed" if ai_allowed else "suppressed"

        print(f"event: {event}")
        print(f"  esp32 voice : {esp32_result} -> {decision.message}")
        print(f"  ai guidance : {ai_result} -> {ai_decision.message}")
        print(f"  state       : {state.mode_summary()}")
        print()

    head_alert = parser._parse_line(b"HEAD_SENSOR_ALERT\n")
    assert head_alert is not None
    state.apply_esp32_decision(head_alert)
    _expect("head sensor alert speaks", head_alert.message, "Head obstacle")
    _expect(
        "head sensor timestamp recorded",
        state.last_head_sensor_alert_time is not None,
        True,
    )

    print()
    print("AI pause and emergency pass-through")
    print("-----------------------------------")
    state = RuntimeState()
    pause = parser._parse_line(b"AI_PAUSE_ON\n")
    assert pause is not None
    state.apply_esp32_decision(pause)

    ai_decision = _sample_ai_decision()
    _expect(
        "AI_PAUSE_ON suppresses camera guidance",
        state.allows_ai_decision(ai_decision),
        False,
    )

    sos = parser._parse_line(b"SOS_SENT\n")
    assert sos is not None
    state.apply_esp32_decision(sos)
    _expect("SOS speaks while AI paused", state.allows_esp32_decision(sos), True)
    _expect("SOS message", sos.message, "Emergency alert sent")


def test_ai_pause_release_guards():
    release_ai = _load_ai_release_guard()
    parser = UartBridge(enabled=False)
    state = RuntimeState()
    navigation = _RecordingNavigation(stable_frames=2)
    voice = _RecordingVoice()

    print()
    print("AI pause release guard tests")
    print("----------------------------")

    old_ai = _sample_ai_decision()
    _expect("first AI frame is stabilizing", release_ai(
        state, navigation, voice, old_ai
    ), None)
    _expect("second AI frame is released", release_ai(
        state, navigation, voice, old_ai
    ).code, "PERSON_AHEAD")

    pause = parser._parse_line(b"AI_PAUSE_ON\n")
    assert pause is not None
    was_enabled = state.ai_guidance_enabled
    state.apply_esp32_decision(pause)
    if was_enabled and not state.ai_guidance_enabled:
        voice.cancel_current_and_pending()
        navigation.reset()

    pause_release = navigation.update(pause)
    assert pause_release is not None
    voice.speak_decision(pause_release)

    ai_updates_before_paused_frames = sum(
        decision.source == "ai" for decision in navigation.received
    )
    ai_speech_before_paused_frames = sum(
        decision.source == "ai" for decision in voice.spoken
    )
    for _ in range(5):
        _expect("paused frame remains silent", release_ai(
            state, navigation, voice, _sample_ai_decision()
        ), None)

    _expect(
        "paused AI never reaches navigation",
        sum(decision.source == "ai" for decision in navigation.received),
        ai_updates_before_paused_frames,
    )
    _expect(
        "paused AI never reaches voice",
        sum(decision.source == "ai" for decision in voice.spoken),
        ai_speech_before_paused_frames,
    )
    _expect("pause transition cancels once", voice.cancel_count, 1)
    _expect("pause transition resets once", navigation.reset_count, 1)
    _expect("pause runtime message speaks", voice.spoken[-1].code, "AI_PAUSE_ON")

    for code in (
        "SOS_SENT",
        "FALL_DETECTED",
        "HEAD_SENSOR_ALERT",
        "BATTERY_LOW",
        "CANE_ON",
    ):
        event = parser._parse_line(f"{code}\n".encode("utf-8"))
        assert event is not None
        state.apply_esp32_decision(event)
        _expect(f"{code} allowed while paused", state.allows_esp32_decision(event), True)
        voice.speak_decision(event)
        _expect(f"{code} reaches voice", voice.spoken[-1].code, code)

    resume = parser._parse_line(b"AI_PAUSE_OFF\n")
    assert resume is not None
    was_enabled = state.ai_guidance_enabled
    state.apply_esp32_decision(resume)
    if not was_enabled and state.ai_guidance_enabled:
        navigation.reset()

    resume_release = navigation.update(resume)
    assert resume_release is not None
    voice.speak_decision(resume_release)
    _expect("resume transition resets navigation", navigation.reset_count, 2)

    fresh_ai = GuidanceDecision(
        code="MOVE_LEFT",
        message="Move left",
        priority=PRIORITY["MOVE_LEFT"],
        risk_level="warning",
        source="ai",
    )
    _expect("resume waits for a fresh stable frame", release_ai(
        state, navigation, voice, fresh_ai
    ), None)
    _expect("old AI command is not replayed", voice.spoken[-1].code, "AI_PAUSE_OFF")
    fresh_release = release_ai(state, navigation, voice, fresh_ai)
    assert fresh_release is not None
    _expect("fresh AI resumes after stabilization", fresh_release.code, "MOVE_LEFT")


def test_voice_cancel_current_and_pending():
    print()
    print("Voice cancellation tests")
    print("------------------------")

    voice = _SubprocessVoiceManager()
    active = _sample_ai_decision()
    queued = GuidanceDecision(
        code="OBJECT_AHEAD",
        message="Obstacle ahead",
        priority=PRIORITY["OBJECT_AHEAD"],
        risk_level="warning",
        source="ai",
    )

    try:
        _expect("active AI speech accepted", voice.speak_decision(active), True)
        _expect(
            "active TTS subprocess started",
            _wait_until(lambda: voice._active_process is not None),
            True,
        )
        active_process = voice._active_process
        assert active_process is not None

        _expect("queued AI speech accepted", voice.speak_decision(queued), True)
        voice.cancel_current_and_pending()

        with voice._lock:
            pending_count = len(voice._pending)
            last_active_code = voice._last_active_code
        _expect("queued speech is discarded", pending_count, 0)
        _expect("last active code is reset", last_active_code, None)
        _expect(
            "active TTS subprocess is terminated",
            _wait_until(lambda: active_process.poll() is not None),
            True,
        )
        _expect(
            "voice worker remains alive",
            voice._thread is not None and voice._thread.is_alive(),
            True,
        )
        time.sleep(0.1)
        _expect(
            "cancelled queued sentence never starts",
            "Obstacle ahead" in voice.started_messages,
            False,
        )

        pause_message = GuidanceDecision(
            code="AI_PAUSE_ON",
            message="AI guidance paused",
            priority=PRIORITY["MODE_CHANGE"],
            risk_level="external",
            source="esp32",
        )
        _expect(
            "runtime speech is accepted after cancellation",
            voice.speak_decision(pause_message),
            True,
        )
        _expect(
            "worker speaks after cancellation",
            _wait_until(
                lambda: "AI guidance paused" in voice.started_messages
            ),
            True,
        )
    finally:
        voice.stop()


def main():
    simulate_ai_decisions()
    simulate_runtime_events()
    test_ai_pause_release_guards()
    test_voice_cancel_current_and_pending()


if __name__ == "__main__":
    main()
