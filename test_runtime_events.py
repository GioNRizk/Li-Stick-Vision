from __future__ import annotations

from decision_engine import DecisionEngine, GuidanceDecision, PRIORITY
from runtime_state import RuntimeState
from uart_bridge import UartBridge


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


def main():
    simulate_ai_decisions()
    simulate_runtime_events()


if __name__ == "__main__":
    main()
