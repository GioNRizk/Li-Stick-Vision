from __future__ import annotations

from decision_engine import GuidanceDecision, PRIORITY
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


def main():
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

    print("Emergency pass-through while all pause modes are active")
    print("-------------------------------------------------------")
    state = RuntimeState(
        ai_guidance_enabled=False,
        silent_mode_enabled=True,
        full_pause_enabled=True,
    )
    ai_decision = _sample_ai_decision()
    print(f"state       : {state.mode_summary()}")
    print(
        "ai guidance : "
        f"{'allowed' if state.allows_ai_decision(ai_decision) else 'suppressed'} "
        f"-> {ai_decision.message}"
    )

    for event in ["SOS_SENT", "FALL_DETECTED", "HEAD_SENSOR_ALERT", "BATTERY_LOW"]:
        decision = parser._parse_line(f"{event}\n".encode("utf-8"))
        if decision is None:
            continue
        print(
            f"{event:<17}: "
            f"{'allowed' if state.allows_esp32_decision(decision) else 'suppressed'} "
            f"-> {decision.message}"
        )


if __name__ == "__main__":
    main()
