import json
import time
from pathlib import Path

import cv2

import config
from camera_source import CameraOpenError, open_camera
from decision_engine import DecisionEngine, GuidanceDecision, PRIORITY
from detector import ObstacleDetector
from runtime_state import RuntimeState
from uart_bridge import UartBridge
from voice_manager import VoiceManager


_RISK_META = {
    #           label                                  cv2 BGR color
    "danger": ("DANGER   Stop now",                   (0, 0, 230)),
    "alert": ("ALERT    Object close",                (0, 100, 255)),
    "warning": ("WARNING  Object approaching",        (0, 200, 255)),
    "approaching": ("DETECTED Monitoring path",       (0, 220, 140)),
    "safe": ("CLEAR    Path safe",                    (0, 200, 60)),
}

_TREND_ARROW = {
    "approaching": ">> getting closer",
    "receding": "<< moving away",
    "stable": "-- stable",
}

BAR_WIDTH = 24


def _prox_bar(score: float) -> str:
    filled = round(score * BAR_WIDTH)
    return "#" * filled + "-" * (BAR_WIDTH - filled)


def _distance_label(score: float) -> str:
    if score >= config.THRESHOLD_DANGER:
        return "~<1 m"
    if score >= config.THRESHOLD_ALERT:
        return "~1-1.5 m"
    if score >= config.THRESHOLD_WARNING:
        return "~2-3 m"
    if score >= config.THRESHOLD_APPROACHING:
        return "~3-5 m"
    return ">5 m"


def _print_status_change(result: dict, decision: GuidanceDecision):
    risk = result["risk_level"]
    label, _ = _RISK_META[risk]
    prox = result["smooth_proximity"]
    trend = result.get("trend", "stable")
    closest = result["obstacles"][0] if result["obstacles"] else None
    obj_info = f"{closest['label'].upper()} in {closest['zone'].upper()}" if closest else "NO OBJECT"
    message = decision.message or "Silent"

    border = "-" * 52
    print(f"\n{border}")
    print(f"  {label}")
    print(f"  Guidance: {message} [{decision.code}]")
    print(f"  {obj_info}   {_distance_label(prox)}")
    print(f"  [{_prox_bar(prox)}] {prox:.2f}  {_TREND_ARROW.get(trend, '')}")
    print(border)


def _print_running_line(result: dict, decision: GuidanceDecision, fps: float):
    risk = result["risk_level"]
    prox = result["smooth_proximity"]
    trend = result.get("trend", "stable")
    arrow = {"approaching": ">>", "receding": "<<", "stable": "--"}.get(trend, "--")
    label = risk.upper().ljust(11)
    closest = result["obstacles"][0] if result["obstacles"] else None
    obj = f"{closest['label'][:8]:<8}.{closest['zone'][:1].upper()}" if closest else "none     "
    command = decision.code[:14].ljust(14)

    line = (
        f"\r  [{label}] {command} {obj}  [{_prox_bar(prox)}] "
        f"{prox:.2f} {arrow}  {_distance_label(prox):<9} {fps:4.1f} fps  "
    )
    print(line, end="", flush=True)


def _draw_overlay(frame, result: dict, decision: GuidanceDecision, fps: float):
    risk = result["risk_level"]
    _, color = _RISK_META[risk]
    prox = result["smooth_proximity"]
    trend = result.get("trend", "stable")
    arrow = {"approaching": ">>", "receding": "<<", "stable": "--"}.get(trend, "--")
    _, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 92), (0, 0, 0), -1)
    cv2.putText(
        frame,
        risk.upper(),
        (8, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
    )
    cv2.putText(
        frame,
        decision.message or "Silent",
        (8, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        1,
    )
    info = f"{_distance_label(prox)}  {arrow}  {fps:.0f}fps"
    cv2.putText(frame, info, (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1)

    bar_x2 = int(8 + prox * 300)
    cv2.rectangle(frame, (8, 76), (308, 84), (60, 60, 60), -1)
    cv2.rectangle(frame, (8, 76), (bar_x2, 84), color, -1)


def _choose_display_decision(
    ai_decision: GuidanceDecision,
    esp32_decisions: list[GuidanceDecision],
    ai_guidance_allowed: bool,
) -> GuidanceDecision:
    active = ai_decision if ai_guidance_allowed else _muted_decision(ai_decision)
    for decision in esp32_decisions:
        if decision.priority > active.priority:
            active = decision
    return active


def _muted_decision(ai_decision: GuidanceDecision) -> GuidanceDecision:
    return GuidanceDecision(
        code="AI_MUTED",
        message=None,
        priority=0,
        risk_level=ai_decision.risk_level,
        source="runtime",
        details={
            "suppressed_ai_code": ai_decision.code,
            "suppressed_ai_message": ai_decision.message,
        },
    )


def _runtime_status_decision(
    code: str,
    message: str,
    **details: str,
) -> GuidanceDecision:
    return GuidanceDecision(
        code=code,
        message=message,
        priority=PRIORITY.get(code, PRIORITY["INFO"]),
        risk_level="info",
        source="runtime",
        details=details,
    )


def _write_latest_json(path: Path, result: dict):
    if config.ENABLE_DETECTION_OUTPUT_JSON:
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def _maybe_write_debug_log(
    path: Path,
    result: dict,
    state: tuple[str, str],
    previous_state: tuple[str, str] | None,
    last_log_time: float,
) -> tuple[tuple[str, str] | None, float]:
    if not config.ENABLE_DEBUG_LOGGING:
        return previous_state, last_log_time

    now = time.monotonic()
    state_changed = state != previous_state
    interval_due = (
        not config.LOG_ON_RISK_CHANGE_ONLY
        and now - last_log_time >= config.DEBUG_LOG_INTERVAL_SECONDS
    )

    if state_changed or interval_due:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result) + "\n")
        return state, now

    return previous_state, last_log_time


def main():
    print("\n" + "=" * 52)
    print("  Li-Stick Vision - Raspberry Pi MVP Guidance")
    print("=" * 52)
    print("  Pipeline: camera -> detector -> decision -> voice -> ESP32 UART")
    print(f"  Voice: {'on' if config.ENABLE_VOICE else 'off'}")
    print(f"  UART : {'on' if config.ENABLE_UART else 'off'}")
    print("  Press 'q' in the camera window to quit.\n")

    voice = VoiceManager(enabled=config.ENABLE_VOICE)

    try:
        cap = open_camera()
    except CameraOpenError as exc:
        print(f"[ERROR] {exc}")
        voice.speak_immediate(
            _runtime_status_decision(
                "AI_UNAVAILABLE",
                "AI camera unavailable",
                component="camera",
                error=str(exc),
            )
        )
        voice.stop()
        return
    except Exception as exc:
        print(f"[ERROR] Camera unavailable: {exc}")
        voice.speak_immediate(
            _runtime_status_decision(
                "AI_UNAVAILABLE",
                "AI camera unavailable",
                component="camera",
                error=str(exc),
            )
        )
        voice.stop()
        return

    try:
        detector = ObstacleDetector()
    except Exception as exc:
        print(f"[ERROR] AI detector unavailable: {exc}")
        voice.speak_immediate(
            _runtime_status_decision(
                "AI_UNAVAILABLE",
                "AI guidance unavailable",
                component="detector",
                error=str(exc),
            )
        )
        cap.release()
        voice.stop()
        return

    decision_engine = DecisionEngine()
    uart = UartBridge(enabled=config.ENABLE_UART)
    runtime_state = RuntimeState()

    voice.speak_immediate(
        _runtime_status_decision("AI_READY", "AI guidance ready")
    )
    if uart.is_connected:
        voice.speak_immediate(
            _runtime_status_decision("CANE_ON", "Cane connected")
        )

    out_json = Path(config.OUTPUT_JSON)
    log_path = Path(config.OUTPUT_LOG)

    previous_display_state: tuple[str, str] | None = None
    previous_log_state: tuple[str, str] | None = None
    last_log_time = 0.0
    frame_times: list[float] = []

    try:
        while True:
            t0 = time.perf_counter()

            esp32_decisions = uart.read_pending()
            allowed_esp32_decisions: list[GuidanceDecision] = []
            for decision in esp32_decisions:
                runtime_state.apply_esp32_decision(decision)
                if runtime_state.allows_esp32_decision(decision):
                    allowed_esp32_decisions.append(decision)
                    voice.speak_decision(decision)

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            result, annotated = detector.detect(frame)
            ai_decision = decision_engine.decide(result)
            ai_guidance_allowed = runtime_state.allows_ai_decision(ai_decision)
            active_decision = _choose_display_decision(
                ai_decision,
                allowed_esp32_decisions,
                ai_guidance_allowed,
            )

            result["ai_decision"] = ai_decision.to_dict()
            result["active_decision"] = active_decision.to_dict()
            result["runtime_state"] = runtime_state.mode_summary()
            result["ai_guidance_suppressed"] = (
                ai_decision.should_speak and not ai_guidance_allowed
            )
            if esp32_decisions:
                result["esp32_decisions"] = [
                    decision.to_dict() for decision in esp32_decisions
                ]
            if allowed_esp32_decisions:
                result["allowed_esp32_decisions"] = [
                    decision.to_dict() for decision in allowed_esp32_decisions
                ]

            if ai_guidance_allowed:
                voice.speak_decision(ai_decision)
            uart.send_ai_decision(ai_decision)

            dt = time.perf_counter() - t0
            fps = 1.0 / max(dt, 1e-6)
            frame_times.append(fps)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_fps = sum(frame_times) / len(frame_times)

            display_state = (result["risk_level"], active_decision.code)
            if display_state != previous_display_state:
                _print_status_change(result, active_decision)
                previous_display_state = display_state
            _print_running_line(result, active_decision, avg_fps)

            _write_latest_json(out_json, result)
            previous_log_state, last_log_time = _maybe_write_debug_log(
                log_path,
                result,
                display_state,
                previous_log_state,
                last_log_time,
            )

            if config.SHOW_WINDOW:
                _draw_overlay(annotated, result, active_decision, avg_fps)
                cv2.imshow("Li-Stick Vision", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        print()
        voice.stop()
        uart.close()
        cap.release()
        if config.SHOW_WINDOW:
            cv2.destroyAllWindows()
        print("\n[Done]")
        if config.ENABLE_DETECTION_OUTPUT_JSON:
            print(f"  Latest JSON: {out_json.resolve()}")
        if config.ENABLE_DEBUG_LOGGING:
            print(f"  Debug log  : {log_path.resolve()}")


if __name__ == "__main__":
    main()
