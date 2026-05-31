import json
import sys
import time
from pathlib import Path

import cv2

import config
from alerter import BuzzerAlerter
from detector import ObstacleDetector

# ── Risk level display config ─────────────────────────────────────────────────

_RISK_META = {
    #           label              cv2 BGR colour
    "danger":      ("!! DANGER !!  OBSTACLE VERY NEAR — STOP!",    (0,   0,   230)),
    "alert":       ("!  ALERT     Object close — slow down",        (0,  100,  255)),
    "warning":     ("   WARNING   Object approaching",              (0,  200,  255)),
    "approaching": ("   DETECTED  Object ahead — monitoring",       (0,  220,  140)),
    "safe":        ("   CLEAR     Path safe",                       (0,  200,   60)),
}

_TREND_ARROW = {
    "approaching": " ▶  GETTING CLOSER",
    "receding":    " ◀  moving away",
    "stable":      " ■  stable",
}

BAR_WIDTH = 24


def _prox_bar(score: float) -> str:
    filled = round(score * BAR_WIDTH)
    return "█" * filled + "░" * (BAR_WIDTH - filled)


def _distance_label(score: float) -> str:
    if score >= config.THRESHOLD_DANGER:
        return "~<1 m"
    elif score >= config.THRESHOLD_ALERT:
        return "~1-1.5 m"
    elif score >= config.THRESHOLD_WARNING:
        return "~2-3 m"
    elif score >= config.THRESHOLD_APPROACHING:
        return "~3-5 m"
    return ">5 m"


def open_camera():
    cap = cv2.VideoCapture(config.CAMERA_ID)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera (id={config.CAMERA_ID}).")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


# ── Console message helpers ───────────────────────────────────────────────────

def _print_status_change(result: dict, prev_risk: str):
    """Print a formatted banner whenever the risk level changes."""
    risk     = result["risk_level"]
    label, _ = _RISK_META[risk]
    prox     = result["smooth_proximity"]
    trend    = result.get("trend", "stable")
    closest  = result["obstacles"][0] if result["obstacles"] else None
    obj_info = f"{closest['label'].upper()} in {closest['zone'].upper()}" if closest else "NO OBJECT"

    border = "━" * 52
    bar    = _prox_bar(prox)
    dist   = _distance_label(prox)

    print(f"\n{border}")
    print(f"  {label}")
    print(f"  {obj_info}   {dist}")
    print(f"  [{bar}] {prox:.2f}  {_TREND_ARROW.get(trend, '')}")
    print(border)


def _print_running_line(result: dict, fps: float):
    """Overwrite the same console line with a compact status."""
    risk  = result["risk_level"]
    prox  = result["smooth_proximity"]
    trend = result.get("trend", "stable")
    arrow = {"approaching": "▶", "receding": "◀", "stable": "■"}.get(trend, "■")
    bar   = _prox_bar(prox)
    label = risk.upper().ljust(11)
    closest = result["obstacles"][0] if result["obstacles"] else None
    obj   = f"{closest['label'][:8]:<8}·{closest['zone'][:1].upper()}" if closest else "none     "
    dist  = _distance_label(prox)

    line = (
        f"\r  [{label}] {obj}  [{bar}] {prox:.2f} {arrow}  "
        f"{dist:<9}  {fps:4.1f} fps  "
    )
    print(line, end="", flush=True)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 52)
    print("  Li-Stick Vision  —  Obstacle Sensor")
    print("═" * 52)
    print()
    print("  Distance zones (car-sensor style):")
    print("  >5 m   → CLEAR      silent")
    print("  ~4 m   → DETECTED   soft ping   every 2.5 s")
    print("  ~2-3 m → WARNING    slow beep   every 0.8 s")
    print("  ~1.5 m → ALERT      fast beep   every 0.35 s")
    print("  <1 m   → DANGER     rapid beep  every 0.18 s")
    print()
    print("  Press 'q' in the camera window to quit.\n")

    cap      = open_camera()
    detector = ObstacleDetector()
    alerter  = BuzzerAlerter(gpio_pin=config.BUZZER_PIN)

    out_json = Path(config.OUTPUT_JSON)
    log_path = Path(config.OUTPUT_LOG)

    prev_risk   = "safe"
    frame_times = []

    try:
        while True:
            t0 = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            result, annotated = detector.detect(frame)
            risk = result["risk_level"]

            # Drive buzzer
            alerter.set_risk(risk)

            # Console: banner on risk-level change, running line every frame
            if risk != prev_risk:
                _print_status_change(result, prev_risk)
                prev_risk = risk

            dt = time.perf_counter() - t0
            fps = 1.0 / max(dt, 1e-6)
            frame_times.append(fps)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_fps = sum(frame_times) / len(frame_times)

            _print_running_line(result, avg_fps)

            # Persist output
            out_json.write_text(json.dumps(result, indent=2))
            with log_path.open("a") as fh:
                fh.write(json.dumps(result) + "\n")

            # ── On-screen overlay ─────────────────────────────────────────────
            if config.SHOW_WINDOW:
                _, color = _RISK_META[risk]
                prox = result["smooth_proximity"]
                trend = result.get("trend", "stable")
                arrow = {"approaching": ">>", "receding": "<<", "stable": "--"}.get(trend, "--")
                label_str, _ = _RISK_META[risk]
                dist_str = _distance_label(prox)

                # Black background strip
                cv2.rectangle(annotated, (0, 0), (640, 70), (0, 0, 0), -1)

                # Risk label
                short_label = risk.upper()
                cv2.putText(annotated, short_label,
                            (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Distance + trend + fps
                info = f"{dist_str}  {arrow}  {avg_fps:.0f}fps"
                cv2.putText(annotated, info,
                            (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1)

                # Proximity bar (drawn as rectangle strip)
                bar_x2 = int(8 + prox * 300)
                cv2.rectangle(annotated, (8, 62), (308, 68), (60, 60, 60), -1)
                cv2.rectangle(annotated, (8, 62), (bar_x2, 68), color, -1)

                cv2.imshow("Li-Stick Vision", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        print()   # newline after running status line
        alerter.stop()
        cap.release()
        if config.SHOW_WINDOW:
            cv2.destroyAllWindows()
        print("\n[Done]")
        print(f"  Latest : {out_json.resolve()}")
        print(f"  Log    : {log_path.resolve()}")


if __name__ == "__main__":
    main()
