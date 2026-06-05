from __future__ import annotations

import json
import time
from typing import Any

import config
from decision_engine import GuidanceDecision, PRIORITY


ESP32_MESSAGES = {
    "CANE_ON": ("Cane is on", PRIORITY["CANE_ON"]),

    "SOS_HOLD_STARTED": ("Hold to send emergency alert", PRIORITY["SOS_SENT"]),
    "SOS_SENT": ("Emergency alert sent", PRIORITY["SOS_SENT"]),
    "SOS_CANCELLED": ("Emergency alert cancelled", PRIORITY["MODE_CHANGE"]),

    "FALL_DETECTED": ("Fall detected", PRIORITY["FALL_DETECTED"]),

    "HEAD_SENSOR_ALERT": ("Head obstacle", PRIORITY["HEAD_OBSTACLE"]),

    "SILENT_MODE_ON": ("Silent mode on", PRIORITY["MODE_CHANGE"]),
    "SILENT_MODE_OFF": ("Silent mode off", PRIORITY["MODE_CHANGE"]),

    "FULL_PAUSE_ON": ("Cane paused", PRIORITY["MODE_CHANGE"]),
    "FULL_PAUSE_OFF": ("Cane resumed", PRIORITY["MODE_CHANGE"]),

    "AI_PAUSE_ON": ("AI guidance paused", PRIORITY["MODE_CHANGE"]),
    "AI_PAUSE_OFF": ("AI guidance resumed", PRIORITY["MODE_CHANGE"]),

    "GPS_WEAK": ("GPS unavailable", PRIORITY["GPS_WEAK"]),
    "GPS_AVAILABLE": ("GPS available", PRIORITY["GPS_AVAILABLE"]),

    "WIFI_LOST": ("Connection lost", PRIORITY["WIFI_LOST"]),
    "WIFI_CONNECTED": ("Connection restored", PRIORITY["WIFI_CONNECTED"]),

    "BATTERY_LOW": ("Battery low", PRIORITY["BATTERY_LOW"]),
}

class UartBridge:
    """
    Optional non-blocking bridge between Raspberry Pi and ESP32.

    Raspberry Pi hardware-specific code starts here: on Pi 5 this normally uses
    /dev/serial0, while PC testing can leave UART disabled or point to a COM port.
    The vision loop continues if serial support, the cable, or the ESP32 fails.
    """

    def __init__(
        self,
        enabled: bool = True,
        port: str = config.UART_PORT,
        baudrate: int = config.UART_BAUDRATE,
    ):
        self.enabled = enabled
        self.port = port
        self.baudrate = baudrate
        self._serial: Any | None = None
        self._last_sent_code: str | None = None
        self._last_send_time = 0.0

        if not self.enabled:
            print("[UART] Disabled.")
            return

        try:
            import serial

            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0,
                write_timeout=0,
            )
            print(f"[UART] Connected on {self.port} @ {self.baudrate}.")
        except Exception as exc:
            self.enabled = False
            self._serial = None
            print(f"[UART] Unavailable: {exc}. Continuing without ESP32.")

    def read_pending(self, max_messages: int = 8) -> list[GuidanceDecision]:
        if not self.enabled or self._serial is None:
            return []

        decisions: list[GuidanceDecision] = []
        for _ in range(max_messages):
            try:
                raw = self._serial.readline()
            except Exception as exc:
                self._disable(f"read failed: {exc}")
                break

            if not raw:
                break

            decision = self._parse_line(raw)
            if decision is not None:
                decisions.append(decision)
        return decisions

    def send_ai_decision(self, decision: GuidanceDecision) -> bool:
        if (
            not config.UART_SEND_AI_DECISIONS
            or not self.enabled
            or self._serial is None
        ):
            return False

        now = time.monotonic()
        changed = decision.code != self._last_sent_code
        interval_elapsed = now - self._last_send_time >= config.UART_SEND_INTERVAL_SECONDS
        if not changed and not interval_elapsed:
            return False

        payload = {
            "type": "AI_DECISION",
            "code": decision.code,
            "message": decision.message,
            "risk_level": decision.risk_level,
            "priority": decision.priority,
        }
        try:
            self._serial.write((json.dumps(payload) + "\n").encode("utf-8"))
            self._last_sent_code = decision.code
            self._last_send_time = now
            return True
        except Exception as exc:
            self._disable(f"write failed: {exc}")
            return False

    def close(self):
        if self._serial is None:
            return
        try:
            self._serial.close()
        except Exception:
            pass

    def _parse_line(self, raw: bytes) -> GuidanceDecision | None:
        text = raw.decode("utf-8", errors="ignore").strip()
        if not text:
            return None

        code = self._extract_code(text)
        if code not in ESP32_MESSAGES:
            if config.ENABLE_DEBUG_LOGGING:
                print(f"\n[UART] Ignored message: {text}")
            return None

        message, priority = ESP32_MESSAGES[code]
        return GuidanceDecision(
            code=code,
            message=message,
            priority=priority,
            risk_level="external",
            source="esp32",
            details={"raw": text},
        )

    def _extract_code(self, text: str) -> str:
        if text.startswith("{"):
            try:
                data = json.loads(text)
                value = data.get("code") or data.get("event") or data.get("type") or ""
                return str(value).strip().upper()
            except json.JSONDecodeError:
                return ""

        normalized = text.replace(":", " ").replace(",", " ").strip().upper()
        return normalized.split()[0] if normalized else ""

    def _disable(self, reason: str):
        print(f"\n[UART] {reason}. Disabling UART bridge.")
        self.enabled = False
        self.close()
        self._serial = None
