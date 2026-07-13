from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import config
from decision_engine import GuidanceDecision, PRIORITY


ESP32_MESSAGES = {
    "CANE_ON": ("Cane is on", PRIORITY["CANE_ON"]),

    "SOS_HOLD_STARTED": ("Hold for emergency", PRIORITY["SOS_HOLD_STARTED"]),
    "SOS_CANCELLED": ("Emergency cancelled", PRIORITY["SOS_CANCELLED"]),
    "SOS_REQUESTED": ("Sending emergency alert", PRIORITY["SOS_REQUESTED"]),
    "SOS_DELIVERED_WITH_LOCATION": (
        "Emergency alert sent with location",
        PRIORITY["SOS_DELIVERED_WITH_LOCATION"],
    ),
    "SOS_DELIVERED_WITHOUT_LOCATION": (
        "Emergency alert sent. Location unavailable",
        PRIORITY["SOS_DELIVERED_WITHOUT_LOCATION"],
    ),
    "SOS_FAILED_NO_CONNECTION": (
        "No connection. Retrying emergency alert",
        PRIORITY["SOS_FAILED_NO_CONNECTION"],
    ),
    "SOS_DELIVERY_FAILED": (
        "Emergency delivery failed. Retrying",
        PRIORITY["SOS_DELIVERY_FAILED"],
    ),
    # Temporary compatibility for older ESP32 firmware.
    "SOS_SENT": ("Emergency alert sent", PRIORITY["SOS_SENT"]),

    "FALL_DETECTED": ("Fall detected", PRIORITY["FALL_DETECTED"]),

    "HEAD_SENSOR_ALERT": ("Head obstacle", PRIORITY["HEAD_SENSOR_ALERT"]),

    "SILENT_MODE_ON": ("Silent mode on", PRIORITY["MODE_CHANGE"]),
    "SILENT_MODE_OFF": ("Silent mode off", PRIORITY["MODE_CHANGE"]),

    "FULL_PAUSE_ON": ("Cane paused", PRIORITY["MODE_CHANGE"]),
    "FULL_PAUSE_OFF": ("Cane resumed", PRIORITY["MODE_CHANGE"]),

    "AI_PAUSE_ON": ("AI guidance paused", PRIORITY["MODE_CHANGE"]),
    "AI_PAUSE_OFF": ("AI guidance resumed", PRIORITY["MODE_CHANGE"]),

    "BATTERY_LOW": ("Battery low", PRIORITY["BATTERY_LOW"]),
}

SOS_DUPLICATE_SUPPRESSION_SECONDS = {
    "SOS_REQUESTED": 3.0,
    "SOS_DELIVERED_WITH_LOCATION": 10.0,
    "SOS_DELIVERED_WITHOUT_LOCATION": 10.0,
    "SOS_FAILED_NO_CONNECTION": 10.0,
    "SOS_DELIVERY_FAILED": 10.0,
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
        clock: Callable[[], float] = time.monotonic,
    ):
        self.enabled = enabled
        self.port = port
        self.baudrate = baudrate
        self._serial: Any | None = None
        self._last_sent_code: str | None = None
        self._last_send_time = 0.0
        self._clock = clock
        self._last_received_at: dict[str, float] = {}

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

    @property
    def is_connected(self) -> bool:
        return self.enabled and self._serial is not None

    def send_ai_decision(self, decision: GuidanceDecision) -> bool:
        if (
            not config.UART_SEND_AI_DECISIONS
            or not self.enabled
            or self._serial is None
        ):
            return False

        now = self._clock()
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
        print(f"\n[UART] Event received: {code or text}")
        if code not in ESP32_MESSAGES:
            print(f"[UART] Unknown message ignored: {text}")
            return None

        now = self._clock()
        suppression_seconds = SOS_DUPLICATE_SUPPRESSION_SECONDS.get(code, 0.0)
        last_received = self._last_received_at.get(code)
        if (
            suppression_seconds > 0.0
            and last_received is not None
            and now - last_received < suppression_seconds
        ):
            print(f"[UART] Duplicate suppressed: {code}")
            return None
        self._last_received_at[code] = now

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
