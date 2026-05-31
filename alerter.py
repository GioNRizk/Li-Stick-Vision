"""
Car-sensor style buzzer alerter.

Mimics a car parking sensor: silence when clear, then increasing beep
frequency as the obstacle gets closer, until a continuous tone at danger.

  SAFE        → silent
  APPROACHING → soft ping  every 2.5 s   (object detected far ahead)
  WARNING     → beep       every 0.8 s   (object approaching)
  ALERT       → fast beep  every 0.35 s  (object close — stay alert)
  DANGER      → rapid beep every 0.12 s  (object very near — STOP)
"""

import math
import platform
import struct
import threading


# ── WAV generator ─────────────────────────────────────────────────────────────

def _make_wav(freq: int, duration_ms: int, volume: float = 0.75) -> bytes:
    """Return a PCM WAV buffer for a sine-wave tone (no extra packages)."""
    sample_rate = 44100
    n = int(sample_rate * duration_ms / 1000)
    data_size = n * 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, 1,
        sample_rate, sample_rate * 2, 2, 16,
        b"data", data_size,
    )
    samples = bytearray(data_size)
    for i in range(n):
        v = int(volume * 32767 * math.sin(2 * math.pi * freq * i / sample_rate))
        struct.pack_into("<h", samples, i * 2, v)
    return header + bytes(samples)


# ── Alerter ───────────────────────────────────────────────────────────────────

class BuzzerAlerter:
    # risk → (freq_hz, beep_ms, gap_ms)
    # gap_ms is the silence between beeps — decreasing gap = higher urgency
    PATTERNS: dict[str, tuple[int, int, int] | None] = {
        "safe":        None,
        "approaching": (480,  120, 2380),   # soft ping every 2.5 s
        "warning":     (620,  140,  660),   # steady pulse every 0.8 s
        "alert":       (780,  110,  240),   # fast beep every 0.35 s
        "danger":      (960,   90,   90),   # rapid every 0.18 s (car sensor ≈ continuous)
    }

    def __init__(self, gpio_pin: int = 18):
        self._risk    = "safe"
        self._lock    = threading.Lock()
        self._stop    = threading.Event()
        self._backend = self._detect_backend(gpio_pin)
        self._wavs: dict[str, bytes] = {}
        if self._backend == "winsound":
            self._pregenerate()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[Alerter] Backend : {self._backend}")

    # ── Backend detection ─────────────────────────────────────────────────────

    def _detect_backend(self, pin: int) -> str:
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.OUT)
            self._pwm      = GPIO.PWM(pin, 1000)
            self._gpio_pin = pin
            return "gpio"
        except Exception:
            pass
        if platform.system() == "Windows":
            try:
                import winsound as _w  # noqa: F401
                return "winsound"
            except ImportError:
                pass
        return "none"

    def _pregenerate(self):
        for risk, pat in self.PATTERNS.items():
            if pat:
                freq, on_ms, _ = pat
                self._wavs[risk] = _make_wav(freq, on_ms)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_risk(self, level: str):
        with self._lock:
            self._risk = level

    def stop(self):
        self._stop.set()
        if self._backend == "gpio":
            try:
                import RPi.GPIO as GPIO
                self._pwm.stop()
                GPIO.cleanup(self._gpio_pin)
            except Exception:
                pass

    # ── Sound loop ────────────────────────────────────────────────────────────

    def _play_once(self, risk: str):
        pat = self.PATTERNS.get(risk)
        if pat is None:
            return
        _, _, gap_ms = pat

        if self._backend == "winsound":
            import winsound
            wav = self._wavs.get(risk)
            if wav:
                winsound.PlaySound(wav, winsound.SND_MEMORY)
            self._stop.wait(timeout=gap_ms / 1000)

        elif self._backend == "gpio":
            freq, on_ms, _ = pat
            self._pwm.ChangeFrequency(freq)
            self._pwm.start(50)
            self._stop.wait(timeout=on_ms / 1000)
            self._pwm.stop()
            self._stop.wait(timeout=gap_ms / 1000)

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                risk = self._risk
            if self.PATTERNS.get(risk) is None:   # "safe" → silence
                self._stop.wait(timeout=0.05)
            else:
                self._play_once(risk)
