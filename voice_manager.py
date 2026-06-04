from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import threading
import time

import config
from decision_engine import GuidanceDecision


class VoiceManager:
    """
    Offline, non-blocking text-to-speech manager.

    Raspberry Pi audio hardware-specific setup starts here if the final build
    needs a USB speaker, audio HAT, or custom ALSA device selection.

    The camera loop only calls speak_decision(). Actual speech runs in a worker
    thread and every backend is isolated behind a short timeout. This keeps one
    bad TTS call from freezing the guidance loop.
    """

    def __init__(
        self,
        enabled: bool = True,
        cooldown_seconds: float = config.VOICE_COOLDOWN_SECONDS,
        emergency_cooldown_seconds: float = config.EMERGENCY_COOLDOWN_SECONDS,
        backend: str = config.TTS_BACKEND,
        timeout_seconds: float = config.TTS_TIMEOUT_SECONDS,
    ):
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.emergency_cooldown_seconds = emergency_cooldown_seconds
        self.backend = self._select_backend(backend)
        self.timeout_seconds = timeout_seconds

        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._process_lock = threading.Lock()
        self._pending: list[GuidanceDecision] = []
        self._current: GuidanceDecision | None = None
        self._active_process: subprocess.Popen[str] | None = None
        self._interrupted_pids: set[int] = set()
        self._last_spoken_at: dict[str, float] = {}
        self._last_active_code: str | None = None

        self._thread: threading.Thread | None = None
        if self.enabled and self.backend != "none":
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            print(f"[Voice] Offline TTS enabled ({self.backend}).")
        else:
            self.enabled = False
            print("[Voice] Disabled.")

    def speak_decision(self, decision: GuidanceDecision) -> bool:
        """
        Queue the active guidance when it is new, urgent, or ready to repeat.

        Repeated messages are intentionally allowed after the configured
        cooldown while the same obstacle remains active. SAFE clears the active
        command and stays silent.
        """
        if not self.enabled:
            return False

        if not decision.should_speak:
            with self._lock:
                self._last_active_code = None
                self._pending.clear()
            return False

        now = time.monotonic()
        cooldown = self._cooldown_for(decision)

        with self._lock:
            is_new_active = decision.code != self._last_active_code
            last_spoken = self._last_spoken_at.get(decision.code, 0.0)
            cooldown_ready = now - last_spoken >= cooldown
            is_interrupt = (
                self._current is not None
                and decision.priority > self._current.priority
            )

            if self._current is not None and self._current.code == decision.code:
                return False

            if not (is_new_active or cooldown_ready or is_interrupt):
                return False

            if any(item.code == decision.code for item in self._pending):
                return False

            self._last_active_code = decision.code
            self._last_spoken_at[decision.code] = now

            if is_interrupt:
                self._pending.clear()
            else:
                self._pending = [
                    item
                    for item in self._pending
                    if item.priority > decision.priority
                ]

            self._pending.append(decision)
            self._pending.sort(key=lambda item: item.priority, reverse=True)
            self._wake.set()

        if is_interrupt:
            self._terminate_active_process()
        return True

    def stop(self):
        self._stop.set()
        self._wake.set()
        self._terminate_active_process()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self):
        while not self._stop.is_set():
            self._wake.wait(timeout=0.1)
            if self._stop.is_set():
                break

            with self._lock:
                if not self._pending:
                    self._wake.clear()
                    continue
                decision = self._pending.pop(0)
                self._current = decision
                if not self._pending:
                    self._wake.clear()

            print(f"\n[Voice] Speaking: {decision.message}")
            ok = self._speak_backend(str(decision.message))

            with self._lock:
                if ok:
                    self._last_spoken_at[decision.code] = time.monotonic()
                self._current = None

    def _cooldown_for(self, decision: GuidanceDecision) -> float:
        if decision.is_emergency:
            return self.emergency_cooldown_seconds
        return self.cooldown_seconds

    def _select_backend(self, requested: str) -> str:
        requested = (requested or "auto").lower()
        if requested != "auto":
            return requested

        system = platform.system()
        if system == "Windows":
            return "windows_sapi"
        if system == "Darwin":
            return "say" if shutil.which("say") else "pyttsx3"
        if shutil.which("espeak-ng"):
            return "espeak-ng"
        if shutil.which("espeak"):
            return "espeak"
        return "pyttsx3"

    def _speak_backend(self, message: str) -> bool:
        try:
            if self.backend == "windows_sapi":
                return self._speak_windows_sapi(message)
            if self.backend in {"espeak", "espeak-ng"}:
                return self._speak_espeak(message, self.backend)
            if self.backend == "say":
                return self._run_command(["say", message])
            if self.backend == "pyttsx3":
                return self._speak_pyttsx3_subprocess(message)
            print(f"[Voice] Unknown TTS backend '{self.backend}'. Disabling voice.")
            self.enabled = False
            return False
        except Exception as exc:
            print(f"[Voice] Speech failed: {exc}. Continuing silently.")
            return False

    def _speak_windows_sapi(self, message: str) -> bool:
        script = """
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 0
$text = [Console]::In.ReadToEnd()
$synth.Speak($text)
"""
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return self._run_command(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            input_text=message,
            creationflags=creationflags,
        )

    def _speak_espeak(self, message: str, command: str) -> bool:
        return self._run_command(
            [command, "-s", str(config.TTS_RATE), message],
        )

    def _speak_pyttsx3_subprocess(self, message: str) -> bool:
        script = """
import sys
import pyttsx3

text = sys.stdin.read()
engine = pyttsx3.init()
engine.setProperty("rate", %d)
engine.say(text)
engine.runAndWait()
""" % config.TTS_RATE
        return self._run_command(
            [sys.executable, "-c", script],
            input_text=message,
        )

    def _run_command(
        self,
        args: list[str],
        input_text: str | None = None,
        creationflags: int = 0,
    ) -> bool:
        stdin = subprocess.PIPE if input_text is not None else subprocess.DEVNULL
        process = subprocess.Popen(
            args,
            stdin=stdin,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creationflags,
        )

        with self._process_lock:
            self._active_process = process

        try:
            _, stderr = process.communicate(
                input=input_text,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            self._kill_process(process)
            print(f"[Voice] Backend '{self.backend}' timed out.")
            return False
        finally:
            with self._process_lock:
                if self._active_process is process:
                    self._active_process = None

        if process.pid in self._interrupted_pids:
            self._interrupted_pids.discard(process.pid)
            return False

        if process.returncode == 0:
            return True
        stderr_text = (stderr or "").strip()
        print(f"[Voice] Backend '{self.backend}' returned {process.returncode}: {stderr_text}")
        return False

    def _terminate_active_process(self):
        with self._process_lock:
            process = self._active_process
            if process is None or process.poll() is not None:
                return
            if process.pid is not None:
                self._interrupted_pids.add(process.pid)
            process.terminate()

    def _kill_process(self, process: subprocess.Popen[str]):
        if process.poll() is not None:
            return
        process.kill()
        try:
            process.communicate(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
