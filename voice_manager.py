from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import config
from decision_engine import GuidanceDecision


@dataclass(frozen=True)
class _CommandResult:
    succeeded: bool
    interrupted: bool = False
    error: str = ""


class VoiceManager:
    """
    Offline, non-blocking text-to-speech manager.
    Keeps only the latest useful guidance to avoid old delayed speech.
    NavigationManager owns frame-to-frame command stabilization.
    """

    def __init__(
        self,
        enabled: bool = True,
        cooldown_seconds: float = config.VOICE_COOLDOWN_SECONDS,
        emergency_cooldown_seconds: float = config.EMERGENCY_COOLDOWN_SECONDS,
        backend: str | None = None,
        timeout_seconds: float = config.TTS_TIMEOUT_SECONDS,
        piper_model_path: str | None = None,
        piper_config_path: str | None = None,
        piper_volume: float | None = None,
        piper_length_scale: float | None = None,
        piper_mode: str | None = None,
        piper_python_path: str | None = None,
    ):
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.emergency_cooldown_seconds = emergency_cooldown_seconds
        self.timeout_seconds = timeout_seconds
        self.piper_model_path = str(
            config.PIPER_MODEL_PATH
            if piper_model_path is None
            else piper_model_path
        )
        self.piper_config_path = str(
            config.PIPER_CONFIG_PATH
            if piper_config_path is None
            else piper_config_path
        )
        self.piper_volume = float(
            config.PIPER_VOLUME if piper_volume is None else piper_volume
        )
        self.piper_length_scale = float(
            config.PIPER_LENGTH_SCALE
            if piper_length_scale is None
            else piper_length_scale
        )
        self.piper_mode = str(
            config.PIPER_MODE if piper_mode is None else piper_mode
        ).strip().lower()
        self.piper_python_path = str(
            config.PIPER_PYTHON_PATH
            if piper_python_path is None
            else piper_python_path
        )

        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._process_lock = threading.Lock()
        self._speech_lock = threading.Lock()

        self._pending: list[GuidanceDecision] = []
        self._current: GuidanceDecision | None = None
        self._cancel_generation = 0
        self._current_generation = 0
        self._active_process: subprocess.Popen[str] | None = None
        self._active_synthesis_process: subprocess.Popen[str] | None = None
        self._active_playback_process: subprocess.Popen[str] | None = None
        self._interrupted_pids: set[int] = set()
        self._last_spoken_at: dict[str, float] = {}
        self._last_active_code: str | None = None
        self._piper_voice: object | None = None
        self._piper_syn_config: object | None = None

        self._thread: threading.Thread | None = None
        requested_backend = config.TTS_BACKEND if backend is None else backend
        self.backend = (
            self._select_backend(requested_backend) if self.enabled else "none"
        )

        if self.enabled and self.backend != "none":
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            if self.backend == "piper":
                if self.piper_mode == "cli":
                    print("[Voice] Piper CLI offline TTS enabled.")
                else:
                    print("[Voice] Piper offline TTS enabled.")
                print(f"[Voice] Model: {Path(self.piper_model_path).stem}")
            else:
                print(f"[Voice] Offline TTS enabled ({self.backend}).")
        else:
            self.enabled = False
            print("[Voice] Disabled.")

    def speak_decision(self, decision: GuidanceDecision) -> bool:
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
            last_spoken = self._last_spoken_at.get(decision.code, 0.0)
            cooldown_ready = now - last_spoken >= cooldown
            is_new_active = decision.code != self._last_active_code

            is_interrupt = (
                self._current is not None
                and decision.priority > self._current.priority
            )

            if self._current is not None and self._current.code == decision.code:
                return False
            if any(item.code == decision.code for item in self._pending):
                return False

            if not (is_new_active or cooldown_ready or is_interrupt):
                return False

            # Routine guidance must never replace an active or queued emergency.
            if not decision.is_emergency and (
                (self._current is not None and self._current.is_emergency)
                or any(item.is_emergency for item in self._pending)
            ):
                return False

            self._last_active_code = decision.code
            self._last_spoken_at[decision.code] = now

            # Invalidate a lower-priority synthesis as well as terminating any
            # active playback process. Piper inference itself is synchronous,
            # so the resulting routine WAV is discarded before it reaches ALSA.
            if is_interrupt:
                self._cancel_generation += 1

            if decision.is_emergency:
                # Discard queued navigation, retain distinct emergency events in
                # arrival order, and put higher-priority emergencies first.
                self._pending = [
                    item for item in self._pending if item.is_emergency
                ]
                self._pending.append(decision)
                self._pending.sort(key=lambda item: item.priority, reverse=True)
            else:
                # Normal guidance keeps only the latest useful command.
                self._pending.clear()
                self._pending.append(decision)
            self._wake.set()

        if is_interrupt:
            self._terminate_active_process()

        return True

    def speak_immediate(self, decision: GuidanceDecision) -> bool:
        if not self.enabled or not decision.should_speak:
            return False

        with self._lock:
            self._last_active_code = decision.code
            self._last_spoken_at[decision.code] = time.monotonic()

        print(f"\n[Voice] Speaking: {decision.message}")
        ok = self._perform_speech(str(decision.message))

        if ok:
            with self._lock:
                self._last_spoken_at[decision.code] = time.monotonic()

        return ok

    def cancel_current_and_pending(self):
        """Cancel current speech and discard queued decisions without stopping."""
        with self._lock:
            self._pending.clear()
            self._last_active_code = None
            self._cancel_generation += 1
            self._wake.clear()

        self._terminate_active_process()

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
                self._current_generation = self._cancel_generation

                if not self._pending:
                    self._wake.clear()

            print(f"\n[Voice] Speaking: {decision.message}")
            ok = self._perform_speech(str(decision.message))

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
        if requested == "piper":
            unavailable_reason = self._load_piper()
            if unavailable_reason is None:
                return "piper"
            return self._activate_espeak_fallback(unavailable_reason)

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

    def _load_piper(self) -> str | None:
        """Initialize the configured Piper mode or return a failure reason."""
        model_path = Path(self.piper_model_path)
        config_path = Path(self.piper_config_path)
        if not model_path.is_file():
            return f"model file not found: {model_path}"
        if not config_path.is_file():
            return f"configuration file not found: {config_path}"

        if self.piper_mode == "cli":
            python_path = Path(self.piper_python_path)
            if not python_path.is_file():
                return f"Python executable not found: {python_path}"
            return None

        if self.piper_mode != "api":
            return f"unknown Piper mode '{self.piper_mode}'"

        try:
            piper = importlib.import_module("piper")
            piper_voice_class = piper.PiperVoice
            synthesis_config_class = piper.SynthesisConfig
        except Exception as exc:
            return f"package import failed: {exc}"

        try:
            self._piper_voice = piper_voice_class.load(
                str(model_path),
                config_path=str(config_path),
                use_cuda=False,
            )
            self._piper_syn_config = synthesis_config_class(
                volume=self.piper_volume,
                length_scale=self.piper_length_scale,
            )
        except Exception as exc:
            self._piper_voice = None
            self._piper_syn_config = None
            return f"model loading failed: {exc}"

        return None

    def _select_espeak_fallback(self) -> str:
        if shutil.which("espeak-ng"):
            return "espeak-ng"
        if shutil.which("espeak"):
            return "espeak"
        print("[Voice] eSpeak fallback unavailable: executable not found.")
        return "none"

    def _activate_espeak_fallback(self, reason: str) -> str:
        if self.piper_mode == "cli":
            print(f"[Voice] Piper CLI synthesis failed: {reason}")
        else:
            print(f"[Voice] Piper unavailable: {reason}")
        print("[Voice] Falling back to eSpeak.")
        self._piper_voice = None
        self._piper_syn_config = None
        return self._select_espeak_fallback()

    def _perform_speech(self, message: str) -> bool:
        # A single lock covers model inference and playback, preventing overlap
        # with speak_immediate calls while the queue worker is active.
        with self._speech_lock:
            if self._worker_speech_was_cancelled():
                return False
            return self._speak_backend(message)

    def _speak_backend(self, message: str) -> bool:
        try:
            if self.backend == "piper":
                return self._speak_piper(message)
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

    def _speak_piper(self, message: str) -> bool:
        if self.piper_mode == "cli":
            return self._speak_piper_cli(message)
        return self._speak_piper_api(message)

    def _speak_piper_cli(self, message: str) -> bool:
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="listick-piper-",
                suffix=".wav",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)

            args = [
                self.piper_python_path,
                "-m",
                "piper",
                "--model",
                self.piper_model_path,
                "--config",
                self.piper_config_path,
                "--length-scale",
                f"{self.piper_length_scale:.2f}",
                "--volume",
                str(self.piper_volume),
                "--output_file",
                str(temp_path),
            ]

            synthesis_started = time.perf_counter()
            synthesis = self._run_command_result(
                args,
                input_text=message,
                process_kind="piper_synthesis",
            )
            synthesis_seconds = time.perf_counter() - synthesis_started
            if config.ENABLE_DEBUG_LOGGING:
                print(
                    f"[Voice] Piper synthesis time: "
                    f"{synthesis_seconds:.3f} seconds"
                )

            if synthesis.interrupted or self._worker_speech_was_cancelled():
                return False
            if not synthesis.succeeded:
                return self._fallback_and_speak_espeak(
                    message,
                    synthesis.error or "Piper process failed",
                )
            if not self._is_valid_wav(temp_path):
                return self._fallback_and_speak_espeak(
                    message,
                    "no valid WAV was produced",
                )

            playback = self._run_command_result(
                ["aplay", str(temp_path)],
                process_kind="playback",
            )
            if playback.succeeded:
                return True
            if playback.interrupted or self._worker_speech_was_cancelled():
                return False
            return self._fallback_and_speak_espeak(
                message,
                f"playback failed: {playback.error}",
            )
        finally:
            if temp_path is not None:
                self._delete_temporary_wav(temp_path)

    def _speak_piper_api(self, message: str) -> bool:
        if self._piper_voice is None or self._piper_syn_config is None:
            return self._fallback_and_speak_espeak(
                message,
                "voice model is not loaded",
            )

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="listick-piper-",
                suffix=".wav",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)

            try:
                with wave.open(str(temp_path), "wb") as wav_file:
                    self._piper_voice.synthesize_wav(
                        message,
                        wav_file,
                        syn_config=self._piper_syn_config,
                    )
            except Exception as exc:
                if self._worker_speech_was_cancelled():
                    return False
                return self._fallback_and_speak_espeak(
                    message,
                    f"synthesis failed: {exc}",
                )

            if self._worker_speech_was_cancelled():
                return False

            playback = self._run_command_result(
                ["aplay", str(temp_path)],
                process_kind="playback",
            )
            if playback.succeeded:
                return True
            if playback.interrupted or self._worker_speech_was_cancelled():
                return False
            return self._fallback_and_speak_espeak(
                message,
                f"playback failed: {playback.error}",
            )
        finally:
            if temp_path is not None:
                self._delete_temporary_wav(temp_path)

    @staticmethod
    def _is_valid_wav(path: Path) -> bool:
        try:
            if not path.is_file() or path.stat().st_size <= 44:
                return False
            with wave.open(str(path), "rb") as wav_file:
                return (
                    wav_file.getnchannels() > 0
                    and wav_file.getsampwidth() > 0
                    and wav_file.getframerate() > 0
                    and wav_file.getnframes() > 0
                )
        except (OSError, EOFError, wave.Error):
            return False

    @staticmethod
    def _delete_temporary_wav(path: Path):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            print(f"[Voice] Could not delete temporary WAV '{path}': {exc}")

    def _fallback_and_speak_espeak(self, message: str, reason: str) -> bool:
        fallback = self._activate_espeak_fallback(reason)
        self.backend = fallback
        if fallback == "none":
            self.enabled = False
            return False
        return self._speak_espeak(message, fallback)

    def _worker_speech_was_cancelled(self) -> bool:
        if threading.current_thread() is not self._thread:
            return False
        with self._lock:
            return (
                self._stop.is_set()
                or self._current_generation != self._cancel_generation
            )

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
            [
                command,
                "-s",
                str(config.TTS_RATE),
                "-v",
                str(config.TTS_VOICE),
                "-a",
                str(config.TTS_AMPLITUDE),
                message,
            ],
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
        result = self._run_command_result(
            args,
            input_text=input_text,
            creationflags=creationflags,
        )
        if not result.succeeded and not result.interrupted:
            print(f"[Voice] Backend '{self.backend}' failed: {result.error}")
        return result.succeeded

    def _run_command_result(
        self,
        args: list[str],
        input_text: str | None = None,
        creationflags: int = 0,
        process_kind: str = "backend",
    ) -> _CommandResult:
        stdin = subprocess.PIPE if input_text is not None else subprocess.DEVNULL

        with self._process_lock:
            if self._worker_speech_was_cancelled():
                return _CommandResult(False, interrupted=True)

            try:
                process = subprocess.Popen(
                    args,
                    stdin=stdin,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    creationflags=creationflags,
                    shell=False,
                )
            except OSError as exc:
                return _CommandResult(False, error=str(exc))
            self._active_process = process
            if process_kind == "piper_synthesis":
                self._active_synthesis_process = process
            elif process_kind == "playback":
                self._active_playback_process = process

        try:
            _, stderr = process.communicate(
                input=input_text,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            self._kill_process(process)
            with self._process_lock:
                if process.pid in self._interrupted_pids:
                    self._interrupted_pids.discard(process.pid)
                    return _CommandResult(False, interrupted=True)
            return _CommandResult(
                False,
                error=f"timed out after {self.timeout_seconds:.1f} seconds",
            )
        finally:
            with self._process_lock:
                if self._active_process is process:
                    self._active_process = None
                if self._active_synthesis_process is process:
                    self._active_synthesis_process = None
                if self._active_playback_process is process:
                    self._active_playback_process = None

        with self._process_lock:
            if process.pid in self._interrupted_pids:
                self._interrupted_pids.discard(process.pid)
                return _CommandResult(False, interrupted=True)

        if process.returncode == 0:
            return _CommandResult(True)

        stderr_text = (stderr or "").strip()
        error = f"returned {process.returncode}"
        if stderr_text:
            error = f"{error}: {stderr_text}"
        return _CommandResult(
            False,
            error=error,
        )

    def _terminate_active_process(self):
        with self._process_lock:
            process = self._active_process
            if process is None or process.poll() is not None:
                return

            if process.pid is not None:
                self._interrupted_pids.add(process.pid)

            try:
                process.terminate()
            except OSError:
                # The process may have exited between poll() and terminate().
                pass

    def _kill_process(self, process: subprocess.Popen[str]):
        if process.poll() is not None:
            return

        try:
            process.kill()
        except OSError:
            return
        try:
            process.communicate(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
