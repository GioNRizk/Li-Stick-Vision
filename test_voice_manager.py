from __future__ import annotations

import importlib
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import Mock

import config
import voice_manager as voice_manager_module
from decision_engine import GuidanceDecision, PRIORITY
from voice_manager import VoiceManager


class _FakePiperVoice:
    def __init__(self, synthesis_error: Exception | None = None):
        self.synthesis_error = synthesis_error
        self.messages: list[str] = []

    def synthesize_wav(self, message, wav_file, syn_config=None):
        self.messages.append(message)
        if self.synthesis_error is not None:
            raise self.synthesis_error
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00\x00" * 32)


class _ImmediateProcess:
    _next_pid = 1000

    def __init__(self, returncode: int = 0, stderr: str = ""):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.returncode = returncode
        self.stderr = stderr
        self.terminate_called = False
        self.kill_called = False

    def communicate(self, input=None, timeout=None):
        return "", self.stderr

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_called = True
        self.returncode = -15

    def kill(self):
        self.kill_called = True
        self.returncode = -9


class _BlockingProcess:
    _next_pid = 2000

    def __init__(self, args):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.args = args
        self.returncode: int | None = None
        self.terminate_called = False
        self.kill_called = False
        self._finished = threading.Event()

    def communicate(self, input=None, timeout=None):
        if not self._finished.wait(timeout):
            raise subprocess.TimeoutExpired(self.args, timeout)
        return "", ""

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_called = True
        self.returncode = -15
        self._finished.set()

    def kill(self):
        self.kill_called = True
        self.returncode = -9
        self._finished.set()


def _voice_paths(tmp_path: Path) -> tuple[Path, Path]:
    model_path = tmp_path / "en_US-lessac-medium.onnx"
    config_path = tmp_path / "en_US-lessac-medium.onnx.json"
    model_path.write_bytes(b"fake model")
    config_path.write_text("{}", encoding="utf-8")
    return model_path, config_path


def _install_fake_piper(
    monkeypatch,
    voice: _FakePiperVoice | None = None,
    load_error: Exception | None = None,
):
    fake_voice = voice or _FakePiperVoice()
    load_calls: list[tuple[str, dict]] = []
    synthesis_configs: list[object] = []

    class FakePiperVoiceClass:
        @staticmethod
        def load(model_path, **kwargs):
            load_calls.append((model_path, kwargs))
            if load_error is not None:
                raise load_error
            return fake_voice

    class FakeSynthesisConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            synthesis_configs.append(self)

    fake_module = types.SimpleNamespace(
        PiperVoice=FakePiperVoiceClass,
        SynthesisConfig=FakeSynthesisConfig,
    )
    monkeypatch.setitem(sys.modules, "piper", fake_module)
    return fake_voice, load_calls, synthesis_configs


def _force_espeak_fallback(monkeypatch):
    monkeypatch.setattr(
        voice_manager_module.shutil,
        "which",
        lambda command: "/usr/bin/espeak" if command == "espeak" else None,
    )


def _decision(
    code: str,
    message: str,
    priority: int,
    source: str = "ai",
) -> GuidanceDecision:
    return GuidanceDecision(
        code=code,
        message=message,
        priority=priority,
        risk_level="warning",
        source=source,
    )


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _new_piper_manager(
    tmp_path: Path,
    *,
    backend: str | None = "piper",
) -> VoiceManager:
    model_path, config_path = _voice_paths(tmp_path)
    return VoiceManager(
        enabled=True,
        backend=backend,
        cooldown_seconds=0.0,
        emergency_cooldown_seconds=0.0,
        piper_model_path=str(model_path),
        piper_config_path=str(config_path),
        piper_volume=1.5,
        piper_length_scale=0.90,
    )


def test_piper_loads_once_and_becomes_active(monkeypatch, tmp_path, capsys):
    _, load_calls, synthesis_configs = _install_fake_piper(monkeypatch)
    voice = _new_piper_manager(tmp_path)
    try:
        assert voice.backend == "piper"
        assert len(load_calls) == 1
        model_path, load_kwargs = load_calls[0]
        assert model_path.endswith("en_US-lessac-medium.onnx")
        assert load_kwargs["config_path"].endswith(
            "en_US-lessac-medium.onnx.json"
        )
        assert load_kwargs["use_cuda"] is False
        assert synthesis_configs[0].kwargs == {
            "volume": 1.5,
            "length_scale": 0.90,
        }
        output = capsys.readouterr().out
        assert "[Voice] Piper offline TTS enabled." in output
        assert "[Voice] Model: en_US-lessac-medium" in output
    finally:
        voice.stop()


def test_missing_piper_model_falls_back_to_espeak(
    monkeypatch,
    tmp_path,
    capsys,
):
    _install_fake_piper(monkeypatch)
    _force_espeak_fallback(monkeypatch)
    missing_model = tmp_path / "missing.onnx"
    config_path = tmp_path / "missing.onnx.json"
    config_path.write_text("{}", encoding="utf-8")

    voice = VoiceManager(
        enabled=True,
        backend="piper",
        piper_model_path=str(missing_model),
        piper_config_path=str(config_path),
    )
    try:
        assert voice.backend == "espeak"
        output = capsys.readouterr().out
        assert "[Voice] Piper unavailable: model file not found:" in output
        assert "[Voice] Falling back to eSpeak." in output
    finally:
        voice.stop()


def test_missing_piper_config_falls_back_to_espeak(monkeypatch, tmp_path):
    _install_fake_piper(monkeypatch)
    _force_espeak_fallback(monkeypatch)
    model_path = tmp_path / "voice.onnx"
    model_path.write_bytes(b"fake model")

    voice = VoiceManager(
        enabled=True,
        backend="piper",
        piper_model_path=str(model_path),
        piper_config_path=str(tmp_path / "missing.json"),
    )
    try:
        assert voice.backend == "espeak"
    finally:
        voice.stop()


def test_piper_import_failure_falls_back_to_espeak(monkeypatch, tmp_path):
    _force_espeak_fallback(monkeypatch)
    real_import_module = voice_manager_module.importlib.import_module

    def fail_piper_import(name, package=None):
        if name == "piper":
            raise ImportError("piper test import failure")
        return real_import_module(name, package)

    monkeypatch.setattr(
        voice_manager_module.importlib,
        "import_module",
        fail_piper_import,
    )
    voice = _new_piper_manager(tmp_path)
    try:
        assert voice.backend == "espeak"
    finally:
        voice.stop()


def test_piper_model_load_failure_falls_back_to_espeak(monkeypatch, tmp_path):
    _install_fake_piper(
        monkeypatch,
        load_error=RuntimeError("bad ONNX model"),
    )
    _force_espeak_fallback(monkeypatch)
    voice = _new_piper_manager(tmp_path)
    try:
        assert voice.backend == "espeak"
    finally:
        voice.stop()


def test_piper_synthesis_failure_speaks_once_with_espeak(
    monkeypatch,
    tmp_path,
):
    _install_fake_piper(
        monkeypatch,
        voice=_FakePiperVoice(RuntimeError("inference failed")),
    )
    _force_espeak_fallback(monkeypatch)
    voice = _new_piper_manager(tmp_path)
    espeak = Mock(return_value=True)
    monkeypatch.setattr(voice, "_speak_espeak", espeak)

    try:
        decision = _decision("AI_READY", "Li-Stick ready", PRIORITY["AI_READY"])
        assert voice.speak_immediate(decision) is True
        assert voice.backend == "espeak"
        espeak.assert_called_once_with("Li-Stick ready", "espeak")
    finally:
        voice.stop()


def test_piper_playback_failure_falls_back_to_espeak(monkeypatch, tmp_path):
    _install_fake_piper(monkeypatch)
    _force_espeak_fallback(monkeypatch)
    monkeypatch.setattr(
        voice_manager_module.subprocess,
        "Popen",
        lambda args, **kwargs: _ImmediateProcess(
            returncode=1,
            stderr="device unavailable",
        ),
    )
    voice = _new_piper_manager(tmp_path)
    espeak = Mock(return_value=True)
    monkeypatch.setattr(voice, "_speak_espeak", espeak)

    try:
        decision = _decision("AI_READY", "Li-Stick ready", PRIORITY["AI_READY"])
        assert voice.speak_immediate(decision) is True
        assert voice.backend == "espeak"
        espeak.assert_called_once_with("Li-Stick ready", "espeak")
    finally:
        voice.stop()


def test_startup_speech_is_produced_once(monkeypatch):
    if "main" not in sys.modules:
        cv2_stub = types.ModuleType("cv2")
        camera_stub = types.ModuleType("camera_source")
        detector_stub = types.ModuleType("detector")

        class CameraOpenError(Exception):
            pass

        camera_stub.CameraOpenError = CameraOpenError
        camera_stub.open_camera = lambda: None
        detector_stub.ObstacleDetector = object
        monkeypatch.setitem(sys.modules, "cv2", cv2_stub)
        monkeypatch.setitem(sys.modules, "camera_source", camera_stub)
        monkeypatch.setitem(sys.modules, "detector", detector_stub)

    main_module = importlib.import_module("main")

    class Navigation:
        def update(self, decision):
            return decision

    class Voice:
        def __init__(self):
            self.messages = []

        def speak_immediate(self, decision):
            self.messages.append(decision.message)
            return True

    voice = Voice()
    assert main_module._announce_ready(voice, Navigation()) is True
    assert voice.messages == ["Li-Stick ready"]


def test_emergency_interrupts_routine_piper_playback(monkeypatch, tmp_path):
    fake_voice, _, _ = _install_fake_piper(monkeypatch)
    processes: list[_BlockingProcess] = []

    def fake_popen(args, **kwargs):
        assert args[0] == "aplay"
        process = _BlockingProcess(args)
        processes.append(process)
        return process

    monkeypatch.setattr(
        voice_manager_module.subprocess,
        "Popen",
        fake_popen,
    )
    voice = _new_piper_manager(tmp_path)
    routine = _decision(
        "PERSON_AHEAD",
        "Person ahead",
        PRIORITY["PERSON_AHEAD"],
    )
    emergency = _decision(
        "FALL_DETECTED",
        "Fall detected",
        PRIORITY["FALL_DETECTED"],
        source="esp32",
    )

    try:
        assert voice.speak_decision(routine) is True
        assert _wait_until(lambda: len(processes) == 1)
        routine_process = processes[0]
        assert voice.speak_decision(emergency) is True
        assert _wait_until(lambda: routine_process.terminate_called)
        assert _wait_until(lambda: "Fall detected" in fake_voice.messages)
        assert _wait_until(lambda: len(processes) == 2)
        assert processes[0].args[1] != processes[1].args[1]
    finally:
        voice.stop()


def test_temporary_wav_is_deleted(monkeypatch, tmp_path):
    _install_fake_piper(monkeypatch)
    playback_paths: list[Path] = []

    def fake_popen(args, **kwargs):
        path = Path(args[1])
        assert path.is_file()
        playback_paths.append(path)
        return _ImmediateProcess()

    monkeypatch.setattr(
        voice_manager_module.subprocess,
        "Popen",
        fake_popen,
    )
    voice = _new_piper_manager(tmp_path)
    try:
        decision = _decision("AI_READY", "Li-Stick ready", PRIORITY["AI_READY"])
        assert voice.speak_immediate(decision) is True
        assert len(playback_paths) == 1
        assert not playback_paths[0].exists()
    finally:
        voice.stop()


def test_subprocess_playback_never_uses_shell_true(monkeypatch, tmp_path):
    _install_fake_piper(monkeypatch)
    popen_calls: list[tuple[list[str], dict]] = []

    def fake_popen(args, **kwargs):
        popen_calls.append((args, kwargs))
        return _ImmediateProcess()

    monkeypatch.setattr(
        voice_manager_module.subprocess,
        "Popen",
        fake_popen,
    )
    voice = _new_piper_manager(tmp_path)
    try:
        decision = _decision("STOP", "Stop", PRIORITY["STOP"])
        assert voice.speak_immediate(decision) is True
        assert popen_calls[0][0][0] == "aplay"
        assert popen_calls[0][1].get("shell") is not True
    finally:
        voice.stop()


def test_backend_and_piper_settings_are_read_from_config(monkeypatch, tmp_path):
    _, _, synthesis_configs = _install_fake_piper(monkeypatch)
    model_path, config_path = _voice_paths(tmp_path)
    monkeypatch.setattr(config, "TTS_BACKEND", "piper")
    monkeypatch.setattr(config, "PIPER_MODEL_PATH", str(model_path))
    monkeypatch.setattr(config, "PIPER_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(config, "PIPER_VOLUME", 1.25)
    monkeypatch.setattr(config, "PIPER_LENGTH_SCALE", 0.85)

    voice = VoiceManager(enabled=True)
    try:
        assert voice.backend == "piper"
        assert synthesis_configs[0].kwargs == {
            "volume": 1.25,
            "length_scale": 0.85,
        }
    finally:
        voice.stop()
