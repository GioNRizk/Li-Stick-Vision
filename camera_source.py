from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import cv2

import config


class CameraOpenError(RuntimeError):
    pass


class OpenCVCamera:
    backend_name = "opencv"

    def __init__(self):
        self._cap = cv2.VideoCapture(config.CAMERA_ID)
        if not self._cap.isOpened():
            raise CameraOpenError(f"OpenCV cannot open camera id={config.CAMERA_ID}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    def read(self) -> tuple[bool, Any]:
        return self._cap.read()

    def release(self):
        self._cap.release()


class Picamera2Camera:
    backend_name = "picamera2"

    def __init__(self):
        # Raspberry Pi hardware-specific camera code starts here. Picamera2 uses
        # the libcamera stack that rpicam-hello also uses, so it works with the
        # Camera Module v2 IMX219 when OpenCV VideoCapture(0) does not.
        from picamera2 import Picamera2

        self._picam2 = Picamera2()
        self._format = config.PICAMERA2_FORMAT
        self._configure(self._format)
        self._picam2.start()

    def read(self) -> tuple[bool, Any]:
        try:
            frame = self._picam2.capture_array()
            if frame is None:
                return False, None

            # BGR888 is already detector/display compatible. If someone changes
            # config to RGB888, convert it once here and keep detector.py clean.
            if self._format.upper() == "RGB888":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            return True, frame.copy()
        except Exception as exc:
            print(f"\n[Camera] Picamera2 capture failed: {exc}")
            return False, None

    def release(self):
        try:
            self._picam2.stop()
        except Exception:
            pass
        try:
            self._picam2.close()
        except Exception:
            pass

    def _configure(self, pixel_format: str):
        try:
            camera_config = self._picam2.create_preview_configuration(
                main={
                    "format": pixel_format,
                    "size": (config.FRAME_WIDTH, config.FRAME_HEIGHT),
                }
            )
            self._picam2.configure(camera_config)
        except Exception:
            if pixel_format.upper() == "RGB888":
                raise
            print(
                f"[Camera] Picamera2 format {pixel_format} unavailable; "
                "retrying with RGB888."
            )
            self._format = "RGB888"
            self._configure(self._format)


def open_camera():
    requested = (config.CAMERA_BACKEND or "auto").lower()
    if requested not in {"auto", "opencv", "picamera2"}:
        raise CameraOpenError(
            f"Invalid CAMERA_BACKEND={config.CAMERA_BACKEND!r}; "
            "use auto, opencv, or picamera2."
        )

    errors: list[str] = []
    for backend in _backend_order(requested):
        try:
            camera = _open_backend(backend)
            print(f"[Camera] Backend: {camera.backend_name}")
            return camera
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
            print(f"[Camera] {backend} unavailable: {exc}")

    raise CameraOpenError("No camera backend available. " + " | ".join(errors))


def _backend_order(requested: str) -> list[str]:
    if requested == "opencv":
        return ["opencv"]
    if requested == "picamera2":
        return ["picamera2", "opencv"]
    if _looks_like_raspberry_pi():
        return ["picamera2", "opencv"]
    return ["opencv", "picamera2"]


def _open_backend(backend: str):
    if backend == "picamera2":
        return Picamera2Camera()
    if backend == "opencv":
        return OpenCVCamera()
    raise CameraOpenError(f"Unknown backend {backend!r}")


def _looks_like_raspberry_pi() -> bool:
    if platform.system() != "Linux":
        return False

    model_path = Path("/proc/device-tree/model")
    try:
        model = model_path.read_text(errors="ignore").lower()
        if "raspberry pi" in model:
            return True
    except Exception:
        pass

    return False
