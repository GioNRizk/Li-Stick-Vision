# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_ID    = 0        # 0 = first USB/Pi camera
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
CAMERA_BACKEND = "auto"          # auto, opencv, picamera2
PICAMERA2_FORMAT = "BGR888"      # BGR888 feeds detector/display without conversion

# ── Model ─────────────────────────────────────────────────────────────────────
YOLO_MODEL      = "yolov8n.pt"   # nano — fastest on Pi 4/5
YOLO_CONFIDENCE = 0.50           # ignore detections below this confidence
YOLO_CLASSES    = [
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    13,  # bench
    15,  # cat
    16,  # dog
    24,  # backpack
    25,  # umbrella
    28,  # suitcase
    56,  # chair
    57,  # couch
    58,  # potted plant
    59,  # bed
    60,  # dining table
]

# ── Depth model ───────────────────────────────────────────────────────────────
USE_DEPTH_MODEL  = False          # True once MiDaS weights are fully downloaded
DEPTH_SKIP_FRAMES = 2

# ── Proximity zones (0.0–1.0 scale; 1.0 = touching the lens)
#    With the 65%-saturation bbox heuristic these map roughly to:
#
#      DANGER      > 0.80  →  ~<1 m   stop guidance          — stop now
#      ALERT       > 0.58  →  ~1.5 m  urgent guidance        — very close
#      WARNING     > 0.33  →  ~2.5 m  direction guidance     — approaching
#      APPROACHING > 0.13  →  ~4 m    awareness guidance     — detected ahead
#      SAFE        ≤ 0.13  →  >5 m    silent                 — all clear
#
# Final proximity thresholds must be calibrated after the 3D enclosure, camera
# angle, and ultrasonic sensor mounting are fixed.
# ──────────────────────────────────────────────────────────────────────────────
THRESHOLD_DANGER      = 0.80
THRESHOLD_ALERT       = 0.58
THRESHOLD_WARNING     = 0.33
THRESHOLD_APPROACHING = 0.13

# ── Smoothing ─────────────────────────────────────────────────────────────────
# EMA alpha: lower = smoother / slower reaction, fewer false danger spikes
PROXIMITY_EMA_ALPHA = 0.25

# ── Guidance / voice ──────────────────────────────────────────────────────────
# Offline text-to-speech replaces the old buzzer loop. If TTS is unavailable,
# the app keeps detecting and simply prints guidance in the console.
ENABLE_VOICE = True

# NavigationManager smooths frame-by-frame AI guidance before TTS.
NAV_STABLE_FRAMES = 2
NAV_COMMAND_LOCK_SECONDS = 1.0
NAV_REPEAT_SECONDS = 3.0

# Ignore non-critical CANE/GPS/WiFi status chatter immediately after ready.
STARTUP_STATUS_SUPPRESS_SECONDS = 2.0

VOICE_COOLDOWN_SECONDS = 2.5
EMERGENCY_COOLDOWN_SECONDS = 1.0
TTS_RATE = 175
TTS_BACKEND = "auto"             # auto, windows_sapi, espeak, say, pyttsx3
TTS_TIMEOUT_SECONDS = 5.0        # allows backend startup while keeping worker safe

# ── ESP32 UART bridge ─────────────────────────────────────────────────────────
# Raspberry Pi hardware-specific code starts at uart_bridge.py. Keep UART off
# for PC testing unless an ESP32 is connected.
ENABLE_UART = False
UART_PORT = "/dev/ttyAMA0"       # Raspberry Pi UART; use "COM3" etc. on Windows
UART_BAUDRATE = 115200
UART_SEND_AI_DECISIONS = True
UART_SEND_INTERVAL_SECONDS = 1.0

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_JSON = "detection_output.json"
OUTPUT_LOG  = "detection_log.jsonl"

# JSON snapshot is optional now; enable to preserve the old detection_output.json
# every-frame behavior for PC tests or integrations that depend on it.
ENABLE_DETECTION_OUTPUT_JSON = False

# Debug logging writes JSONL only on risk/command changes by default. Set
# LOG_ON_RISK_CHANGE_ONLY=False to also log a periodic heartbeat.
ENABLE_DEBUG_LOGGING = False
LOG_ON_RISK_CHANGE_ONLY = True
DEBUG_LOG_INTERVAL_SECONDS = 5.0

# ── Display ───────────────────────────────────────────────────────────────────
SHOW_WINDOW = True       # set False on headless Pi
