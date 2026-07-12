# â”€â”€ Camera â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CAMERA_ID    = 0        # 0 = first USB/Pi camera
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
CAMERA_BACKEND = "auto"          # auto, opencv, picamera2
PICAMERA2_FORMAT = "BGR888"      # BGR888 feeds detector/display without conversion

# â”€â”€ Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
YOLO_MODEL      = "yolov8n.pt"   # nano â€” fastest on Pi 4/5
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

# â”€â”€ Depth model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
USE_DEPTH_MODEL  = False          # True once MiDaS weights are fully downloaded
DEPTH_SKIP_FRAMES = 2

# â”€â”€ Proximity zones (0.0â€“1.0 scale; 1.0 = touching the lens)
#    With the 65%-saturation bbox heuristic these map roughly to:
#
#      DANGER      > 0.80  â†’  ~<1 m   stop guidance          â€” stop now
#      ALERT       > 0.58  â†’  ~1.5 m  urgent guidance        â€” very close
#      WARNING     > 0.33  â†’  ~2.5 m  direction guidance     â€” approaching
#      APPROACHING > 0.13  â†’  ~4 m    awareness guidance     â€” detected ahead
#      SAFE        â‰¤ 0.13  â†’  >5 m    silent                 â€” all clear
#
# Final proximity thresholds must be calibrated after the 3D enclosure, camera
# angle, and ultrasonic sensor mounting are fixed.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
THRESHOLD_DANGER      = 0.80
THRESHOLD_ALERT       = 0.58
THRESHOLD_WARNING     = 0.33
THRESHOLD_APPROACHING = 0.13

# â”€â”€ Smoothing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# EMA alpha: lower = smoother / slower reaction, fewer false danger spikes
PROXIMITY_EMA_ALPHA = 0.25

# â”€â”€ Guidance / voice â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

TTS_RATE = 135
TTS_VOICE = "en-us"
TTS_AMPLITUDE = 200
TTS_BACKEND = "auto"
TTS_TIMEOUT_SECONDS = 5.0

# â”€â”€ ESP32 UART bridge â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Raspberry Pi hardware-specific code starts at uart_bridge.py. Keep UART off
# for PC testing unless an ESP32 is connected.
ENABLE_UART = True
UART_PORT = "/dev/ttyAMA0"       # Raspberry Pi UART; use "COM3" etc. on Windows
UART_BAUDRATE = 115200
UART_SEND_AI_DECISIONS = True
UART_SEND_INTERVAL_SECONDS = 1.0

# â”€â”€ Output â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

# â”€â”€ Display â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SHOW_WINDOW = True       # set False on headless Pi
