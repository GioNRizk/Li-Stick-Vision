# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_ID    = 0        # 0 = first USB/Pi camera
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ── Model ─────────────────────────────────────────────────────────────────────
YOLO_MODEL      = "yolov8n.pt"   # nano — fastest on Pi 4/5
YOLO_CONFIDENCE = 0.50           # ignore detections below this confidence
YOLO_CLASSES    = [
    0,   # person
    13,  # bench
    15,  # cat
    16,  # dog
    24,  # backpack
    25,  # umbrella
    26,  # handbag
    28,  # suitcase
    39,  # bottle
    56,  # chair
    57,  # couch
    58,  # potted plant
    59,  # bed
    60,  # dining table
    61,  # toilet
    62,  # tv / monitor (on a stand)
    63,  # laptop
    72,  # refrigerator
    73,  # book
    75,  # vase
]

# ── Depth model ───────────────────────────────────────────────────────────────
USE_DEPTH_MODEL  = False          # True once MiDaS weights are fully downloaded
DEPTH_SKIP_FRAMES = 2

# ── Proximity zones (0.0–1.0 scale; 1.0 = touching the lens)
#    With the 65%-saturation bbox heuristic these map roughly to:
#
#      DANGER      > 0.80  →  ~<1 m   rapid continuous beep  — STOP NOW
#      ALERT       > 0.58  →  ~1.5 m  fast beep              — very close
#      WARNING     > 0.33  →  ~2.5 m  slow beep              — approaching
#      APPROACHING > 0.13  →  ~4 m    soft ping              — detected ahead
#      SAFE        ≤ 0.13  →  >5 m    silent                 — all clear
# ──────────────────────────────────────────────────────────────────────────────
THRESHOLD_DANGER      = 0.80
THRESHOLD_ALERT       = 0.58
THRESHOLD_WARNING     = 0.33
THRESHOLD_APPROACHING = 0.13

# ── Smoothing ─────────────────────────────────────────────────────────────────
# EMA alpha: lower = smoother / slower reaction, fewer false danger spikes
PROXIMITY_EMA_ALPHA = 0.25

# ── Buzzer ────────────────────────────────────────────────────────────────────
BUZZER_PIN = 18          # BCM GPIO pin (Raspberry Pi only)

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_JSON = "detection_output.json"
OUTPUT_LOG  = "detection_log.jsonl"

# ── Display ───────────────────────────────────────────────────────────────────
SHOW_WINDOW = True       # set False on headless Pi
