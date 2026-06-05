# Li-Stick Vision

Li-Stick Vision is a Raspberry Pi 5 MVP guidance module for visually impaired
navigation. It still runs on a PC for camera testing, but the main loop is now
structured for the final hardware split:

```
Camera -> Detector -> Decision Engine -> Voice Manager -> optional ESP32 UART
```

## What It Does

- Detects useful mobility obstacles with YOLOv8.
- Keeps optical-flow wall/surface detection for plain walls and large surfaces.
- Converts detections into short navigation commands instead of speaking every
  object.
- Uses offline text-to-speech instead of the old buzzer loop.
- Optionally exchanges status with an ESP32 over UART.
- Keeps JSON snapshot output available, but disabled by default.

## Guidance Commands

The decision engine uses left, center, and right zones:

| Situation | Spoken guidance |
|---|---|
| Obstacle left | Move slightly right |
| Obstacle right | Move slightly left |
| Dangerous center obstacle | Stop |
| Person center | Person ahead |
| Wall/surface center | Wall ahead |
| Upper-center object | Head obstacle |
| Multiple close people | Crowded area |
| Safe path | Silence |

Priority order:

1. `FALL_DETECTED`
2. `SOS_SENT`
3. `HEAD_SENSOR_ALERT`
4. `BATTERY_LOW`
5. Runtime mode changes
6. `HEAD_OBSTACLE`
7. `STOP` / `CENTER_DANGER`
8. `MOVE_LEFT` / `MOVE_RIGHT`
9. `OBJECT_AHEAD`
10. `SAFE` / silence

ESP32 events such as `GPS_WEAK`, `GPS_AVAILABLE`, `WIFI_LOST`, and
`WIFI_CONNECTED` are supported as status messages.

## Runtime Modes

ESP32 mode events control Raspberry Pi voice behavior at runtime:

| ESP32 event | Pi speech | Behavior |
|---|---|---|
| `AI_PAUSE_ON` | AI guidance paused | Stops AI camera guidance speech |
| `AI_PAUSE_OFF` | AI guidance resumed | Resumes AI camera guidance speech |
| `SILENT_MODE_ON` | Silent mode on | Mutes normal AI navigation speech |
| `SILENT_MODE_OFF` | Silent mode off | Allows AI speech if AI pause/full pause are off |
| `FULL_PAUSE_ON` | Cane paused | Mutes normal AI navigation speech |
| `FULL_PAUSE_OFF` | Cane resumed | Allows AI speech if AI pause/silent mode are off |

These safety ESP32 messages still speak during AI pause, silent mode, or full
pause: `FALL_DETECTED`, `SOS_SENT`, `HEAD_SENSOR_ALERT`, and `BATTERY_LOW`.

You can test runtime state handling without hardware:

```bash
python test_runtime_events.py
```

## Hardware Responsibilities

Future Raspberry Pi responsibilities:

- AI vision
- Text-to-speech
- Camera processing
- Decision engine
- UART communication

Future ESP32 responsibilities:

- SOS button
- Fall detection
- Ultrasonic sensors
- Vibration motors
- GPS acquisition
- Heartbeat and connectivity

Both sides are designed to keep working independently if the other module is
temporarily unavailable.

## Installation

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

On first run, YOLOv8 may download `yolov8n.pt` if it is not already present.

## Running On PC

```bash
python main.py
```

A camera window opens for testing. Press `q` to quit.

If offline TTS is unavailable on the PC, detection and console guidance continue
without crashing.

## Running On Raspberry Pi 5

Recommended settings in `config.py`:

```python
SHOW_WINDOW = False
CAMERA_BACKEND = "auto"
ENABLE_VOICE = True
ENABLE_UART = True
UART_PORT = "/dev/serial0"
```

For Raspberry Pi Camera Module v2 IMX219, `CAMERA_BACKEND = "auto"` will try
Picamera2 first on Raspberry Pi and fall back to OpenCV if Picamera2 is not
available. Use `CAMERA_BACKEND = "opencv"` for laptop/USB webcam testing.
Picamera2 is provided by Raspberry Pi OS packages on the Pi; it is intentionally
not required for Windows laptop testing.

Raspberry Pi hardware-specific camera code starts in `camera_source.py`, and
UART code starts in `uart_bridge.py`. Keep `ENABLE_UART = False` for normal PC
testing unless an ESP32 is connected.

## Configuration

Key settings live in `config.py`:

| Setting | Default | Description |
|---|---:|---|
| `CAMERA_ID` | `0` | Camera index |
| `CAMERA_BACKEND` | `"auto"` | `auto`, `opencv`, or `picamera2` |
| `PICAMERA2_FORMAT` | `"BGR888"` | Picamera2 frame format for detector compatibility |
| `YOLO_CONFIDENCE` | `0.50` | Minimum YOLO confidence |
| `SHOW_WINDOW` | `True` | Disable for headless Pi |
| `ENABLE_VOICE` | `True` | Offline TTS guidance |
| `ENABLE_UART` | `False` | Optional ESP32 serial bridge |
| `ENABLE_DEBUG_LOGGING` | `False` | JSONL debug logging |
| `LOG_ON_RISK_CHANGE_ONLY` | `True` | Log only risk/command changes |
| `VOICE_COOLDOWN_SECONDS` | `2.5` | Suppress repeated AI phrases |
| `EMERGENCY_COOLDOWN_SECONDS` | `1.0` | Suppress repeated emergency phrases |
| `TTS_BACKEND` | `"auto"` | Offline TTS backend selection |
| `TTS_TIMEOUT_SECONDS` | `5.0` | Prevent one TTS call from blocking repeats |
| `ENABLE_DETECTION_OUTPUT_JSON` | `False` | Preserve old detection_output.json behavior |

## Detected Classes

YOLO is limited to mobility-relevant obstacles:

`person`, `chair`, `bench`, `couch`, `dining table`, `bed`, `potted plant`,
`backpack`, `suitcase`, `umbrella`, `dog`, `cat`, `bicycle`, `motorcycle`,
`car`, `bus`, `truck`

Ignored classes include `book`, `bottle`, `laptop`, `tv`, `vase`, `toilet`, and
`refrigerator`. Wall/surface detection remains separate and does not use a YOLO
class label.

## Project Structure

```
Li-Stick-Vision/
├── main.py              # Clean orchestration loop
├── detector.py          # YOLOv8 + wall/surface detection
├── decision_engine.py   # Converts detections into guidance commands
├── runtime_state.py     # Runtime AI pause / silent / full pause state
├── voice_manager.py     # Offline non-blocking TTS with cooldowns
├── camera_source.py     # OpenCV and Picamera2 camera backends
├── uart_bridge.py       # Optional ESP32 UART bridge
├── config.py            # Settings and feature flags
├── test_runtime_events.py
└── requirements.txt
```
