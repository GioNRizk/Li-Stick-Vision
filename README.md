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
3. `HEAD_OBSTACLE`
4. `STOP` / `CENTER_DANGER`
5. `MOVE_LEFT` / `MOVE_RIGHT`
6. `OBJECT_AHEAD`
7. `SAFE` / silence

ESP32 events such as `BATTERY_LOW`, `GPS_WEAK`, `WIFI_LOST`, and
`WIFI_CONNECTED` are supported as lower-priority status messages.

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
ENABLE_VOICE = True
ENABLE_UART = True
UART_PORT = "/dev/serial0"
```

Raspberry Pi hardware-specific UART code starts in `uart_bridge.py`. Keep
`ENABLE_UART = False` for normal PC testing unless an ESP32 is connected.

## Configuration

Key settings live in `config.py`:

| Setting | Default | Description |
|---|---:|---|
| `CAMERA_ID` | `0` | Camera index |
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
├── voice_manager.py     # Offline non-blocking TTS with cooldowns
├── uart_bridge.py       # Optional ESP32 UART bridge
├── config.py            # Settings and feature flags
└── requirements.txt
```
