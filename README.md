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
- Speaks `AI guidance ready` once after camera, detector, voice, and optional
  UART initialization succeed.
- Speaks `AI camera unavailable` or `AI guidance unavailable` for fatal startup
  camera/model failures when voice is available.

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
| Vehicle close or centered | Vehicle nearby |
| Vehicle centered and very close | Stop |
| Stairs label from detector | Stairs ahead |
| Pole/post label from detector | Pole ahead |
| Cat/dog below danger threshold | Obstacle ahead |
| Multiple close people | Crowded area |
| Safe path | Silence |

Priority order:

1. `FALL_DETECTED`
2. SOS request, delivery, failure, cancellation, and hold events
3. `HEAD_SENSOR_ALERT`
4. `BATTERY_LOW` / `AI_UNAVAILABLE`
5. Runtime mode changes
6. `HEAD_OBSTACLE`
7. `STOP` / `CENTER_DANGER`
8. `STAIRS_AHEAD`
9. `VEHICLE_NEARBY`
10. `MOVE_LEFT` / `MOVE_RIGHT`
11. `PERSON_AHEAD` / `WALL_AHEAD` / `POLE_AHEAD`
12. `OBJECT_AHEAD`
13. `INFO` / `AI_READY` / `SAFE` silence

SOS UART speech mappings:

| ESP32 event | Pi speech |
|---|---|
| `SOS_HOLD_STARTED` | Hold for emergency |
| `SOS_CANCELLED` | Emergency cancelled |
| `SOS_REQUESTED` | Sending emergency alert |
| `SOS_DELIVERED_WITH_LOCATION` | Emergency alert sent with location |
| `SOS_DELIVERED_WITHOUT_LOCATION` | Emergency alert sent. Location unavailable |
| `SOS_FAILED_NO_CONNECTION` | No connection. Retrying emergency alert |
| `SOS_DELIVERY_FAILED` | Emergency delivery failed. Retrying |

`SOS_SENT` remains a temporary compatibility alias for older firmware. GPS and
Wi-Fi state changes have no speech mappings and are ignored safely.

Vehicle context uses existing YOLOv8 COCO classes: `bicycle`, `motorcycle`,
`car`, `bus`, and `truck`. Nearby vehicles are announced as `Vehicle nearby`;
centered vehicles at the danger threshold announce `Stop`.

Stairs and pole support is MVP-ready but detector-limited. Standard YOLOv8 COCO
does not reliably provide `stairs`, `staircase`, `pole`, `traffic light pole`,
`sign pole`, or `post` labels. The decision engine handles those labels if a
future custom trained model, depth/segmentation model, or dedicated detector
emits them, but it does not fake stairs or poles from unrelated objects.

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
pause: `FALL_DETECTED`, all SOS events, `HEAD_SENSOR_ALERT`, and `BATTERY_LOW`.
`HEAD_SENSOR_ALERT` speaks `Head obstacle`, can override normal AI guidance, and
uses the emergency speech cooldown so repeated sensor events do not spam speech.

SOS request events are duplicate-suppressed for 3 seconds. SOS delivery and
failure events are duplicate-suppressed independently for 10 seconds, so a
successful delivery is still announced after an earlier failure.

`RuntimeState` also keeps lightweight sensor-fusion readiness fields:
`last_esp32_event`, `last_safety_event_time`, and
`last_head_sensor_alert_time`. These are intentionally simple placeholders for
future fusion between ESP32 safety sensors and Pi camera guidance.

Left/right ultrasonic guidance remains vibration-only on the ESP32. Spoken
left/right guidance still comes from the AI camera decisions: `Move slightly
left` and `Move slightly right`.

You can test runtime state handling without hardware:

```bash
python test_runtime_events.py
```

The simulation covers all SOS UART acceptance sequences, duplicate suppression,
emergency speech priority, startup status, vehicle/stairs/pole placeholder
labels, `HEAD_SENSOR_ALERT`, and SOS pass-through during AI pause, silent mode,
and full pause.

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

Final proximity thresholds should be calibrated after the 3D enclosure, camera
angle, and final sensor mounting are fixed. The current values are kept broad
for breadboard/prototype testing.

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

Vehicle classes receive special MVP context. Cat/dog detections are treated as
lower-priority obstacles unless they are centered and very close.

Placeholder decision support exists for future labels: `stairs`, `staircase`,
`pole`, `traffic light pole`, `sign pole`, and `post`. The included YOLOv8 COCO
model may not emit these labels without custom training or another detector.

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
