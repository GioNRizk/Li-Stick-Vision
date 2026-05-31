# Li-Stick Vision

A real-time obstacle detection system for visually impaired people, designed to run on a **Raspberry Pi 4/5** (or any PC for testing). Uses a camera to detect obstacles and warns the user with a buzzer — just like a car parking sensor.

---

## How it works

- Detects people, chairs, tables, furniture, and other indoor obstacles using **YOLOv8**
- Detects walls and plain surfaces using **optical flow** (even featureless painted walls)
- Gives audio feedback through a buzzer (or PC speaker on Windows) with increasing urgency as obstacles get closer
- 5 alert zones: **SAFE → APPROACHING → WARNING → ALERT → DANGER**

---

## Hardware required

| Part | Details |
|---|---|
| Raspberry Pi 4 or 5 | Any RAM size works; 4 GB recommended |
| Camera | Pi Camera Module or any USB webcam |
| Buzzer | Active buzzer wired to GPIO pin 18 (BCM) |
| Power bank | To make it portable |

> **Testing on Windows/Mac/Linux PC**: works without a buzzer — uses the PC speaker instead. No Raspberry Pi needed to test.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/GioNRizk/Li-Stick-Vision.git
cd Li-Stick-Vision
```

### 2. Install Python dependencies

Requires **Python 3.10 or newer**.

```bash
pip install -r requirements.txt
```

This installs:
- `ultralytics` — YOLOv8 object detection
- `opencv-python` — camera capture and image processing
- `torch` + `torchvision` — deep learning backend
- `numpy` — array math
- `timm` — model utilities

> **On Raspberry Pi**, use `pip3` and make sure you have at least 2 GB free disk space for the model weights.

### 3. First run — automatic model download

On the first run, YOLOv8 will automatically download `yolov8n.pt` (~6 MB). This only happens once.

---

## Running

```bash
python main.py
```

A camera window will open showing live detections. Press **`q`** to quit.

> On Raspberry Pi headless (no monitor): set `SHOW_WINDOW = False` in `config.py` before running.

---

## Configuration (`config.py`)

All settings are in `config.py`. The most useful ones:

| Setting | Default | Description |
|---|---|---|
| `CAMERA_ID` | `0` | Camera index — try `1` if your webcam isn't detected |
| `YOLO_CONFIDENCE` | `0.50` | Minimum detection confidence (0–1). Lower = more sensitive but more false positives |
| `SHOW_WINDOW` | `True` | Set to `False` on headless Raspberry Pi (no monitor) |
| `BUZZER_PIN` | `18` | BCM GPIO pin for the buzzer (Raspberry Pi only) |
| `PROXIMITY_EMA_ALPHA` | `0.25` | Smoothing factor — lower = smoother but slower to react |

### Alert thresholds

| Zone | Threshold | Estimated distance | Buzzer |
|---|---|---|---|
| APPROACHING | > 0.13 | ~4 m | Soft ping every 2.5 s |
| WARNING | > 0.33 | ~2.5 m | Beep every 0.8 s |
| ALERT | > 0.58 | ~1.5 m | Fast beep every 0.35 s |
| DANGER | > 0.80 | < 1 m | Rapid beep every 0.18 s |

---

## Raspberry Pi — buzzer wiring

```
Raspberry Pi GPIO 18 (BCM)  →  Buzzer positive (+)
Raspberry Pi GND            →  Buzzer negative (-)
```

The system auto-detects whether it's running on a Pi (uses GPIO PWM) or a PC (uses the sound card). No code change needed.

---

## Enabling the depth model (more accurate distances)

The system works without it, but enabling **MiDaS** gives better real-world distance estimates, especially for walls.

1. Set `USE_DEPTH_MODEL = True` in `config.py`
2. On first run it downloads ~82 MB of model weights
3. Requires a stable internet connection for the download

---

## Detected obstacle classes

The system detects 20 indoor/pedestrian obstacle types:

`person` · `bench` · `chair` · `couch` · `bed` · `dining table` · `toilet` · `refrigerator` · `potted plant` · `vase` · `tv` · `laptop` · `book` · `bottle` · `backpack` · `suitcase` · `handbag` · `umbrella` · `cat` · `dog`

Walls and plain surfaces are detected separately using optical flow (no class label needed).

---

## Project structure

```
Li-Stick-Vision/
├── main.py          # Main loop: camera → detect → buzzer → display
├── detector.py      # YOLOv8 + wall detector + proximity scoring
├── alerter.py       # Buzzer sound logic (GPIO on Pi, winsound on Windows)
├── config.py        # All settings in one place
└── requirements.txt # Python dependencies
```

---

## Requirements summary

```
Python >= 3.10
pip install -r requirements.txt
```

For Raspberry Pi GPIO buzzer:
```
pip install RPi.GPIO
```
