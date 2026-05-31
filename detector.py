import cv2
import numpy as np
import torch
from datetime import datetime
from ultralytics import YOLO

import config


class ObstacleDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Detector] Device : {self.device}")

        print("[Detector] Loading YOLO model...")
        self.yolo = YOLO(config.YOLO_MODEL)

        self.midas = None
        if config.USE_DEPTH_MODEL:
            self._load_midas()

        # Exponential moving average for smooth proximity score
        self._ema: float = 0.0
        self._prev_ema: float = 0.0

        self._cached_depth: np.ndarray | None = None
        self._depth_counter: int = 0
        self.frame_id: int = 0

        # Previous grayscale frame — used by optical-flow wall detector
        self._prev_gray: np.ndarray | None = None

    # ── Depth model ───────────────────────────────────────────────────────────

    def _load_midas(self):
        print("[Detector] Loading MiDaS (first run downloads ~82 MB)...")
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.midas.to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.midas_transform = transforms.small_transform
        print("[Detector] MiDaS ready.")

    def _get_depth_map(self, frame: np.ndarray) -> np.ndarray:
        self._depth_counter += 1
        if self._cached_depth is None or self._depth_counter >= config.DEPTH_SKIP_FRAMES:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch = self.midas_transform(img_rgb).to(self.device)
            with torch.no_grad():
                pred = self.midas(batch)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=frame.shape[:2],
                    mode="bicubic", align_corners=False,
                ).squeeze()
            d = pred.cpu().numpy().astype(np.float32)
            lo, hi = d.min(), d.max()
            self._cached_depth = (d - lo) / (hi - lo + 1e-6)
            self._depth_counter = 0
        return self._cached_depth

    # ── Proximity helpers ─────────────────────────────────────────────────────

    def _bbox_proximity(self, x1: int, y1: int, x2: int, y2: int,
                        frame_h: int, frame_w: int) -> float:
        """
        Bbox-area heuristic (0–1 scale).  Saturates at 65 % frame coverage.

        Rough distance table (640×480, normal webcam):
          bbox covers 65 %+  →  1.00  ≈ <1 m   (danger)
          bbox covers 52 %   →  0.80  ≈ ~1 m   (danger threshold)
          bbox covers 38 %   →  0.58  ≈ ~1.5 m (alert threshold)
          bbox covers 21 %   →  0.33  ≈ ~2.5 m (warning threshold)
          bbox covers  8 %   →  0.13  ≈ ~4 m   (approaching threshold)
        """
        area = (x2 - x1) * (y2 - y1)
        return min(area / (frame_h * frame_w * 0.65), 1.0)

    def _detect_blocking_surface(self, frame: np.ndarray) -> dict | None:
        """
        Two-signal surface detector for walls and large unclassified obstacles.

        Signal 1 — optical flow divergence:
          When the user walks toward any surface (plain wall, door, glass),
          the whole scene expands outward from the centre of the frame.
          This expanding (diverging) flow is measured and mapped to proximity.
          Works on featureless painted walls where edge detection finds nothing.

        Signal 2 — edge fill in centre-bottom:
          For textured surfaces close to the camera (furniture, shelves, doors
          with frames) large edge blobs fill the lower centre of the frame.

        The stronger of the two signals is used as the final proximity score.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Signal 1: optical flow divergence ─────────────────────────────
        flow_score = 0.0
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.2, flags=0,
            )
            cy, cx = h // 2, w // 2
            roi_y1, roi_y2 = h // 4, 3 * h // 4
            roi_x1, roi_x2 = w // 4, 3 * w // 4
            roi_flow = flow[roi_y1:roi_y2, roi_x1:roi_x2]

            # Grid of (x, y) offsets from the frame centre for each ROI pixel
            xs = np.arange(roi_x1, roi_x2) - cx
            ys = np.arange(roi_y1, roi_y2) - cy
            xs_grid, ys_grid = np.meshgrid(xs, ys)
            dist = np.sqrt(xs_grid ** 2 + ys_grid ** 2) + 1e-6

            # Radial component of flow: positive = points moving away from centre
            divergence = (
                roi_flow[..., 0] * xs_grid + roi_flow[..., 1] * ys_grid
            ) / dist
            mean_div = float(np.mean(divergence))
            if mean_div > 0:
                flow_score = min(mean_div / 6.0, 1.0)

        self._prev_gray = gray

        # ── Signal 2: edge fill in centre-bottom ──────────────────────────
        roi = frame[h // 2:, w // 4: 3 * w // 4]
        roi_area = roi.shape[0] * roi.shape[1]
        blurred = cv2.GaussianBlur(
            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (11, 11), 0
        )
        edges   = cv2.Canny(blurred, 25, 75)
        kernel  = np.ones((9, 9), np.uint8)
        dilated = cv2.dilate(edges, kernel)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        large_area = sum(
            cv2.contourArea(c) for c in contours
            if cv2.contourArea(c) > roi_area * 0.04
        )
        fill_ratio = min(large_area / roi_area, 1.0)
        edge_score = min(fill_ratio * 1.5, 1.0) if fill_ratio >= 0.35 else 0.0

        # ── Combine: take the stronger signal ─────────────────────────────
        proximity = max(flow_score, edge_score)
        if proximity < config.THRESHOLD_APPROACHING:
            return None

        return {
            "label":      "wall/surface",
            "confidence": round(proximity, 3),
            "proximity":  round(proximity, 3),
            "zone":       "center",
            "bbox":       [w // 4, h // 2, 3 * w // 4, h],
        }

    def _zone(self, x1: int, x2: int, frame_w: int) -> str:
        cx = (x1 + x2) / 2 / frame_w
        if cx < 0.33:
            return "left"
        elif cx < 0.67:
            return "center"
        return "right"

    def _risk_level(self, score: float) -> str:
        if score >= config.THRESHOLD_DANGER:
            return "danger"
        elif score >= config.THRESHOLD_ALERT:
            return "alert"
        elif score >= config.THRESHOLD_WARNING:
            return "warning"
        elif score >= config.THRESHOLD_APPROACHING:
            return "approaching"
        return "safe"

    # ── Main detect call ──────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> tuple[dict, np.ndarray]:
        self.frame_id += 1
        h, w = frame.shape[:2]

        depth_map = self._get_depth_map(frame) if self.midas else None

        yolo_results = self.yolo(
            frame,
            conf=config.YOLO_CONFIDENCE,
            classes=config.YOLO_CLASSES,
            verbose=False,
        )

        obstacles: list[dict] = []
        for result in yolo_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label      = result.names[int(box.cls[0])]
                confidence = round(float(box.conf[0]), 3)

                if depth_map is not None:
                    roi = depth_map[max(y1, 0):y2, max(x1, 0):x2]
                    raw_prox = float(np.mean(roi)) if roi.size > 0 else 0.0
                else:
                    raw_prox = self._bbox_proximity(x1, y1, x2, y2, h, w)

                obstacles.append({
                    "label":      label,
                    "confidence": confidence,
                    "proximity":  round(raw_prox, 3),
                    "zone":       self._zone(x1, x2, w),
                    "bbox":       [x1, y1, x2, y2],
                })

        # ── Wall / large-surface detector (catches what YOLO doesn't label) ─────
        if not self.midas:   # skip if MiDaS already gives per-pixel depth
            surface = self._detect_blocking_surface(frame)
            if surface is not None:
                # Only add if there's no YOLO obstacle already covering this zone
                # with a higher proximity (avoid double-counting)
                best_center = max(
                    (o["proximity"] for o in obstacles if o["zone"] == "center"),
                    default=0.0,
                )
                if surface["proximity"] > best_center:
                    obstacles.append(surface)

        # Sort closest first
        obstacles.sort(key=lambda o: o["proximity"], reverse=True)

        # ── Smooth the worst-case proximity with EMA ──────────────────────────
        raw_max   = obstacles[0]["proximity"] if obstacles else 0.0
        alpha     = config.PROXIMITY_EMA_ALPHA
        self._prev_ema = self._ema
        self._ema = alpha * raw_max + (1 - alpha) * self._ema

        smooth_prox = round(self._ema, 3)
        delta       = round(self._ema - self._prev_ema, 3)

        # Approach direction: >+0.01 = getting closer, <-0.01 = moving away
        if delta > 0.01:
            trend = "approaching"
        elif delta < -0.01:
            trend = "receding"
        else:
            trend = "stable"

        risk = self._risk_level(smooth_prox)

        # Attach smoothed values back to the closest obstacle for the caller
        if obstacles:
            obstacles[0]["smooth_proximity"] = smooth_prox
            obstacles[0]["trend"]            = trend
            obstacles[0]["risk_level"]       = risk

        result_dict = {
            "timestamp":       datetime.now().isoformat(),
            "frame_id":        self.frame_id,
            "risk_level":      risk,
            "smooth_proximity": smooth_prox,
            "trend":           trend if obstacles else "stable",
            "obstacle_count":  len(obstacles),
            "obstacles":       obstacles,
            "depth_model":     self.midas is not None,
        }

        annotated = yolo_results[0].plot() if yolo_results else frame
        return result_dict, annotated
