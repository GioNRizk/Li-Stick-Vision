from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import config


VEHICLE_LABELS = {
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
}

PET_LABELS = {
    "dog",
    "cat",
}

# YOLOv8 COCO does not include reliable stairs classes. Keep these labels wired
# for a future custom detector, depth/segmentation model, or class-capable model.
STAIRS_LABELS = {
    "stairs",
    "staircase",
}

# YOLOv8 COCO also does not provide a dependable pole/post class. These labels
# are placeholders for future custom model support and external detectors.
POLE_LABELS = {
    "pole",
    "traffic light pole",
    "sign pole",
    "post",
}

MOBILITY_OBSTACLE_LABELS = {
    "person",
    "chair",
    "bench",
    "couch",
    "dining table",
    "bed",
    "potted plant",
    "backpack",
    "suitcase",
    "umbrella",
    "wall/surface",
} | VEHICLE_LABELS | PET_LABELS | STAIRS_LABELS | POLE_LABELS


# Higher number means higher speech priority.
PRIORITY = {
    "SAFE": 0,

    # Info/status messages
    "INFO": 10,
    "AI_READY": 10,
    "WIFI_CONNECTED": 10,
    "GPS_AVAILABLE": 10,
    "CANE_ON": 10,

    # Warnings
    "GPS_WEAK": 25,
    "WIFI_LOST": 25,

    # AI navigation
    "OBJECT_AHEAD": 30,
    "PERSON_AHEAD": 35,
    "WALL_AHEAD": 35,
    "POLE_AHEAD": 35,
    "CROWDED_AREA": 30,
    "MOVE_LEFT": 45,
    "MOVE_RIGHT": 45,
    "VEHICLE_NEARBY": 55,
    "STAIRS_AHEAD": 58,
    "STOP": 60,
    "CENTER_DANGER": 60,
    "HEAD_OBSTACLE": 75,

    # Runtime controls should interrupt normal AI guidance.
    "MODE_CHANGE": 80,

    # Safety/status messages that must cut through pause/silent modes
    "BATTERY_LOW": 85,
    "AI_UNAVAILABLE": 85,
    "HEAD_SENSOR_ALERT": 88,

    # Emergencies
    "SOS_SENT": 90,
    "FALL_DETECTED": 100,
}

EMERGENCY_CODES = {
    "FALL_DETECTED",
    "SOS_SENT",
    "HEAD_SENSOR_ALERT",
    "BATTERY_LOW",
}


@dataclass(slots=True)
class GuidanceDecision:
    code: str
    message: str | None
    priority: int
    risk_level: str
    source: str = "ai"
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def is_emergency(self) -> bool:
        return self.code in EMERGENCY_CODES

    @property
    def should_speak(self) -> bool:
        return bool(self.message) and self.code != "SAFE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "priority": self.priority,
            "risk_level": self.risk_level,
            "details": self.details,
        }


class DecisionEngine:
    """
    Converts raw detector output into short mobility guidance.

    The detector can return many objects per frame. This engine deliberately
    chooses one useful command instead of speaking every object label.
    """

    def __init__(self):
        self.center_danger_proximity = config.THRESHOLD_ALERT
        self.pet_stop_proximity = config.THRESHOLD_DANGER
        self.vehicle_stop_proximity = config.THRESHOLD_DANGER
        self.vehicle_nearby_proximity = config.THRESHOLD_WARNING
        self.side_obstacle_proximity = config.THRESHOLD_WARNING
        self.object_ahead_proximity = config.THRESHOLD_APPROACHING
        self.crowded_people_proximity = config.THRESHOLD_WARNING
        self.head_obstacle_proximity = config.THRESHOLD_APPROACHING
        self.head_obstacle_max_y_ratio = 0.45

    def decide(self, detector_result: dict[str, Any]) -> GuidanceDecision:
        risk = detector_result.get("risk_level", "safe")
        obstacles = [
            obstacle
            for obstacle in detector_result.get("obstacles", [])
            if self._label(obstacle) in MOBILITY_OBSTACLE_LABELS
        ]

        if not obstacles:
            return self._decision("SAFE", None, risk)

        frame_h = int(detector_result.get("frame_height") or config.FRAME_HEIGHT)

        head_obstacle = self._find_head_obstacle(obstacles, frame_h)
        if head_obstacle is not None:
            return self._decision(
                "HEAD_OBSTACLE",
                "Head obstacle",
                risk,
                obstacle=head_obstacle,
            )

        vehicle_stop = self._closest(
            obstacles,
            labels=VEHICLE_LABELS,
            zone="center",
            min_proximity=self.vehicle_stop_proximity,
        )
        if vehicle_stop is not None:
            return self._decision(
                "STOP",
                "Stop",
                risk,
                obstacle=vehicle_stop,
                reason="VEHICLE_CENTER_DANGER",
            )

        pet_stop = self._closest(
            obstacles,
            labels=PET_LABELS,
            zone="center",
            min_proximity=self.pet_stop_proximity,
        )
        if pet_stop is not None:
            return self._decision(
                "STOP",
                "Stop",
                risk,
                obstacle=pet_stop,
                reason="PET_CENTER_DANGER",
            )

        center_danger = self._closest(
            obstacles,
            zone="center",
            min_proximity=self.center_danger_proximity,
            exclude_labels=VEHICLE_LABELS | PET_LABELS | STAIRS_LABELS | POLE_LABELS,
        )
        if center_danger is not None:
            return self._decision(
                "STOP",
                "Stop",
                risk,
                obstacle=center_danger,
                reason="CENTER_DANGER",
            )

        stairs = self._closest(
            obstacles,
            labels=STAIRS_LABELS,
            min_proximity=self.object_ahead_proximity,
        )
        if stairs is not None:
            return self._decision(
                "STAIRS_AHEAD",
                "Stairs ahead",
                risk,
                obstacle=stairs,
            )

        vehicle = self._find_vehicle_nearby(obstacles)
        if vehicle is not None:
            return self._decision(
                "VEHICLE_NEARBY",
                "Vehicle nearby",
                risk,
                obstacle=vehicle,
            )

        close_people = [
            obstacle
            for obstacle in obstacles
            if self._label(obstacle) == "person"
            and obstacle.get("proximity", 0.0) >= self.crowded_people_proximity
        ]
        if len(close_people) >= 2:
            return self._decision(
                "CROWDED_AREA",
                "Crowded area",
                risk,
                people_count=len(close_people),
            )

        side_obstacle = self._closest(
            obstacles,
            zones={"left", "right"},
            min_proximity=self.side_obstacle_proximity,
            exclude_labels=PET_LABELS,
        )
        if side_obstacle is not None:
            if side_obstacle.get("zone") == "left":
                return self._decision(
                    "MOVE_RIGHT",
                    "Move slightly right",
                    risk,
                    obstacle=side_obstacle,
                )
            return self._decision(
                "MOVE_LEFT",
                "Move slightly left",
                risk,
                obstacle=side_obstacle,
            )

        center_person = self._closest(
            obstacles,
            label="person",
            zone="center",
            min_proximity=self.object_ahead_proximity,
        )
        if center_person is not None:
            return self._decision(
                "PERSON_AHEAD",
                "Person ahead",
                risk,
                obstacle=center_person,
            )

        wall = self._closest(
            obstacles,
            label="wall/surface",
            zone="center",
            min_proximity=self.object_ahead_proximity,
        )
        if wall is not None:
            return self._decision(
                "WALL_AHEAD",
                "Wall ahead",
                risk,
                obstacle=wall,
            )

        pole = self._closest(
            obstacles,
            labels=POLE_LABELS,
            zone="center",
            min_proximity=self.object_ahead_proximity,
        )
        if pole is not None:
            return self._decision(
                "POLE_AHEAD",
                "Pole ahead",
                risk,
                obstacle=pole,
            )

        center_object = self._closest(
            obstacles,
            zone="center",
            min_proximity=self.object_ahead_proximity,
        )
        if center_object is not None:
            return self._decision(
                "OBJECT_AHEAD",
                "Obstacle ahead",
                risk,
                obstacle=center_object,
            )

        return self._decision("SAFE", None, risk)

    def _decision(
        self,
        code: str,
        message: str | None,
        risk_level: str,
        obstacle: dict[str, Any] | None = None,
        **details: Any,
    ) -> GuidanceDecision:
        if obstacle is not None:
            details["obstacle"] = {
                "label": obstacle.get("label"),
                "zone": obstacle.get("zone"),
                "proximity": obstacle.get("proximity"),
            }
        return GuidanceDecision(
            code=code,
            message=message,
            priority=PRIORITY.get(code, PRIORITY["OBJECT_AHEAD"]),
            risk_level=risk_level,
            details=details,
        )

    def _closest(
        self,
        obstacles: list[dict[str, Any]],
        label: str | None = None,
        labels: set[str] | None = None,
        exclude_labels: set[str] | None = None,
        zone: str | None = None,
        zones: set[str] | None = None,
        min_proximity: float = 0.0,
    ) -> dict[str, Any] | None:
        matches = []
        for obstacle in obstacles:
            obstacle_label = self._label(obstacle)
            if label is not None and obstacle_label != label:
                continue
            if labels is not None and obstacle_label not in labels:
                continue
            if exclude_labels is not None and obstacle_label in exclude_labels:
                continue
            if zone is not None and obstacle.get("zone") != zone:
                continue
            if zones is not None and obstacle.get("zone") not in zones:
                continue
            if obstacle.get("proximity", 0.0) < min_proximity:
                continue
            matches.append(obstacle)
        return max(matches, key=lambda item: item.get("proximity", 0.0), default=None)

    def _find_vehicle_nearby(
        self,
        obstacles: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        candidates = []
        for obstacle in obstacles:
            if self._label(obstacle) not in VEHICLE_LABELS:
                continue
            proximity = obstacle.get("proximity", 0.0)
            is_close = proximity >= self.vehicle_nearby_proximity
            is_centered = (
                obstacle.get("zone") == "center"
                and proximity >= self.object_ahead_proximity
            )
            if is_close or is_centered:
                candidates.append(obstacle)
        return max(candidates, key=lambda item: item.get("proximity", 0.0), default=None)

    def _find_head_obstacle(
        self,
        obstacles: list[dict[str, Any]],
        frame_h: int,
    ) -> dict[str, Any] | None:
        candidates = []
        for obstacle in obstacles:
            if self._label(obstacle) in (
                PET_LABELS | VEHICLE_LABELS | STAIRS_LABELS | POLE_LABELS
            ):
                continue
            if obstacle.get("zone") != "center":
                continue
            if obstacle.get("proximity", 0.0) < self.head_obstacle_proximity:
                continue
            bbox = obstacle.get("bbox") or []
            if len(bbox) != 4:
                continue
            _, y1, _, y2 = bbox
            center_y_ratio = ((y1 + y2) / 2) / max(frame_h, 1)
            if center_y_ratio <= self.head_obstacle_max_y_ratio:
                candidates.append(obstacle)
        return max(candidates, key=lambda item: item.get("proximity", 0.0), default=None)

    def _label(self, obstacle: dict[str, Any]) -> str:
        return str(obstacle.get("label", "")).strip().lower()
