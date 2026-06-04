from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import config


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
    "dog",
    "cat",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
    "wall/surface",
}


# Higher number means higher speech priority.
PRIORITY = {
    "SAFE": 0,
    "WIFI_CONNECTED": 5,
    "GPS_WEAK": 15,
    "WIFI_LOST": 15,
    "BATTERY_LOW": 25,
    "OBJECT_AHEAD": 30,
    "PERSON_AHEAD": 30,
    "WALL_AHEAD": 30,
    "CROWDED_AREA": 30,
    "MOVE_LEFT": 45,
    "MOVE_RIGHT": 45,
    "STOP": 60,
    "CENTER_DANGER": 60,
    "HEAD_OBSTACLE": 75,
    "SOS_SENT": 90,
    "FALL_DETECTED": 100,
}

EMERGENCY_CODES = {"FALL_DETECTED", "SOS_SENT"}


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
            if obstacle.get("label") in MOBILITY_OBSTACLE_LABELS
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

        center_danger = self._closest(
            obstacles,
            zone="center",
            min_proximity=self.center_danger_proximity,
        )
        if center_danger is not None:
            return self._decision(
                "STOP",
                "Stop",
                risk,
                obstacle=center_danger,
                reason="CENTER_DANGER",
            )

        close_people = [
            obstacle
            for obstacle in obstacles
            if obstacle.get("label") == "person"
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

        center_object = self._closest(
            obstacles,
            zone="center",
            min_proximity=self.object_ahead_proximity,
        )
        if center_object is not None:
            return self._decision(
                "OBJECT_AHEAD",
                "Object ahead",
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
        zone: str | None = None,
        zones: set[str] | None = None,
        min_proximity: float = 0.0,
    ) -> dict[str, Any] | None:
        matches = []
        for obstacle in obstacles:
            if label is not None and obstacle.get("label") != label:
                continue
            if zone is not None and obstacle.get("zone") != zone:
                continue
            if zones is not None and obstacle.get("zone") not in zones:
                continue
            if obstacle.get("proximity", 0.0) < min_proximity:
                continue
            matches.append(obstacle)
        return max(matches, key=lambda item: item.get("proximity", 0.0), default=None)

    def _find_head_obstacle(
        self,
        obstacles: list[dict[str, Any]],
        frame_h: int,
    ) -> dict[str, Any] | None:
        candidates = []
        for obstacle in obstacles:
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
