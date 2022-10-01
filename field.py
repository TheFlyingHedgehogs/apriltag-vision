from dataclasses import dataclass


@dataclass
class FoundTarget:
    x: float
    y: float
    z: float
    rot: float
    name: str


@dataclass
class Target:
    x: float
    y: float
    z: float
    rot: float
    name: str

    def compute_pose(self, found: FoundTarget):
        angle = self.rot + found.rot


@dataclass
class Field:
    targets: list[Target]
