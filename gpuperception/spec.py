"""Course and vehicle specification, and the seeded obstacle sampler.

A `ScenarioSpec` fully determines a course: `sample_obstacles(spec)` places the same
obstacles for the same seed on any machine. Evaluation courses are seeds 9000-9002 and
9100-9102; training draws fresh layouts per episode from seeds >= 1e6 (see env.py).

Field set and order are kept exactly as in the code that produced the archived results,
because `spec_hash` / `course_hash` (recorded in every evaluation summary) are hashes of
the serialised spec. Fields marked "unused" have no effect in this code base.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SensorSpec:
    """Forward depth camera (RealSense D435-like field of view)."""

    width: int = 64
    height: int = 64
    hfov_deg: float = 87.0
    vfov_deg: float = 58.0
    near_m: float = 0.1
    far_m: float = 10.0
    rate_hz: float = 20.0


@dataclass(frozen=True)
class DegradationSpec:
    """Unused (sensor-noise hooks of an earlier version); kept for hash compatibility."""

    depth_noise_m: float = 0.0
    dropout_prob: float = 0.0
    latency_s: float = 0.0
    frame_drop_prob: float = 0.0


@dataclass(frozen=True)
class ObstacleFieldSpec:
    """Obstacle field: every third obstacle is a floor sphere, the rest are pillars."""

    n_obstacles: int = 20
    radius_min_m: float = 0.2
    radius_max_m: float = 0.6
    height_m: float = 4.0
    area_x_m: tuple[float, float] = (4.0, 36.0)
    area_y_m: tuple[float, float] = (-10.0, 10.0)
    min_separation_m: float = 1.2
    # Textured MDL materials on obstacles and ground. Rendered depth does not depend on
    # them; kept because the archived runs loaded them.
    textured: bool = True
    texture_scale: float = 2.0


@dataclass(frozen=True)
class ScenarioSpec:
    """A complete, reproducible course and vehicle envelope."""

    name: str = "corridor_v1"
    seed: int = 0
    world: str = "procedural"           # unused; only the procedural course exists here
    start_m: tuple[float, float, float] = (0.0, 0.0, 2.0)
    goal_m: tuple[float, float, float] = (40.0, 0.0, 2.0)
    goal_tolerance_m: float = 1.5
    # Pillars span floor to ceiling, and flying above the ceiling counts as a collision,
    # so the field cannot be overflown.
    ceiling_m: float = 4.0
    timeout_s: float = 60.0
    control_dt_s: float = 0.05          # 20 Hz
    max_speed_mps: float = 3.0
    # Vehicle response: first-order velocity tracking with time constant vel_tau_s and
    # an acceleration limit reduced by aerodynamic drag (0.5 * rho * CdA * v^2 / m).
    max_accel_mps2: float = 6.0
    vel_tau_s: float = 0.15
    mass_kg: float = 2.0
    drag_cd_area_m2: float = 0.10
    safety_margin_m: float = 0.5        # the reward's proximity penalty starts inside this
    obstacles: ObstacleFieldSpec = field(default_factory=ObstacleFieldSpec)
    sensor: SensorSpec = field(default_factory=SensorSpec)
    degradation: DegradationSpec = field(default_factory=DegradationSpec)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def spec_hash(self) -> str:
        """Short hash of the whole spec, including its name."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:12]

    @property
    def course_hash(self) -> str:
        """Short hash of the course itself (everything except the name)."""
        d = json.loads(self.to_json())
        d.pop("name", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    @property
    def max_steps(self) -> int:
        return int(round(self.timeout_s / self.control_dt_s))


@dataclass(frozen=True)
class Obstacle:
    """One placed obstacle. `kind` is "sphere" or "box"."""

    kind: str
    x: float
    y: float
    z: float          # centre height
    radius: float     # sphere radius, or half-width for a box
    height: float     # box height; equals 2*radius for a sphere


def sample_obstacles(spec: ScenarioSpec) -> list[Obstacle]:
    """Reproducible obstacle field for a spec.

    The draw order is load-bearing: each attempt consumes two uniforms and the radius
    one more, so reordering silently produces a different course for the same seed.
    """
    import numpy as np

    obs = spec.obstacles
    rng = np.random.default_rng(spec.seed)
    out: list[Obstacle] = []
    placed: list[tuple[float, float, float]] = []
    for i in range(obs.n_obstacles):
        x = y = 0.0
        for _ in range(50):
            x = float(rng.uniform(*obs.area_x_m))
            y = float(rng.uniform(*obs.area_y_m))
            if all((x - px) ** 2 + (y - py) ** 2 >= obs.min_separation_m ** 2
                   for px, py, _ in placed):
                break
        r = float(rng.uniform(obs.radius_min_m, obs.radius_max_m))
        placed.append((x, y, r))
        if i % 3 == 0:
            out.append(Obstacle("sphere", x, y, r, r, 2.0 * r))
        else:
            out.append(Obstacle("box", x, y, obs.height_m / 2.0, r, obs.height_m))
    return out
