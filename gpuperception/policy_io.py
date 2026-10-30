"""The policy's input transform, shared by training and evaluation."""

from __future__ import annotations


def course_length_m(spec) -> float:
    """Straight-line start-to-goal distance, the normaliser for goal distance."""
    return max(float(sum((g - s) ** 2 for g, s in zip(spec.goal_m, spec.start_m)) ** 0.5), 1.0)


def normalise_depth(depth, spec):
    """Depth in metres -> [0, 1] over the sensor's range."""
    s = spec.sensor
    return ((depth - s.near_m) / (s.far_m - s.near_m)).clamp(0.0, 1.0)


def policy_state(goal_rel_b, vel_b, spec, torch):
    """The 7-D state: [unit goal direction (3), goal distance / course length (1),
    velocity / max speed (3)], goal and velocity in the body frame.

    The direction is a unit vector so it cannot saturate far from the goal; the
    distance is clamped to [0, 2].
    """
    gd = goal_rel_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    goal_dir = goal_rel_b / gd
    goal_dist = (gd / course_length_m(spec)).clamp(0.0, 2.0)
    vel = vel_b / max(float(spec.max_speed_mps), 1e-3)
    return torch.cat([goal_dir, goal_dist, vel], dim=-1)
