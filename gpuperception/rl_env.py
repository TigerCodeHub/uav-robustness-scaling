"""RL view of the course: policy encoding, reward and auto-reset.

Reward per step: +1 per metre of goal distance closed, a small alive cost, a jerk
penalty on the applied velocity command, and a proximity penalty inside the safety
margin. Terminal: +20 on arrival, -5 on collision (including ceiling and lateral
boundary) and -5 on timeout, so failing slowly costs as much as crashing.
"""

from __future__ import annotations

from dataclasses import dataclass

from .env import IsaacLabConfig, IsaacLabCourse
from .spec import ScenarioSpec


@dataclass
class RewardCfg:
    progress: float = 1.0        # per metre of goal distance closed
    collision: float = 5.0       # one-off penalty, also terminates
    timeout: float = 5.0         # one-off penalty
    arrival: float = 20.0        # one-off bonus
    jerk: float = 0.002          # per (m/s)^2 of change in the applied command
    proximity: float = 0.5       # weight of the squared margin violation
    alive: float = -0.01         # per-step cost


class CourseRLEnv:
    """Batched RL environment. Finished environments are reset inside `step`."""

    def __init__(self, spec: ScenarioSpec, cfg: IsaacLabConfig | None = None,
                 reward: RewardCfg | None = None):
        import torch

        self.course = IsaacLabCourse(spec, cfg)
        self.spec = spec
        self.rcfg = reward or RewardCfg()
        self.torch = torch
        self.n = self.course.n
        self.device = self.course.cfg.device

        obs = self.course.observation()
        self.prev_goal_dist = obs["goal_rel"].norm(dim=-1)
        self.prev_cmd = torch.zeros((self.n, 3), device=self.device)
        self.ep_return = torch.zeros(self.n, device=self.device)
        self.ep_len = torch.zeros(self.n, dtype=torch.long, device=self.device)

    @property
    def depth_hw(self) -> tuple[int, int]:
        return self.spec.sensor.height, self.spec.sensor.width

    @property
    def state_dim(self) -> int:
        return 7  # goal unit direction (3) + normalised goal distance (1) + velocity (3)

    def encode(self, obs: dict):
        """Normalised depth in [0, 1] and the 7-D state vector."""
        from .policy_io import normalise_depth, policy_state
        depth = normalise_depth(obs["depth"], self.spec)
        state = policy_state(obs["goal_rel"], obs["vel"], self.spec, torch=self.torch)
        return depth, state

    def step(self, command):
        torch = self.torch
        obs, info = self.course.step(command)

        goal_dist = info["goal_dist"]
        progress = self.prev_goal_dist - goal_dist
        cmd3 = info["cmd_applied_b"][:, :3]
        jerk = ((cmd3 - self.prev_cmd) ** 2).sum(dim=-1)

        margin = self.spec.safety_margin_m
        viol = (1.0 - info["clearance"] / max(margin, 1e-6)).clamp(min=0.0)

        r = (self.rcfg.progress * progress
             + self.rcfg.alive
             - self.rcfg.jerk * jerk
             - self.rcfg.proximity * viol ** 2
             + self.rcfg.arrival * info["success"].float()
             - self.rcfg.collision * info["collision"].float()
             - self.rcfg.timeout * info["timeout"].float())

        self.ep_return += r
        self.ep_len += 1
        done = info["done"]

        if bool(done.any()):
            idx = done.nonzero(as_tuple=False).flatten()
            # The camera is not re-rendered here, so a reset env's first observation carries
            # the depth frame rendered before the reset (kept as run; see README).
            self.course.reset_idx(idx)
            self.ep_return[idx] = 0.0
            self.ep_len[idx] = 0
            obs = self.course.observation()
            goal_dist = obs["goal_rel"].norm(dim=-1)
            cmd3 = cmd3.clone()
            cmd3[idx] = 0.0

        self.prev_goal_dist = goal_dist
        self.prev_cmd = cmd3
        return obs, r, done, info

    def reset(self):
        idx = self.torch.arange(self.n, device=self.device)
        self.course.reset_idx(idx)
        obs = self.course.observation()
        self.prev_goal_dist = obs["goal_rel"].norm(dim=-1)
        self.prev_cmd = self.torch.zeros((self.n, 3), device=self.device)
        self.ep_return.zero_()
        self.ep_len.zero_()
        return obs
