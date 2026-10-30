"""The two perturbations studied: command latency and attitude (tilt) coupling.

Both are off by default, so an unset `RealismCfg` is the clean condition.

Command latency
    The vehicle executes the command issued d steps earlier, d ~ N(tau/dt, (sigma/dt)^2)
    drawn per environment and per step, rounded and truncated at zero. With sigma = 0 it
    is a fixed FIFO delay. Until an environment has enough post-reset history it receives
    a zero command.

Attitude coupling
    A multirotor accelerates horizontally by tilting its thrust vector, so the body (and
    a body-fixed camera) tilts by tan(tilt) = a / g. Each step the roll and pitch implied
    by the achieved acceleration, capped at `attitude_max_deg`, are written into the
    vehicle pose, which tilts the rendered depth view and the body-frame observations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RealismCfg:
    cmd_latency_s: float = 0.0          # mean transport delay, seconds
    cmd_latency_jitter_s: float = 0.0   # std dev of the delay, seconds (0 = fixed delay)
    attitude_coupling: bool = False
    attitude_max_deg: float = 20.0

    def any_on(self) -> bool:
        return bool(self.cmd_latency_s > 0 or self.cmd_latency_jitter_s > 0
                    or self.attitude_coupling)

    def tag(self) -> str:
        """Condition identifier used in run ids and result filenames, e.g. "lat200j100_att"."""
        if not self.any_on():
            return "none"
        p = []
        if self.cmd_latency_s > 0 or self.cmd_latency_jitter_s > 0:
            lat = f"lat{int(round(self.cmd_latency_s * 1000))}"
            if self.cmd_latency_jitter_s > 0:
                lat += f"j{int(round(self.cmd_latency_jitter_s * 1000))}"
            p.append(lat)
        if self.attitude_coupling:
            p.append("att")
        return "_".join(p)


class CommandDelay:
    """Per-environment transport delay on the raw policy command.

    Fixed delay: a FIFO of `n_steps`. Jittered delay: a ring buffer read back by a
    per-env, per-step Gaussian number of steps, drawn from its own seeded generator.
    """

    def __init__(self, latency_s: float, dt: float, n: int, device: str, torch,
                 jitter_s: float = 0.0, seed: int = 0):
        self.torch = torch
        self.dt = max(dt, 1e-9)
        self.n_steps = int(round(max(latency_s, 0.0) / self.dt))
        self.n = n
        self.device = device
        self.jitter_steps = max(jitter_s, 0.0) / self.dt
        self.jittered = self.jitter_steps > 0.0
        self.mean_steps = max(latency_s, 0.0) / self.dt  # unrounded, for sampling
        if not self.jittered:
            self.buf: list = []
            return
        # The ring covers the mean plus six standard deviations, so tail samples are
        # still in the buffer.
        import math
        self.cap = max(self.n_steps + int(math.ceil(6.0 * self.jitter_steps)) + 1, 1)
        self.ring = None            # sized to the command shape on first call
        self.head = 0               # ring index the current command is written to
        self.age = torch.zeros(n, dtype=torch.long, device=device)  # steps since reset
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(seed))
        self._ar = torch.arange(n, device=device)

    def __call__(self, cmd):
        if not self.jittered:
            if self.n_steps <= 0:
                return cmd
            self.buf.append(cmd.clone())
            if len(self.buf) <= self.n_steps:
                return self.torch.zeros_like(cmd)
            return self.buf.pop(0)
        t = self.torch
        if self.ring is None:
            self.ring = t.zeros((self.cap,) + tuple(cmd.shape),
                                device=self.device, dtype=cmd.dtype)
        self.ring[self.head] = cmd
        # delay_steps ~ round(N(mean_steps, jitter_steps)), clamped to [0, history].
        d = t.normal(self.mean_steps, self.jitter_steps, size=(self.n,),
                     generator=self.gen, device=self.device).round().long()
        d = d.clamp_(min=0)
        avail = self.age.clamp(max=self.cap - 1)
        hold = d > avail                     # asked to read before this env's history
        d = t.minimum(d, avail)
        idx = (self.head - d) % self.cap
        out = self.ring[idx, self._ar].clone()
        out[hold] = 0.0
        self.head = (self.head + 1) % self.cap
        self.age = self.age + 1
        return out

    def reset_idx(self, idx) -> None:
        """Forget the queued commands of the given envs (called on episode reset)."""
        if not self.jittered:
            for b in self.buf:
                b[idx] = 0.0
            return
        # With age 0 the env reads nothing older than its reset; the ring itself need
        # not be cleared.
        self.age[idx] = 0


def tilt_from_accel(a_w, yaw, g: float = 9.80665, max_deg: float = 20.0, torch=None):
    """Body roll and pitch implied by a horizontal acceleration.

    tan(tilt) = |a| / g, resolved into the heading frame (forward accel -> pitch,
    lateral accel -> roll). Positive pitch is nose up, as in Isaac Lab's
    `euler_xyz_from_quat`, so accelerating forward gives negative pitch.

    Args:
        a_w: (n, 3) acceleration, world frame (only x and y are used).
        yaw: (n,) heading, radians.
    Returns:
        (roll, pitch), each (n,), radians, clamped to +/- max_deg.
    """
    import math
    if torch is None:
        import torch as _t
        torch = _t
    ax, ay = a_w[:, 0], a_w[:, 1]
    c, s = torch.cos(yaw), torch.sin(yaw)
    a_fwd = c * ax + s * ay
    a_lat = -s * ax + c * ay
    lim = math.radians(max_deg)
    pitch = (-torch.atan2(a_fwd, torch.full_like(a_fwd, g))).clamp(-lim, lim)
    roll = (torch.atan2(a_lat, torch.full_like(a_lat, g))).clamp(-lim, lim)
    return roll, pitch


def quat_from_euler_zyx(roll, pitch, yaw, torch=None):
    """(w, x, y, z) quaternion from roll/pitch/yaw; inverse of Isaac Lab's
    `euler_xyz_from_quat`."""
    if torch is None:
        import torch as _t
        torch = _t
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dim=-1)
