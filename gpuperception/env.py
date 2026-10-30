"""Batched Isaac Lab obstacle course: depth camera in, body-frame velocity command out.

* All environments render through one `TiledCamera` (64x64 `distance_to_image_plane`).
* Physics runs at 240 Hz; observations and control at 20 Hz.
* The vehicle is a kinematic sphere (radius 0.25 m) driven by writing its velocity: it
  tracks the commanded body-frame velocity through a first-order response with a
  drag-reduced acceleration limit. Yaw is not a policy output; the env turns the nose
  toward the horizontal velocity so the camera faces the flight path.
* Collisions come from PhysX contact forces. Flying above the ceiling or leaving
  |y| <= lateral_limit_m also ends the episode as a collision.

Every isaaclab import needs a live SimulationApp, so they are done lazily.

Smoke check (from the Isaac Lab python):
    python -m gpuperception.env --num-envs 64 --steps 20
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .realism import CommandDelay, RealismCfg, quat_from_euler_zyx, tilt_from_accel
from .spec import ScenarioSpec

# Textured MDL materials from the Isaac Sim asset server.
TEXTURE_MDLS: tuple[str, ...] = (
    "Masonry/Concrete_Rough.mdl",
    "Masonry/Brick_Wall_Brown.mdl",
    "Stone/Gravel.mdl",
    "Wood/Oak.mdl",
    "Wood/Plywood.mdl",
    "Carpet/Carpet_Berber_Gray.mdl",
)

# Isaac's default horizontal aperture, in mm; every FOV calculation depends on it.
_H_APERTURE_MM = 20.955


def _focal_px(fov_deg: float, px: int) -> float:
    import math
    return (px / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def _focal_mm(sensor) -> float:
    """Focal length in mm that renders `sensor.hfov_deg` at `sensor.width`."""
    return _focal_px(sensor.hfov_deg, sensor.width) * _H_APERTURE_MM / sensor.width


def _vertical_aperture_mm(sensor) -> float:
    """Vertical aperture giving `sensor.vfov_deg` at the same focal length (87 x 58 deg
    is not square, so one aperture for both axes would be wrong)."""
    return _focal_mm(sensor) * sensor.height / _focal_px(sensor.vfov_deg, sensor.height)


def _mdl_material(sim_utils, index: int, scale: float = 2.0):
    """An MDL material config, or None if the asset root is unavailable.
    `project_uvw=True` is needed because procedural primitives carry no UVs."""
    try:
        from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR
        rel = TEXTURE_MDLS[index % len(TEXTURE_MDLS)]
        return sim_utils.MdlFileCfg(
            mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/{rel}",
            project_uvw=True,
            texture_scale=(scale, scale),
        )
    except Exception:  # noqa: BLE001
        return None


@dataclass
class IsaacLabConfig:
    """Backend settings that are not part of the course itself."""

    num_envs: int = 256
    device: str = "cuda:0"
    physics_hz: float = 240.0
    obs_hz: float = 20.0
    env_spacing: float = 40.0
    # Seeds the per-env start/goal offsets, the latency draws and (in training) the
    # course stream.
    seed: int = 0
    realism: RealismCfg = field(default_factory=RealismCfg)
    # Training: re-draw every env's obstacle layout on each episode reset, from the same
    # sampler as the evaluation courses with fresh seeds in [1e6, 2^31). Obstacles are
    # then kinematic bodies that keep their spawn-time shapes and radii. Evaluation
    # leaves this off and flies the fixed course of `spec.seed`.
    random_courses: bool = False
    # |y| > lateral_limit_m ends the episode as a collision (0 disables the boundary).
    lateral_limit_m: float = 11.0
    # Clamp the commanded speed magnitude (True) rather than each axis (False).
    speed_norm_clamp: bool = True

    @property
    def render_interval(self) -> int:
        """Physics steps per rendered observation."""
        return max(1, int(round(self.physics_hz / self.obs_hz)))


def build_scene_cfg(spec: ScenarioSpec, cfg: IsaacLabConfig):
    """Compile a ScenarioSpec into an Isaac Lab InteractiveSceneCfg."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
    from isaaclab.utils import configclass

    from .spec import sample_obstacles

    obs = spec.obstacles
    sensor = spec.sensor

    @configclass
    class CourseSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.CuboidCfg(
                size=(1000.0, 1000.0, 0.2),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=(_mdl_material(sim_utils, 0, obs.texture_scale) if obs.textured
                                 else sim_utils.PreviewSurfaceCfg(
                                     diffuse_color=(0.25, 0.25, 0.28))),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0),
        )
        drone = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/drone",
            spawn=sim_utils.SphereCfg(
                radius=0.25,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True, max_depenetration_velocity=5.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=spec.mass_kg),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                activate_contact_sensors=True,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.6, 0.9)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=spec.start_m),
        )
        contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/drone",
            update_period=0.0,
            history_length=0,
        )
        tiled_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/drone/camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.25, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0),
                convention="world"),
            data_types=["distance_to_image_plane"],
            # FOV derived from the spec: f_px = (px/2)/tan(fov/2), f_mm = f_px * aperture / px.
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=_focal_mm(sensor),
                horizontal_aperture=_H_APERTURE_MM,
                vertical_aperture=_vertical_aperture_mm(sensor),
                clipping_range=(sensor.near_m, sensor.far_m)),
            width=sensor.width,
            height=sensor.height,
        )

    scene_cfg = CourseSceneCfg(num_envs=cfg.num_envs, env_spacing=cfg.env_spacing)

    placed = sample_obstacles(spec)
    for i, ob in enumerate(placed):
        if ob.kind == "sphere":
            spawn = sim_utils.SphereCfg(
                radius=ob.radius,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=(_mdl_material(sim_utils, i + 1, obs.texture_scale) if obs.textured
                                 else sim_utils.PreviewSurfaceCfg(
                                     diffuse_color=(0.55, 0.5, 0.45))))
        else:
            spawn = sim_utils.CuboidCfg(
                size=(2 * ob.radius, 2 * ob.radius, ob.height),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=(_mdl_material(sim_utils, i + 2, obs.texture_scale) if obs.textured
                                 else sim_utils.PreviewSurfaceCfg(
                                     diffuse_color=(0.45, 0.45, 0.5))))
        pos = (ob.x, ob.y, ob.z)
        if cfg.random_courses:
            # Kinematic so the pose can be written per env on reset; PhysX still reports
            # contacts against the dynamic drone body.
            spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True)
            spawn.mass_props = sim_utils.MassPropertiesCfg(mass=1.0)
            setattr(scene_cfg, f"obs_{i}", RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/obs_{i}",
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos)))
        else:
            setattr(scene_cfg, f"obs_{i}", AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/obs_{i}",
                spawn=spawn,
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos)))

    return scene_cfg, placed


class IsaacLabCourse:
    """Batched obstacle course exposing plain tensors (not a gymnasium env)."""

    def __init__(self, spec: ScenarioSpec, cfg: IsaacLabConfig | None = None):
        import math

        import isaaclab.sim as sim_utils
        import torch
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext

        self.spec = spec
        self.cfg = cfg or IsaacLabConfig()
        self.torch = torch

        # Cheapest renderer settings; DLSS and the OptiX denoiser are not needed for depth.
        render_cfg = None
        try:
            import dataclasses

            from isaaclab.sim import RenderCfg
            wanted = {"antialiasing_mode": "Off", "enable_dl_denoiser": False,
                      "enable_dlssg": False, "rendering_mode": "performance",
                      "enable_reflections": False,
                      "enable_global_illumination": False,
                      "samples_per_pixel": 1}
            valid = {f.name for f in dataclasses.fields(RenderCfg)}
            render_cfg = RenderCfg(**{k: v for k, v in wanted.items() if k in valid})
        except Exception:  # noqa: BLE001
            pass

        sim_kwargs: dict[str, Any] = dict(
            device=self.cfg.device,
            dt=1.0 / self.cfg.physics_hz,
            render_interval=self.cfg.render_interval,
        )
        if render_cfg is not None:
            sim_kwargs["render"] = render_cfg
        self.sim = SimulationContext(sim_utils.SimulationCfg(**sim_kwargs))

        scene_cfg, self.obstacles = build_scene_cfg(spec, self.cfg)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.drone = self.scene["drone"]
        self.camera = self.scene["tiled_camera"]
        self.contact = self.scene["contact"]
        self.n = self.cfg.num_envs
        dev = self.cfg.device

        self.goal = torch.tensor(spec.goal_m, device=dev).repeat(self.n, 1)
        self.start = torch.tensor(spec.start_m, device=dev).repeat(self.n, 1)
        self.step_count = torch.zeros(self.n, dtype=torch.long, device=dev)
        self.max_steps = spec.max_steps

        self.realism = self.cfg.realism
        self.reset_realism()
        if self.realism.any_on():
            print(f"[env] perturbation on: tag={self.realism.tag()} "
                  f"latency={self.realism.cmd_latency_s}s jitter={self.realism.cmd_latency_jitter_s}s "
                  f"attitude={self.realism.attitude_coupling} "
                  f"(max {self.realism.attitude_max_deg} deg)", flush=True)

        # Start and goal are offset laterally (perpendicular to the route) at random on
        # every reset, so each env is a different problem.
        self._rng = torch.Generator(device=dev)
        self._rng.manual_seed(self.cfg.seed)
        dx = spec.goal_m[0] - spec.start_m[0]
        dy = spec.goal_m[1] - spec.start_m[1]
        L = max(math.hypot(dx, dy), 1e-6)
        self._perp = (-dy / L, dx / L)
        self._lateral_m = 6.0
        self.randomize_endpoints(torch.arange(self.n, device=dev))

        self._rc = bool(self.cfg.random_courses)
        self.courses_drawn = 0
        if self._rc:
            import numpy as _np
            self._obj = [self.scene[f"obs_{i}"] for i in range(len(self.obstacles))]
            # Slot z is fixed by the slot's spawned geometry.
            self._slot_z = [o.z for o in self.obstacles]
            # Course seeds are drawn from [1e6, 2^31), disjoint from the evaluation courses.
            self._course_rng = _np.random.default_rng([int(self.cfg.seed), 7919])
            self.randomize_courses(torch.arange(self.n, device=dev))
            print(f"[env] random courses: {len(self._obj)} kinematic obstacles/env, "
                  f"re-drawn per episode", flush=True)

    def randomize_courses(self, idx) -> None:
        """Draw a fresh obstacle layout for each env in `idx`."""
        if not self._rc or idx.numel() == 0:
            return
        torch = self.torch
        from dataclasses import replace

        from .spec import sample_obstacles
        k = idx.numel()
        n_obs = len(self._obj)
        pos = torch.empty((k, n_obs, 3), dtype=torch.float32)
        for j in range(k):
            seed = int(self._course_rng.integers(1_000_000, 2 ** 31 - 1))
            field_ = sample_obstacles(replace(self.spec, seed=seed))
            pos[j, :, 0] = torch.tensor([o.x for o in field_])
            pos[j, :, 1] = torch.tensor([o.y for o in field_])
        pos[:, :, 2] = torch.tensor(self._slot_z)
        self.courses_drawn += k
        dev = self.scene.env_origins.device
        pos = pos.to(dev) + self.scene.env_origins[idx][:, None, :]
        quat = torch.zeros((k, 4), device=dev)
        quat[:, 0] = 1.0
        for i, obj in enumerate(self._obj):
            obj.write_root_pose_to_sim(torch.cat([pos[:, i, :], quat], dim=-1), env_ids=idx)

    def randomize_endpoints(self, idx) -> None:
        """Offset start and goal laterally by up to +/- 6 m."""
        torch = self.torch
        if idx.numel() == 0:
            return
        k = idx.numel()
        span = self._lateral_m
        px, py = self._perp
        so = (torch.rand(k, generator=self._rng, device=idx.device) * 2 - 1) * span
        go = (torch.rand(k, generator=self._rng, device=idx.device) * 2 - 1) * span
        self.start[idx, 0] = float(self.spec.start_m[0]) + so * px
        self.start[idx, 1] = float(self.spec.start_m[1]) + so * py
        self.goal[idx, 0] = float(self.spec.goal_m[0]) + go * px
        self.goal[idx, 1] = float(self.spec.goal_m[1]) + go * py

    # ---- observation ----------------------------------------------------

    def depth_m(self):
        """Depth in metres, (n, H, W), clipped to the sensor range; no return reads far_m."""
        d = self.camera.data.output["distance_to_image_plane"]
        d = d.squeeze(-1) if d.ndim == 4 else d
        d = self.torch.nan_to_num(d, nan=self.spec.sensor.far_m, posinf=self.spec.sensor.far_m)
        return d.clamp(self.spec.sensor.near_m, self.spec.sensor.far_m)

    def observation(self) -> dict:
        """Depth plus body-frame goal and velocity (Isaac reports world frame)."""
        from isaaclab.utils.math import quat_rotate_inverse

        pos = self.drone.data.root_pos_w
        quat = self.drone.data.root_quat_w
        goal_rel_w = self.goal + self.scene.env_origins - pos
        return {
            "depth": self.depth_m(),
            "goal_rel": quat_rotate_inverse(quat, goal_rel_w),
            "vel": self.drone.data.root_lin_vel_b,
            "pos_w": pos,
            "goal_dist_true": goal_rel_w.norm(dim=-1),
        }

    # ---- step -----------------------------------------------------------

    def reset_realism(self) -> None:
        """Fresh latency queue for every env (seeded by cfg.seed)."""
        self._delay = CommandDelay(self.realism.cmd_latency_s, 1.0 / float(self.cfg.obs_hz),
                                   self.n, self.cfg.device, self.torch,
                                   jitter_s=self.realism.cmd_latency_jitter_s,
                                   seed=self.cfg.seed)

    def step(self, command):
        """command: (n, 3) body-frame [vx, vy, vz] in m/s."""
        torch = self.torch
        import math

        from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate

        # The delay wraps the raw policy output, before clamping or frame rotation.
        command = self._delay(command)
        if self.cfg.speed_norm_clamp:
            v_b = command[:, :3]
            _n = v_b.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            v_b = v_b * (float(self.spec.max_speed_mps) / _n).clamp(max=1.0)
        else:
            v_b = command[:, :3].clamp(-self.spec.max_speed_mps, self.spec.max_speed_mps)
        quat = self.drone.data.root_quat_w
        v_cmd_w = quat_rotate(quat, v_b)

        # First-order velocity response, then an acceleration clamp whose budget is
        # reduced by aerodynamic drag.
        dt = 1.0 / float(self.cfg.obs_hz)
        v_cur = self.drone.data.root_lin_vel_w
        tau = max(float(self.spec.vel_tau_s), 1e-3)
        alpha = 1.0 - math.exp(-dt / tau)
        v_target = v_cur + alpha * (v_cmd_w - v_cur)
        dv = v_target - v_cur
        rho = 1.225
        speed = v_cur.norm(dim=-1, keepdim=True)
        a_drag = (0.5 * rho * float(self.spec.drag_cd_area_m2)
                  * speed ** 2 / max(float(self.spec.mass_kg), 1e-3))
        a_avail = (float(self.spec.max_accel_mps2) - a_drag).clamp(min=0.5)
        dv_max = a_avail * dt
        nrm = dv.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        dv = dv * (dv_max / nrm).clamp(max=1.0)
        v_w = v_cur + dv

        # Nose along horizontal travel, only when actually moving.
        yaw_des = torch.atan2(v_w[:, 1], v_w[:, 0])
        _, _, yaw = euler_xyz_from_quat(quat)
        err = torch.atan2(torch.sin(yaw_des - yaw), torch.cos(yaw_des - yaw))
        yaw_rate = (2.0 * err).clamp(-2.0, 2.0)
        moving = v_w[:, :2].norm(dim=-1) > 0.2
        yaw_rate = torch.where(moving, yaw_rate, torch.zeros_like(yaw_rate))

        root_vel = torch.zeros((self.n, 6), device=command.device)
        root_vel[:, :3] = v_w
        root_vel[:, 5] = yaw_rate
        self.drone.write_root_velocity_to_sim(root_vel)

        if self.realism.attitude_coupling:
            # Write the tilt implied by the achieved acceleration, at the CURRENT yaw
            # (physics integrates yaw_rate itself over this step).
            a_w = dv / dt
            roll_t, pitch_t = tilt_from_accel(
                a_w, yaw, max_deg=self.realism.attitude_max_deg, torch=torch)
            q_t = quat_from_euler_zyx(roll_t, pitch_t, yaw, torch=torch)
            pose = torch.cat([self.drone.data.root_pos_w, q_t], dim=-1)
            self.drone.write_root_pose_to_sim(pose)

        for _ in range(self.cfg.render_interval):
            self.sim.step(render=False)
        self.sim.render()
        self.scene.update(1.0 / self.cfg.obs_hz)
        self.step_count += 1

        obs = self.observation()
        goal_dist = obs["goal_rel"].norm(dim=-1)
        clearance = self.depth_m().flatten(1).min(dim=-1).values
        success = obs["goal_dist_true"] < self.spec.goal_tolerance_m
        forces = self.contact.data.net_forces_w
        collision = forces.flatten(1).norm(dim=-1) > 1.0
        ceil = float(self.spec.ceiling_m or 0.0)
        if ceil > 0:
            z_local = obs["pos_w"][:, 2] - self.scene.env_origins[:, 2]
            collision = collision | (z_local > ceil - 0.25)
        lateral_breach = torch.zeros_like(collision)
        if self.cfg.lateral_limit_m > 0:
            y_local = obs["pos_w"][:, 1] - self.scene.env_origins[:, 1]
            lateral_breach = y_local.abs() > self.cfg.lateral_limit_m
            collision = collision | lateral_breach
        timeout = self.step_count >= self.max_steps
        return obs, {
            "goal_dist": goal_dist,
            "clearance": clearance,
            "cmd_applied_b": v_b.detach(),     # what the vehicle chased; used by the jerk penalty
            "success": success,
            "collision": collision,
            "lateral_breach": lateral_breach,
            "timeout": timeout,
            "done": success | collision | timeout,
        }

    def reset_idx(self, idx) -> None:
        if idx.numel() == 0:
            return
        self.randomize_endpoints(idx)
        if self._rc:
            self.randomize_courses(idx)
        state = self.drone.data.default_root_state[idx].clone()
        state[:, :3] = self.start[idx] + self.scene.env_origins[idx]
        self.drone.write_root_pose_to_sim(state[:, :7], idx)
        self.drone.write_root_velocity_to_sim(self.torch.zeros_like(state[:, 7:]), idx)
        self.step_count[idx] = 0
        self._delay.reset_idx(idx)


def _check() -> int:
    """Smoke check: build a course, fly forward, verify the depth contract."""
    import argparse
    import json
    import os
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    from isaaclab.app import AppLauncher
    AppLauncher({"headless": True, "enable_cameras": True})

    result: dict[str, Any] = {}
    try:
        import torch

        spec = ScenarioSpec(name="smoke", seed=0)
        env = IsaacLabCourse(spec, IsaacLabConfig(num_envs=args.num_envs))
        cmd = torch.zeros((env.n, 3), device=env.cfg.device)
        cmd[:, 0] = 2.0
        for _ in range(args.steps):
            obs, info = env.step(cmd)
        d = obs["depth"]
        result = {
            "ok": True,
            "num_envs": env.n,
            "depth_shape": list(d.shape),
            "depth_min_m": round(float(d.min()), 3),
            "depth_max_m": round(float(d.max()), 3),
            "obstacles_placed": len(env.obstacles),
            "mean_goal_dist_m": round(float(info["goal_dist"].mean()), 2),
            "collision_frac": round(float(info["collision"].float().mean()), 3),
        }
    except Exception as exc:  # noqa: BLE001
        import traceback
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300],
                  "traceback": traceback.format_exc()[-1500:]}

    print(json.dumps(result, indent=2), flush=True)
    sys.stdout.flush()
    # Isaac Lab's close() can deadlock; hard-exit once results are out.
    os._exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    _check()
