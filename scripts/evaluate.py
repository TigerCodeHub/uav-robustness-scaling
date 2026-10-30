#!/usr/bin/env python3
"""Evaluate one trained policy on one fixed evaluation course.

Deterministic policy (tanh of the mean action), 256 environments, 1500 steps. The
reported success is over the FIRST episode of every environment: every first episode
finishes inside the 1200-step timeout, so scoring only those avoids the bias of
discarding the slow episodes still in flight when the rollout ends.

    python scripts/evaluate.py --ckpt results/r1/r1_w1.0_s0_d2000000.pt --course 9000 \
        --out-dir results/r1/evalmc/c9000/w1.0_d2000000_s0 --cmd-latency 0.2 --cmd-latency-jitter 0.1

Writes summary_T_64x64{tag}.json (success, collision, counts, settings, provenance)
and outcomes_ppo_T{tag}.jsonl (one line per first episode: env, outcome, wall breach,
max |y|, steps), where {tag} is empty for clean and e.g. "_lat200j100_att" otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _provenance(args) -> dict:
    """Code version, software versions and checkpoint hash behind this summary."""
    import hashlib
    import platform
    import subprocess
    out = {"argv": sys.argv[1:], "python": platform.python_version()}
    try:
        import torch
        out["torch"] = torch.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        out["git_commit"] = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        pass
    h = hashlib.sha256()
    with open(args.ckpt, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    out["ckpt"] = os.path.basename(args.ckpt)
    out["ckpt_sha256"] = h.hexdigest()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="policy weights (*.pt) written by train.py")
    ap.add_argument("--course", type=int, default=9000,
                    help="evaluation course seed (paper: 9000-9002, 9100-9102)")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-obstacles", type=int, default=140)
    ap.add_argument("--lateral-limit-m", type=float, default=11.0)
    ap.add_argument("--cmd-latency", type=float, default=0.0, help="mean command delay, s")
    ap.add_argument("--cmd-latency-jitter", type=float, default=0.0,
                    help="std dev of the command delay, s")
    ap.add_argument("--attitude-coupling", action="store_true")
    ap.add_argument("--attitude-max-deg", type=float, default=20.0)
    ap.add_argument("--out-dir", default="results/eval")
    args = ap.parse_args()

    from isaaclab.app import AppLauncher
    AppLauncher({"headless": True, "enable_cameras": True})

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataclasses import replace

    from gpuperception.env import IsaacLabConfig
    from gpuperception.ppo_net import build_policy, infer_arch
    from gpuperception.realism import RealismCfg
    from gpuperception.rl_env import CourseRLEnv
    from gpuperception.spec import ScenarioSpec

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The spec name only enters spec_hash; kept so hashes match the archived summaries.
    spec = replace(ScenarioSpec(seed=args.course), name="m2_heldout")
    spec = replace(spec, obstacles=replace(spec.obstacles, n_obstacles=args.n_obstacles))
    rz = RealismCfg(cmd_latency_s=args.cmd_latency,
                    cmd_latency_jitter_s=args.cmd_latency_jitter,
                    attitude_coupling=args.attitude_coupling,
                    attitude_max_deg=args.attitude_max_deg)
    # Env seed 0 for every evaluation: all policies see the same start/goal offsets and
    # the same latency draws on a given course.
    env = CourseRLEnv(spec, IsaacLabConfig(num_envs=args.num_envs, seed=0, realism=rz,
                                           lateral_limit_m=args.lateral_limit_m,
                                           speed_norm_clamp=True))
    dev = env.device
    H, W = env.depth_hw
    act_scale = float(spec.max_speed_mps)

    sd = torch.load(args.ckpt, map_location="cpu")
    arch = infer_arch(sd)
    if arch["frame_stack"] != 1 or arch["state_dim"] != env.state_dim:
        raise SystemExit(f"checkpoint does not match this observation contract: {arch}")
    net = build_policy(state_dim=env.state_dim, channels=arch["channels"],
                       trunk=arch["trunk"], device=dev)
    net.load_state_dict(sd)
    net.eval()

    with torch.no_grad():
        env.course.reset_realism()
        n = env.n
        ep_index = [0] * n
        rows: list[dict] = []
        overlap_n = 0                        # same-step success and collision
        # running per-env max |y| and step count of the current episode (env-local)
        ymax = torch.zeros(n, device=dev)
        nstep = torch.zeros(n, dtype=torch.long, device=dev)
        t0 = time.time()

        obs = env.reset()
        for _ in range(args.steps):
            depth, state = env.encode(obs)
            mu, _, _ = net(depth.unsqueeze(1), state)
            cmd = torch.tanh(mu) * act_scale
            y = (obs["pos_w"] - env.course.scene.env_origins)[:, 1].abs()
            ymax = torch.maximum(ymax, y)
            nstep += 1
            obs, _, done, info = env.step(cmd)
            if bool(done.any()):
                di = done.nonzero(as_tuple=False).flatten()
                for i in di.tolist():
                    s_, c_ = bool(info["success"][i]), bool(info["collision"][i])
                    overlap_n += int(s_ and c_)
                    outcome = "success" if s_ else "collision" if c_ else "timeout"
                    if ep_index[i] == 0:
                        rows.append({"env": i, "outcome": outcome, "overlap": bool(s_ and c_),
                                     "lateral_breach": bool(info["lateral_breach"][i]),
                                     "max_abs_y": float(ymax[i]), "steps": int(nstep[i])})
                    ep_index[i] += 1
                ymax[di] = 0.0
                nstep[di] = 0

    first = [r["outcome"] for r in rows]
    arm = {"policy": "ppo", "kind": "learned",
           "n_first_ep": len(first),
           "success_first_ep": round(sum(o == "success" for o in first) / max(len(first), 1), 4),
           "collision_first_ep": round(sum(o == "collision" for o in first) / max(len(first), 1), 4),
           "n_envs_no_episode": sum(1 for k in ep_index if k == 0),
           "n_success_collision_overlap": overlap_n,
           "wall_clock_s": round(time.time() - t0, 1)}
    tag = "" if not rz.any_on() else f"_{rz.tag()}"
    summary = {"spec_hash": spec.spec_hash, "course_hash": spec.course_hash,
               "heldout_seed": args.course, "n_obstacles": spec.obstacles.n_obstacles,
               "depth_hw": [H, W], "num_envs": args.num_envs, "steps": args.steps,
               "realism": {"tag": rz.tag(), "cmd_latency_s": rz.cmd_latency_s,
                           "cmd_latency_jitter_s": rz.cmd_latency_jitter_s,
                           "attitude_coupling": rz.attitude_coupling,
                           "attitude_max_deg": rz.attitude_max_deg},
               "env": {"lateral_limit_m": args.lateral_limit_m, "speed_norm_clamp": True},
               "provenance": _provenance(args),
               "arms": {"ppo": arm}}
    with open(out_dir / f"outcomes_ppo_T{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out_dir / f"summary_T_{H}x{W}{tag}.json").write_text(json.dumps(summary, indent=2),
                                                         encoding="utf-8")
    print(f"success_first_ep={arm['success_first_ep']:.4f} collision_first_ep="
          f"{arm['collision_first_ep']:.4f} n={arm['n_first_ep']}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    # A crash must exit, not hang in Kit's teardown: print, then hard-exit.
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(3)
