#!/usr/bin/env python3
"""Train one depth -> velocity PPO policy in Isaac Lab.

PPO with CleanRL conventions (orthogonal init, per-minibatch advantage normalisation,
clipped value loss, Adam eps 1e-5) on 256 parallel environments, with obstacle layouts
re-drawn at every episode reset. The defaults are the paper's settings; a regime is
chosen with the perturbation flags:

    P1-clean   (none)
    P2-lat     --cmd-latency 0.2 --cmd-latency-jitter 0.1
    P3-tilt    --attitude-coupling
    P4-comb    --cmd-latency 0.2 --cmd-latency-jitter 0.1 --attitude-coupling

    python scripts/train.py --width-mult 1.0 --seed 0 --out-dir results/r1 --run-id r1_w1.0_s0

Writes, for each --save-at budget D: {run_id}_d{D}.pt (weights) and
{run_id}_d{D}_snap.json; at the end {run_id}.pt and {run_id}_summary.json. The run id
gets the perturbation tag appended (e.g. r2_w1.0_s0_lat200j100).

The in-training evaluations (--eval-every) reset every environment and so change the
training trajectory; keep them at the defaults to reproduce the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--width-mult", type=float, default=1.0,
                    help="network width multiplier w (paper: 0.25 .. 2.0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--total-steps", type=int, default=2_000_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-horizon", type=int, default=3_000_000,
                    help="env-steps over which the LR anneals linearly (10%% floor)")
    ap.add_argument("--save-at", default="500000,1000000,1500000,2000000",
                    help="comma-separated env-step budgets; weights are saved at the first "
                         "rollout boundary at or past each")
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=500_000,
                    help="env-steps between in-training evaluations")
    ap.add_argument("--eval-steps", type=int, default=1400)
    ap.add_argument("--n-obstacles", type=int, default=140)
    ap.add_argument("--lateral-limit-m", type=float, default=11.0,
                    help="|y| beyond this ends the episode as a collision (0 = no boundary)")
    ap.add_argument("--cmd-latency", type=float, default=0.0, help="mean command delay, s")
    ap.add_argument("--cmd-latency-jitter", type=float, default=0.0,
                    help="std dev of the command delay, s")
    ap.add_argument("--attitude-coupling", action="store_true",
                    help="tilt body and camera by the acceleration-implied attitude")
    ap.add_argument("--attitude-max-deg", type=float, default=20.0)
    ap.add_argument("--out-dir", default="results/train")
    ap.add_argument("--run-id", default="")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    from isaaclab.app import AppLauncher
    AppLauncher({"headless": True, "enable_cameras": True})

    import torch
    import torch.nn as nn

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataclasses import replace

    from gpuperception.env import IsaacLabConfig
    from gpuperception.ppo_net import build_policy, param_count
    from gpuperception.realism import RealismCfg
    from gpuperception.rl_env import CourseRLEnv
    from gpuperception.spec import ScenarioSpec

    torch.manual_seed(args.seed)
    rz = RealismCfg(cmd_latency_s=args.cmd_latency,
                    cmd_latency_jitter_s=args.cmd_latency_jitter,
                    attitude_coupling=args.attitude_coupling,
                    attitude_max_deg=args.attitude_max_deg)
    run_id = args.run_id or f"ppo_w{args.width_mult}_s{args.seed}"
    if rz.any_on():
        run_id = f"{run_id}_{rz.tag()}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The spec seed only sets the spawn-time obstacle shapes; positions are re-drawn
    # per episode (random_courses). The name only enters spec_hash.
    spec = replace(ScenarioSpec(seed=1000), name="m1_course")
    spec = replace(spec, obstacles=replace(spec.obstacles, n_obstacles=args.n_obstacles))
    cfg = IsaacLabConfig(num_envs=args.num_envs, seed=args.seed, realism=rz,
                         random_courses=True, lateral_limit_m=args.lateral_limit_m,
                         speed_norm_clamp=True)
    env = CourseRLEnv(spec, cfg)
    H, W = env.depth_hw
    dev = env.device

    net = build_policy(state_dim=env.state_dim, width_mult=args.width_mult, device=dev)
    print(f"[train] {run_id}: width_mult={args.width_mult} params={param_count(net)}", flush=True)

    # CleanRL init: orthogonal weights, gain sqrt(2) on encoder and trunk, 0.01 on the
    # actor-mean head, 1.0 on the value head; zero biases.
    def _orth(module, gain):
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    _orth(net.enc, 2 ** 0.5)
    _orth(net.trunk, 2 ** 0.5)
    _orth(net.mu, 0.01)
    _orth(net.v, 1.0)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    act_scale = float(spec.max_speed_mps)

    def to_command(raw):
        """tanh-squash into the speed envelope."""
        return torch.tanh(raw) * act_scale

    @torch.no_grad()
    def evaluate(steps: int) -> dict:
        """Deterministic rollout (mean action) on the training distribution."""
        net.eval()
        obs = env.reset()
        succ = coll = tout = eps = 0
        for _ in range(steps):
            depth, state = env.encode(obs)
            mu, _, _ = net(depth.unsqueeze(1), state)
            obs, _, done, info = env.step(to_command(mu))
            if bool(done.any()):
                idx = done.nonzero(as_tuple=False).flatten()
                eps += int(idx.numel())
                succ += int(info["success"][idx].sum())
                coll += int(info["collision"][idx].sum())
                tout += int(info["timeout"][idx].sum())
        net.train()
        env.reset()
        return {"episodes": eps,
                "success_rate": round(succ / max(eps, 1), 4),
                "collision_rate": round(coll / max(eps, 1), 4),
                "timeout_rate": round(tout / max(eps, 1), 4)}

    obs = env.reset()
    n = env.n
    T = args.rollout
    env_steps = 0
    next_eval = 0
    save_at = sorted(int(x) for x in args.save_at.split(",") if x.strip())
    evals = []
    t_start = time.time()
    print(f"[train] {n} envs, {H}x{W} depth, {args.total_steps:,} env-steps", flush=True)

    while env_steps < args.total_steps:
        # ---- rollout (depth stored as float16 to save GPU memory) ----
        D = torch.zeros((T, n, 1, H, W), device=dev, dtype=torch.float16)
        S = torch.zeros((T, n, env.state_dim), device=dev)
        A = torch.zeros((T, n, 3), device=dev)
        LP = torch.zeros((T, n), device=dev)
        R = torch.zeros((T, n), device=dev)
        V = torch.zeros((T, n), device=dev)
        M = torch.zeros((T, n), device=dev)

        for t in range(T):
            depth, state = env.encode(obs)
            depth = depth.unsqueeze(1)
            with torch.no_grad():
                mu, log_std, value = net(depth, state)
                std = log_std.exp()
                raw = mu + std * torch.randn_like(mu)
                logp = (-0.5 * (((raw - mu) / std) ** 2) - log_std - 0.9189385).sum(-1)
            obs, rew, done, _ = env.step(to_command(raw))
            D[t], S[t], A[t], LP[t], R[t], V[t] = depth, state, raw, logp, rew, value
            M[t] = (~done).float()
            env_steps += n

        with torch.no_grad():
            depth, state = env.encode(obs)
            _, _, last_v = net(depth.unsqueeze(1), state)

        # ---- GAE ----
        adv = torch.zeros_like(R)
        gae = torch.zeros(n, device=dev)
        for t in reversed(range(T)):
            nxt = last_v if t == T - 1 else V[t + 1]
            delta = R[t] + args.gamma * nxt * M[t] - V[t]
            gae = delta + args.gamma * args.lam * M[t] * gae
            adv[t] = gae
        ret = adv + V

        # ---- update ----
        bD = D.reshape(-1, 1, H, W); bS = S.reshape(-1, env.state_dim)
        bA = A.reshape(-1, 3); bLP = LP.reshape(-1)
        bAdv = adv.reshape(-1); bRet = ret.reshape(-1)
        bV = V.reshape(-1)
        total = bD.shape[0]
        mb = max(1, total // args.minibatches)
        frac = 1.0 - min(env_steps / args.lr_horizon, 1.0)
        for g in opt.param_groups:
            g["lr"] = args.lr * max(frac, 0.1)
        for _ in range(args.epochs):
            perm = torch.randperm(total, device=dev)
            for i in range(0, total, mb):
                j = perm[i:i + mb]
                mu, log_std, value = net(bD[j].float(), bS[j])
                std = log_std.exp()
                logp = (-0.5 * (((bA[j] - mu) / std) ** 2) - log_std - 0.9189385).sum(-1)
                ratio = (logp - bLP[j]).exp()
                mbAdv = bAdv[j]
                mbAdv = (mbAdv - mbAdv.mean()) / (mbAdv.std() + 1e-8)
                pg = -torch.min(ratio * mbAdv,
                                ratio.clamp(1 - args.clip, 1 + args.clip) * mbAdv).mean()
                v_clip = bV[j] + (value - bV[j]).clamp(-args.clip, args.clip)
                vl = 0.5 * torch.max((value - bRet[j]) ** 2, (v_clip - bRet[j]) ** 2).mean()
                ent = (log_std + 1.4189385).sum(-1).mean()
                loss = pg + 0.5 * vl - args.entropy * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

        # Snapshots are taken before the periodic evaluation, which resets every env.
        while save_at and env_steps >= save_at[0]:
            d_ = save_at.pop(0)
            torch.save(net.state_dict(), out_dir / f"{run_id}_d{d_}.pt")
            (out_dir / f"{run_id}_d{d_}_snap.json").write_text(json.dumps({
                "run_id": run_id, "budget_steps": d_, "env_steps": env_steps,
                "wall_clock_s": time.time() - t_start,
                "width_mult": args.width_mult, "net_params": param_count(net),
                "seed": args.seed, "lr": args.lr, "lr_horizon": args.lr_horizon,
                "lr_now": opt.param_groups[0]["lr"],
                "courses_drawn": int(env.course.courses_drawn)}, indent=1))
            print(f"[snapshot] d={d_:,} at {env_steps:,} env-steps", flush=True)

        if env_steps >= next_eval:
            ev = evaluate(args.eval_steps)
            wall = time.time() - t_start
            evals.append({"env_steps": env_steps, "wall_clock_s": round(wall, 1), **ev})
            print(f"[{env_steps:>10,} steps] {env_steps / max(wall, 1e-9):>8,.0f} st/s  "
                  f"success={ev['success_rate']:.3f} collide={ev['collision_rate']:.3f} "
                  f"timeout={ev['timeout_rate']:.3f}  ret={float(R.sum(0).mean()):.2f}", flush=True)
            torch.save(net.state_dict(), out_dir / f"{run_id}.pt")
            next_eval = env_steps + args.eval_every
            obs = env.reset()

    ev = evaluate(args.eval_steps * 2)
    torch.save(net.state_dict(), out_dir / f"{run_id}.pt")
    wall = time.time() - t_start
    summary = {"run_id": run_id, "width_mult": args.width_mult, "net_params": param_count(net),
               "seed": args.seed, "realism": rz.tag(),
               "attitude_max_deg": args.attitude_max_deg if args.attitude_coupling else None,
               "courses_drawn": int(env.course.courses_drawn),
               "total_steps_target": args.total_steps, "lr_horizon": args.lr_horizon,
               "save_at": args.save_at, "evals": evals, "final_eval": ev,
               "env_steps": env_steps, "wall_clock_s": round(wall, 1),
               "steps_per_second": round(env_steps / max(wall, 1e-9), 1)}
    (out_dir / f"{run_id}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    sys.stdout.flush()
    # Isaac Lab's close() can deadlock; hard-exit once artefacts are on disk.
    os._exit(0)


if __name__ == "__main__":
    main()
