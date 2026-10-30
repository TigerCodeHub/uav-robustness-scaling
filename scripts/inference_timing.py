#!/usr/bin/env python3
"""Time one policy decision (batch 1) at every width, from the archived checkpoints.

A decision is what the deployed policy runs each control step: the forward pass on one
64x64 depth image and the 7-D state, then tanh of the mean scaled to 3 m/s. On cuda the
time also covers copying the inputs to the GPU and the action back. No simulator is
needed; inputs are random, which does not change the cost of this network.

Widths are interleaved over several rounds so slow drift in clock speed is shared
across widths. Pin the process to chosen cores from outside, e.g.

    taskset -c 2 python scripts/inference_timing.py --threads 1 --out timing_cpu1.json
    python scripts/inference_timing.py --device cuda --out timing_cuda.json

Writes per-width median, p99 and mean time per decision (ms) and the parameter count.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpuperception.ppo_net import build_policy, infer_arch, param_count  # noqa: E402

WIDTHS = ("0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "1.75", "2.0")


def _cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt-dir", default="data/results/r1")
    ap.add_argument("--pattern", default="r1_w{w}_s0_d2000000.pt")
    ap.add_argument("--widths", default=",".join(WIDTHS))
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=1, help="torch intra-op threads (cpu)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=2000, help="timed decisions per width per round")
    ap.add_argument("--warmup", type=int, default=200, help="untimed decisions per width per round")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    gen = torch.Generator().manual_seed(0)

    nets = {}
    for w in args.widths.split(","):
        sd = torch.load(Path(args.ckpt_dir) / args.pattern.format(w=w), map_location="cpu")
        a = infer_arch(sd)
        net = build_policy(state_dim=a["state_dim"], channels=a["channels"], trunk=a["trunk"],
                           frame_stack=a["frame_stack"])
        net.load_state_dict(sd)
        nets[w] = net.eval().to(dev)

    depth = torch.rand(1, 1, 64, 64, generator=gen)
    state = torch.randn(1, 7, generator=gen)

    def decide(net):
        mu = net(depth.to(dev), state.to(dev))[0]
        return (torch.tanh(mu) * 3.0).cpu()  # .cpu() waits for the GPU

    times = {w: [] for w in nets}
    with torch.inference_mode():
        for _ in range(args.rounds):
            for w, net in nets.items():
                for _ in range(args.warmup):
                    decide(net)
                for _ in range(args.iters):
                    t0 = time.perf_counter_ns()
                    decide(net)
                    times[w].append(time.perf_counter_ns() - t0)

    rows = []
    for w, net in nets.items():
        ms = np.asarray(times[w]) / 1e6
        rows.append({"width": float(w), "params": param_count(net), "n": len(ms),
                     "median_ms": float(np.median(ms)), "p99_ms": float(np.percentile(ms, 99)),
                     "mean_ms": float(ms.mean())})
        print(f"w={w:>5}  N={rows[-1]['params']:>7,}  median {rows[-1]['median_ms']:.4f} ms  "
              f"p99 {rows[-1]['p99_ms']:.4f} ms")

    meta = {"device": args.device, "threads": args.threads, "rounds": args.rounds,
            "iters": args.iters, "warmup": args.warmup, "cpu": _cpu_model(),
            "torch": torch.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None}
    Path(args.out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
