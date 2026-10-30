"""Analytic parameter and FLOP counts for the actor-critic in gpuperception/ppo_net.py.

Forward cost F per decision covers the encoder, trunk, actor and critic heads
(FLOP = 2 x multiply-accumulate; biases and ReLU ignored). Training compute is
TOTAL_STEPS x (F + EPOCHS x 3F): one rollout forward per env-step plus EPOCHS passes of
forward and backward (~2F) over every sample, i.e. 13F per step at 4 epochs.

    python analysis/flops.py
"""

from __future__ import annotations

BASE_CHANNELS = (16, 32, 32)
BASE_TRUNK = (256, 128)
N_ACT = 3


def scale(width):
    s = lambda x: max(1, int(round(x * width)))  # noqa: E731
    return tuple(s(c) for c in BASE_CHANNELS), tuple(s(t) for t in BASE_TRUNK)


def conv_out(h, k, stride, pad):
    return (h + 2 * pad - k) // stride + 1


def count(width, hw=64, frame_stack=1, state_dim=7):
    """Return (params, forward_flop_per_decision)."""
    (c1, c2, c3), (t1, t2) = scale(width)
    h1 = conv_out(hw, 5, 2, 2)
    h2 = conv_out(h1, 3, 2, 1)
    h3 = conv_out(h2, 3, 2, 1)
    macs = (h1 * h1 * c1 * frame_stack * 25
            + h2 * h2 * c2 * c1 * 9
            + h3 * h3 * c3 * c2 * 9)
    enc = c3 * 4
    macs += (enc + state_dim) * t1 + t1 * t2 + t2 * N_ACT + t2 * 1
    params = (c1 * frame_stack * 25 + c1 + c2 * c1 * 9 + c2 + c3 * c2 * 9 + c3
              + (enc + state_dim) * t1 + t1 + t1 * t2 + t2
              + t2 * N_ACT + N_ACT + t2 + 1 + N_ACT)
    return params, 2 * macs


def budget(fwd, steps, epochs=4):
    """(rollout, update, total) training FLOP."""
    rollout = fwd * steps
    update = epochs * 3 * fwd * steps
    return rollout, update, rollout + update


if __name__ == "__main__":
    print(f"{'w':>5} {'params':>9} {'F MFLOP':>8} {'train TFLOP @2M':>16}")
    for w in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
        n, f = count(w)
        print(f"{w:>5} {n:>9,} {f / 1e6:>8.2f} {budget(f, 2_000_000)[2] / 1e12:>16.0f}")
