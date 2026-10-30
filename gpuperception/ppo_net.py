"""The PPO actor-critic network, shared by training and evaluation.

A conv encoder on the depth image, concatenated with the 7-D state, feeds a two-layer
trunk with actor-mean and value heads; the action log-std is a free parameter.
`width_mult` scales the three conv channel groups (16, 32, 32) and the two trunk widths
(256, 128) together; w = 1.0 is the base network.

torch is imported inside the functions: Isaac Lab's AppLauncher must start before
torch touches CUDA.
"""

from __future__ import annotations

BASE_CHANNELS = (16, 32, 32)  # three conv encoder stages
BASE_TRUNK = (256, 128)       # two trunk layers; the heads read the last one
N_ACT = 3                     # body-frame velocity command


def scale_dims(width_mult: float):
    """(channels, trunk) widths for a multiplier, each rounded and at least 1."""
    def s(x: float) -> int:
        return max(1, int(round(x * float(width_mult))))
    return tuple(s(c) for c in BASE_CHANNELS), tuple(s(t) for t in BASE_TRUNK)


def build_policy(*, state_dim: int, width_mult: float = 1.0, channels=None, trunk=None,
                 frame_stack: int = 1, n_act: int = N_ACT, device=None):
    """Build the actor-critic net. Explicit `channels`/`trunk` (see `infer_arch`)
    override `width_mult`."""
    import torch
    from torch import nn

    if channels is None or trunk is None:
        channels, trunk = scale_dims(width_mult)
    c1, c2, c3 = channels
    t1, t2 = trunk

    class Policy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(frame_stack, c1, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(2), nn.Flatten(),
            )
            enc_dim = c3 * 2 * 2
            self.trunk = nn.Sequential(
                nn.Linear(enc_dim + state_dim, t1), nn.ReLU(),
                nn.Linear(t1, t2), nn.ReLU(),
            )
            self.mu = nn.Linear(t2, n_act)
            self.v = nn.Linear(t2, 1)
            self.log_std = nn.Parameter(torch.full((n_act,), -0.7))

        def forward(self, depth, state):
            """depth: (n, 1, H, W) in [0, 1]; state: (n, 7). Returns (mu, log_std, value)."""
            f = self.trunk(torch.cat([self.enc(depth), state], dim=-1))
            mu = self.mu(f)
            return mu, self.log_std.expand_as(mu), self.v(f).squeeze(-1)

    net = Policy()
    return net.to(device) if device is not None else net


def infer_arch(state_dict) -> dict:
    """Recover the architecture of a checkpoint from its tensor shapes."""
    def w(suffix):
        return next(v for k, v in state_dict.items() if k.endswith(suffix))

    w0 = w("enc.0.weight")           # (c1, frame_stack, 5, 5)
    c1 = int(w0.shape[0])
    c2 = int(w("enc.2.weight").shape[0])
    c3 = int(w("enc.4.weight").shape[0])
    t0 = w("trunk.0.weight")          # (t1, enc_dim + state_dim)
    t1 = int(t0.shape[0])
    t2 = int(w("trunk.2.weight").shape[0])
    return {"frame_stack": int(w0.shape[1]), "channels": (c1, c2, c3), "trunk": (t1, t2),
            "state_dim": int(t0.shape[1]) - c3 * 2 * 2}


def param_count(net) -> int:
    return sum(p.numel() for p in net.parameters())
