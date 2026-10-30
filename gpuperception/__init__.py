"""Depth-camera UAV obstacle navigation in Isaac Lab: network size vs. robustness.

Modules:
    spec.py       course and vehicle specification, seeded obstacle sampling
    realism.py    the two perturbations: command latency and attitude (tilt) coupling
    env.py        batched Isaac Lab obstacle course (depth camera, velocity commands)
    rl_env.py     reward and auto-reset wrapper used for PPO
    policy_io.py  the policy's input transform (depth normalisation, 7-D state)
    ppo_net.py    the actor-critic network and its width multiplier

Only `spec`, `realism`, `policy_io` and `ppo_net` import without Isaac Lab; `env` and
`rl_env` need a running Isaac Sim app (see scripts/train.py).
"""

__version__ = "1.0.0"
