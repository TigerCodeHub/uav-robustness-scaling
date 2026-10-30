"""The two perturbations do nothing when off and the specified thing when on."""

import math

import pytest
import torch

from gpuperception.realism import (CommandDelay, RealismCfg, quat_from_euler_zyx,
                                   tilt_from_accel)

DT = 0.05  # 20 Hz control


def test_tags_name_the_paper_conditions():
    assert RealismCfg().tag() == "none" and not RealismCfg().any_on()
    assert RealismCfg(cmd_latency_s=0.2, cmd_latency_jitter_s=0.1).tag() == "lat200j100"
    assert RealismCfg(attitude_coupling=True).tag() == "att"
    assert RealismCfg(cmd_latency_s=0.2, cmd_latency_jitter_s=0.1,
                      attitude_coupling=True).tag() == "lat200j100_att"


def test_no_delay_is_identity():
    d = CommandDelay(0.0, DT, 4, "cpu", torch)
    x = torch.randn(4, 3)
    assert torch.equal(d(x), x)


def test_fixed_delay_returns_the_command_from_n_steps_ago():
    d = CommandDelay(0.15, DT, 1, "cpu", torch)          # 3 steps
    outs = [d(torch.full((1, 3), float(i)))[0, 0].item() for i in range(8)]
    assert outs == [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]


def test_jittered_mean_lag_equals_the_mean_delay():
    n = 4096
    d = CommandDelay(0.2, DT, n, "cpu", torch, jitter_s=0.1, seed=0)   # 4 +/- 2 steps
    outs = [d(torch.full((n, 3), float(i))).clone() for i in range(200)]
    lags = [(i - outs[i]).mean().item() for i in range(50, 200)]
    # truncation at zero lifts the mean slightly above 4 steps
    assert abs(sum(lags) / len(lags) - 4.0) < 0.3


def test_jittered_delay_spreads_and_never_reads_the_future():
    n = 1024
    d = CommandDelay(0.2, DT, n, "cpu", torch, jitter_s=0.1, seed=1)
    for i in range(100):
        out = d(torch.full((n, 3), float(i)))
        assert (out <= i + 1e-6).all() and (out >= -1e-6).all()
    assert (99 - out).std().item() > 1.0


def test_jittered_reset_holds_only_that_env():
    d = CommandDelay(0.15, DT, 3, "cpu", torch, jitter_s=0.02, seed=3)
    for _ in range(6):
        d(torch.ones(3, 3))
    d.reset_idx(torch.tensor([1]))
    out = d(torch.ones(3, 3) * 5.0)
    assert out[1, 0].item() == 0.0
    assert out[0, 0].item() != 0.0 and out[2, 0].item() != 0.0


def test_jitter_is_deterministic_given_a_seed():
    a = CommandDelay(0.2, DT, 64, "cpu", torch, jitter_s=0.1, seed=7)
    b = CommandDelay(0.2, DT, 64, "cpu", torch, jitter_s=0.1, seed=7)
    for i in range(30):
        x = torch.full((64, 3), float(i))
        assert torch.equal(a(x), b(x))


def test_forward_accel_is_nose_down_pitch():
    g = 9.80665
    roll, pitch = tilt_from_accel(torch.tensor([[g, 0.0, 0.0]]), torch.zeros(1),
                                  max_deg=60.0, torch=torch)
    assert pitch.item() == pytest.approx(-math.pi / 4, abs=1e-6)
    assert roll.item() == pytest.approx(0.0, abs=1e-6)


def test_tilt_is_resolved_in_the_heading_frame():
    g = 9.80665
    roll, pitch = tilt_from_accel(torch.tensor([[0.0, g, 0.0]]), torch.tensor([math.pi / 2]),
                                  max_deg=60.0, torch=torch)
    assert pitch.item() == pytest.approx(-math.pi / 4, abs=1e-5)
    assert roll.item() == pytest.approx(0.0, abs=1e-5)


def test_tilt_is_capped():
    roll, pitch = tilt_from_accel(torch.tensor([[500.0, 500.0, 0.0]]), torch.zeros(1),
                                  max_deg=20.0, torch=torch)
    assert abs(pitch.item()) == pytest.approx(math.radians(20.0), abs=1e-6)
    assert abs(roll.item()) == pytest.approx(math.radians(20.0), abs=1e-6)


def test_quaternion_is_unit_and_identity_at_zero():
    r, p, y = torch.randn(16) * 0.3, torch.randn(16) * 0.3, torch.randn(16)
    q = quat_from_euler_zyx(r, p, y, torch=torch)
    assert torch.allclose(q.norm(dim=-1), torch.ones(16), atol=1e-6)
    z = torch.zeros(1)
    assert torch.allclose(quat_from_euler_zyx(z, z, z, torch=torch), torch.tensor([[1.0, 0, 0, 0]]))
