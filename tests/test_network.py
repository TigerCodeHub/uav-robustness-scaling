"""Network sizes, the analytic counts and checkpoint architecture recovery agree."""

import os
import sys

import pytest
import torch

from gpuperception.ppo_net import build_policy, infer_arch, param_count

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import flops  # noqa: E402

WIDTHS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)


@pytest.mark.parametrize("w", WIDTHS)
def test_analytic_param_count_matches_the_network(w):
    assert flops.count(w)[0] == param_count(build_policy(state_dim=7, width_mult=w))


def test_paper_size_range():
    assert flops.count(0.25)[0] == 5759
    assert flops.count(2.0)[0] == 323783


@pytest.mark.parametrize("w", (0.25, 1.0, 1.75))
def test_infer_arch_round_trip(w):
    net = build_policy(state_dim=7, width_mult=w)
    arch = infer_arch(net.state_dict())
    rebuilt = build_policy(state_dim=arch["state_dim"], channels=arch["channels"],
                           trunk=arch["trunk"])
    rebuilt.load_state_dict(net.state_dict())
    assert arch["frame_stack"] == 1 and arch["state_dim"] == 7
    x, s = torch.rand(5, 1, 64, 64), torch.rand(5, 7)
    assert torch.equal(net(x, s)[0], rebuilt(x, s)[0])


def test_training_compute_is_13F_per_step():
    _, f = flops.count(1.0)
    assert flops.budget(f, 1, 4)[2] == 13 * f
