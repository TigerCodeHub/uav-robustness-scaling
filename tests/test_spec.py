"""Courses are reproducible and match the archived results."""

from dataclasses import replace

from gpuperception.spec import ScenarioSpec, sample_obstacles


def eval_spec(course):
    s = replace(ScenarioSpec(seed=course), name="m2_heldout")
    return replace(s, obstacles=replace(s.obstacles, n_obstacles=140))


def test_hashes_match_the_archived_evaluations():
    # spec_hash / course_hash as recorded in the archived summary files
    assert (eval_spec(9000).spec_hash, eval_spec(9000).course_hash) == ("51ede65bbd9d", "1dd822f109b8")
    assert (eval_spec(9100).spec_hash, eval_spec(9100).course_hash) == ("182c6433f022", "8f6168451a51")


def test_course_composition():
    obs = sample_obstacles(eval_spec(9000))
    assert len(obs) == 140
    assert sum(o.kind == "sphere" for o in obs) == 47      # every third obstacle
    assert all(0.2 <= o.radius <= 0.6 for o in obs)
    assert all(o.height == 4.0 for o in obs if o.kind == "box")   # floor-to-ceiling pillars
    assert all(o.z + o.radius < 2.0 for o in obs if o.kind == "sphere")  # below flight altitude


def test_same_seed_same_course_different_seed_different_course():
    a, b, c = (sample_obstacles(eval_spec(s)) for s in (9000, 9000, 9001))
    assert a == b
    assert a != c


def test_first_obstacle_is_stable():
    o = sample_obstacles(eval_spec(9000))[0]
    assert o.kind == "sphere"
    assert abs(o.x - 11.743747768289094) < 1e-12
    assert abs(o.radius - 0.4561341968379374) < 1e-12


def test_timeout_is_1200_steps():
    assert ScenarioSpec().max_steps == 1200
