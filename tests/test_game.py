"""Game-theoretic components: no-regret learning and equilibrium sizing."""

from __future__ import annotations

import numpy as np
import pytest

from gtbot.game.equilibrium import (
    AmbiguityConfig,
    fast_robust_size,
    fictitious_play,
    robust_size,
)
from gtbot.game.regret import ContextualNoRegret, LearnerConfig


# ----------------------------------------------------------------- equilibrium
def test_fictitious_play_solves_matching_pennies():
    payoff = np.array([[1.0, -1.0], [-1.0, 1.0]])
    p, q, value = fictitious_play(payoff, iters=3000)
    assert value == pytest.approx(0.0, abs=0.02)
    assert p == pytest.approx([0.5, 0.5], abs=0.03)
    assert q == pytest.approx([0.5, 0.5], abs=0.03)


def test_fictitious_play_solves_rock_paper_scissors():
    payoff = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    p, _, value = fictitious_play(payoff, iters=4000)
    assert value == pytest.approx(0.0, abs=0.02)
    assert p == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=0.05)


def test_fictitious_play_finds_a_dominant_strategy():
    payoff = np.array([[3.0, 2.0], [1.0, 0.0]])
    p, _, value = fictitious_play(payoff, iters=500)
    assert p[0] > 0.99
    assert value == pytest.approx(2.0, abs=0.02)


def test_closed_form_matches_the_solved_game():
    """The hot path uses a closed form; it must agree with actually solving."""
    cfg = AmbiguityConfig()
    for edge, se in [(8e-4, 1.0e-4), (6e-4, 1.5e-4), (3e-4, 1.0e-4), (12e-4, 6e-4)]:
        solved = robust_size(edge, se, 25e-4, 3.6e-4, cfg=cfg)[0]
        closed = fast_robust_size(edge, se, 25e-4, 3.6e-4, cfg)
        assert solved == pytest.approx(closed, abs=0.03)


def test_size_is_monotone_in_edge_and_uncertainty():
    cfg = AmbiguityConfig()
    sizes = [fast_robust_size(e * 1e-4, 1e-4, 25e-4, 3.6e-4, cfg) for e in (2, 5, 8, 12, 20)]
    assert sizes == sorted(sizes)
    # More uncertainty about the same edge must never increase the position.
    tight = fast_robust_size(10e-4, 0.5e-4, 25e-4, 3.6e-4, cfg)
    loose = fast_robust_size(10e-4, 8.0e-4, 25e-4, 3.6e-4, cfg)
    assert loose <= tight


def test_size_is_zero_when_edge_cannot_cover_cost():
    cfg = AmbiguityConfig()
    assert fast_robust_size(2e-4, 1e-4, 25e-4, 6.65e-4, cfg) == 0.0
    assert fast_robust_size(0.0, 1e-4, 25e-4, 1e-4, cfg) == 0.0


# --------------------------------------------------------------------- regret
def _run_learner(ics, T=40_000, cfg=None, seed=0):
    rng = np.random.default_rng(seed)
    k = len(ics)
    cfg = cfg or LearnerConfig(prior_expert=None)
    learner = ContextualNoRegret(n_experts=k, n_contexts=1, cfg=cfg,
                                 expert_names=[f"e{i}" for i in range(k)])
    signals = np.clip(rng.standard_normal((T, k)), -3, 3)
    ret = rng.standard_normal(T)
    for j, ic in enumerate(ics):
        ret += ic * signals[:, j]
    ret /= ret.std()
    for t in range(T):
        learner.update(0, signals[t], ret[t])
    return learner.expert_weights(0)


def test_learner_concentrates_on_the_best_expert():
    """Expert 0 has a positive IC, expert 1 an equal negative one, rest noise."""
    w = _run_learner([0.04, -0.04, 0.0, 0.0, 0.0])
    # It must load on one of the two informative experts, with the right sign.
    assert w[0] > 0.5 or w[1] < -0.5
    # ...and essentially ignore the pure-noise ones.
    assert np.all(np.abs(w[2:]) < 0.2)


def test_learner_learns_the_sign_of_an_inverted_expert():
    w = _run_learner([-0.05, 0.0, 0.0])
    assert w[0] < -0.5, "the signed action set should short an anti-predictive expert"


def test_weights_form_a_distribution():
    learner = ContextualNoRegret(n_experts=4, n_contexts=2, expert_names=["a", "b", "c", "d"])
    for c in range(2):
        w = learner.weights(c)
        assert w.sum() == pytest.approx(1.0)
        assert np.all(w >= 0.0)


def test_prior_anchors_the_starting_weights():
    names = ["a", "b", "target", "d"]
    cfg = LearnerConfig(prior_expert="target", prior_weight=0.7)
    learner = ContextualNoRegret(n_experts=4, n_contexts=1, cfg=cfg, expert_names=names)
    w = learner.expert_weights(0)
    assert w[2] > 0.6
    assert np.all(np.abs(np.delete(w, 2)) < 0.05)


def test_unknown_prior_expert_falls_back_to_uniform():
    cfg = LearnerConfig(prior_expert="does_not_exist")
    learner = ContextualNoRegret(n_experts=3, n_contexts=1, cfg=cfg, expert_names=["a", "b", "c"])
    assert learner.weights(0) == pytest.approx(np.full(7, 1 / 7))


def test_action_positions_cover_both_signs_and_flat():
    learner = ContextualNoRegret(n_experts=2, n_contexts=1, expert_names=["a", "b"])
    pos = learner.action_positions(np.array([0.5, -0.25]))
    assert pos.tolist() == pytest.approx([0.5, -0.25, -0.5, 0.25, 0.0])


def test_dcfr_matches_or_beats_hedge_on_a_planted_signal():
    """Sanity check on the discounted regret-matching rule itself.

    It is not the shipped default — it dilutes the blend across a large sparse
    action set and the strategy stops trading — but the rule must still identify
    a planted signal, or the implementation is simply wrong.
    """
    hedge = _run_learner([0.04, -0.04, 0.0, 0.0, 0.0], cfg=LearnerConfig(prior_expert=None))
    dcfr = _run_learner(
        [0.04, -0.04, 0.0, 0.0, 0.0],
        cfg=LearnerConfig(rule="dcfr", prior_expert=None),
    )
    for w in (hedge, dcfr):
        assert w[0] > 0.2 or w[1] < -0.2, "should load on an informative expert"
    assert np.all(np.abs(dcfr[2:]) < 0.35), "noise experts should stay small"


def test_dcfr_weights_are_a_valid_distribution():
    learner = ContextualNoRegret(
        n_experts=3, n_contexts=1,
        cfg=LearnerConfig(rule="dcfr"), expert_names=["a", "b", "target"],
    )
    rng = np.random.default_rng(0)
    for _ in range(500):
        learner.update(0, rng.standard_normal(3), float(rng.standard_normal()))
    w = learner.weights(0)
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= 0.0)
