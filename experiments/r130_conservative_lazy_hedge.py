"""R-130 CONSERVATIVE branch: an Online-Lazy-Updates weight update for `hedge_experts`.

**Direction (recap, see `experiments/r130_shared.py` for the round-level
framing).** R-128 and R-129 each bolted a Kelly quadratic-cost no-trade
BAND (Constantinides 1986; Davis & Norman 1990) onto `hedge_experts`'s
OUTPUT, at three different application points (the blended signal, the
raw pre-blend experts, three bucket sub-blends) -- and all four
constructions failed. This round instead makes the Hedge weight-update
recursion itself (`logw`/`p` inside `HedgeExperts.prepare()`) cost-aware,
via a cost algebra outside the Kelly-quadratic-cost family entirely.

**Citation.** Das, P., Johnson, N., & Banerjee, A. (2013). "Online Lazy
Updates for Portfolio Selection with Transaction Costs." Proceedings of
the AAAI Conference on Artificial Intelligence, 27(1), 202-208. Fetched
and read in full this session (WebFetch on the AAAI CDN PDF,
https://cdn.aaai.org/ojs/8693/8693-13-12221-1-2-20201228.pdf).

**The paper's mechanism, in the terms it actually uses.** OLU frames
online portfolio selection with proportional transaction costs as
non-smooth online convex optimization: at each round the algorithm does
not just minimize its own loss `phi_t(p) = -log(p^T x_t)` (negative
log-wealth growth against that round's realized price relatives), it
minimizes `f_t(p) = phi_t(p) + gamma*||p - p_{t-1}||_1` -- the transaction
cost (`gamma`, a proportional fee rate) enters as an L1 penalty on the
weight CHANGE, added directly into the SAME per-round objective that
determines the next portfolio, not as a downstream filter applied to an
already-computed target (paper's eq. 3, eq. 6). Their `Online Lazy
Updates (OLU)` algorithm solves the resulting non-smooth problem each
round with ADMM (Boyd et al. 2011): it decouples the L1 term via an
auxiliary variable `z`, and the closed-form update for `z` is precisely
an entrywise **soft-threshold/shrinkage operator**, `z = S_{alpha/beta}(p
- p_t + u)` where `S_rho(v) = sign(v) * max(|v| - rho, 0)` -- the
classical proximal operator for an L1 norm. Because their formulation
regularizes the LEAST-SQUARES / L2-Euclidean distance to `p_t` (a
deliberate choice, contrasted explicitly in the paper's own text against
Helmbold et al. 1998's Exponentiated Gradient (EG), which instead uses
KL/relative-entropy distance -- i.e. multiplicative weights, exactly
`hedge_experts`'s own geometry), solving the full projected problem needs
ADMM's iterative primal-dual loop plus a simplex projection (Duchi,
Singer & Chandra 2008). Empirically (S&P500/NYSE, 1990-2010 and
1962-1984) they show total L1 turnover falls monotonically in the
penalty weight `alpha = eta*gamma`, trade COUNT does not always fall
monotonically with it (their own Figure 1(b) -- named here because it is
exactly this round's own failure-mode #1 below), and OLU with realistic
transaction costs (0.1%-0.5%) beats EG and buy-and-hold in realized
wealth.

**What is ported LITERALLY vs. ADAPTED (disclosed, per this repo's R-90
precedent for a paper whose exact formula does not map cleanly onto the
object being modified):**

- LITERAL: the core idea that the transaction-cost penalty is an L1 norm
  on the weight CHANGE, `alpha*||p_t - p_{t-1}||_1`, added into the SAME
  optimization that produces the weight update every round (paper's eq.
  3/6) -- not a band gating an already-computed `p` or `x` afterward.
- LITERAL: the shrinkage/soft-threshold operator `S_rho(v) = sign(v) *
  max(|v| - rho, 0)` that solves the L1 sub-problem -- this is exactly
  the paper's own ADMM `z`-update (Algorithm 1), reused verbatim as the
  mechanism that turns the penalty into a concrete weight adjustment.
- LITERAL: the calibration point, `alpha = fee_rate` (the paper's own
  `gamma`, "a fixed percentage of the amount of transactions" charged on
  L1 portfolio-weight turnover) mapped directly onto `hedge_experts`'s
  own `fee_rate` parameter -- same units (a proportional cost per unit of
  L1 weight-turnover on a probability-simplex vector), no fitting. The
  paper's own `alpha = eta*gamma` relation is available too (both `eta`
  and `fee_rate` already exist on `HedgeExperts`) but is not used here:
  `hedge_experts`'s `eta` (0.05) is a per-bar Hedge learning-rate scaling
  a POST-hoc-clipped payoff `g` in [-1, 1], not a "weight on
  log-wealth-gain" comparable in role to the paper's `eta` (which scales
  an UNCLIPPED, un-bounded log-wealth term) -- multiplying `fee_rate` by
  0.05 before it ever compares to weight moves of order 1e-3 to 1e-1
  would make the penalty numerically inert for any bar this project's
  data actually produces test-checked below), so `alpha = fee_rate`
  itself (not `eta*fee_rate`) is used as the "1x" calibration, and the B3
  sweep varies THIS multiplier, disclosed as a deviation from the paper's
  own two-parameter form for exactly this reason.
- ADAPTED (the substantive change, and the reason this is a port of the
  CONCEPT rather than a reproduction of the exact formula): OLU's own
  proximal geometry is L2-Euclidean (`(1/2)||p-p_t||_2^2`), solved by
  iterative ADMM with an explicit simplex projection, because that is
  what a direct L1-penalized Euclidean least-squares problem over the
  simplex requires. `hedge_experts`'s own Hedge/multiplicative-weights
  recursion is NOT Euclidean -- its native proximal geometry is KL/
  relative-entropy (the paper's own text names this explicitly:
  "analogous to... EG... We use ||.||_2^2 ... instead of the relative
  entropy in EG"), and its softmax step already computes the exact
  closed-form solution to the entropy-regularized linear problem (the
  discounted-multiplicative-weights update the strategy already runs,
  left completely unmodified up to this point). Rather than discard that
  native geometry and switch to the paper's L2/ADMM machinery (a much
  larger, harder-to-audit rewrite of `prepare()`'s loop, and a mechanism
  the paper itself frames as EG's ALTERNATIVE, not its extension), this
  port keeps Hedge's entropic step exactly as-is and layers the paper's
  L1 lazy-update penalty ON TOP of its output, applying the same
  shrinkage operator `S_alpha` directly to `w_raw - p_{t-1}` (the
  softmax's proposed weight vector minus last bar's ACTUAL, already-lazy
  weight vector) instead of solving the L2-ADMM sub-problem the paper
  derives for a from-scratch L2 base. This is a one-shot (non-iterative)
  shrink-then-renormalize step, not the paper's multi-iteration ADMM loop
  -- a first-order approximation to jointly solving (KL proximal + L1
  penalty) rather than an exact solve, disclosed because ADMM would need
  either abandoning Hedge's own update or a second nested optimization
  inside every one of >600k per-bar iterations, out of proportion to what
  a "port the mechanism, not the microcode" adaptation calls for.
  Projection to the simplex after shrinkage is also simplified: paper
  uses Duchi et al. (2008)'s exact L1-ball projection inside ADMM; this
  port clips negative entries to zero and renormalizes to sum to one --
  correct (yields a valid probability vector) but not the identical
  algorithm, disclosed rather than silently assumed equivalent.
- ADAPTED (cold start): `p_{t-1}` is undefined before the first live
  iteration. It is initialized to the uniform prior `1/N`, matching
  exactly what `logw = zeros(N)` already encodes at bar 2 in the
  unmodified `HedgeExperts.prepare()` -- i.e. the FIRST bar's shrinkage
  target is the same uniform starting point the baseline's own softmax
  would have produced with no history, not an arbitrary choice. This is a
  bounded, one-bar cold-start artifact out of >600,000 inner-train bars,
  disclosed rather than treated as free of any assumption (same
  convention R-129 used for its own `self._held` cold start).

**Mechanism, one sentence.** Every bar, after `hedge_experts`'s own
softmax step proposes a new expert-confidence vector `w_raw`, the
proposed CHANGE `w_raw - p_{t-1}` is soft-thresholded by
`alpha = lazy_alpha_mult * fee_rate` before being added back onto
`p_{t-1}` and renormalized, so any expert whose confidence would move by
less than `alpha` this bar does not move at all, and one that clears the
threshold moves by `alpha` less than the softmax proposed -- making the
Hedge weight vector `p` itself, not just the downstream blended signal
`x = p @ a`, cost-aware.

**Why it should reduce cost without destroying the signal.** Fee-scale
noise in the softmax weights (small day-to-day reshuffling of expert
confidence driven by short-run payoff noise rather than a genuine
change in which expert is winning) is exactly what an L1 shrinkage of
size `fee_rate` should absorb -- a change smaller than what it would
cost to act on is, by the same discounted-cash-flow logic the whole
Kelly-band literature already uses, not worth having moved for in the
first place. Because the shrinkage acts on `p`, not on `a` or `x`
directly, and `x = p @ a` is linear in `p`, a genuine, large,
sustained shift in which expert is winning still moves `p` (each bar's
shrinkage removes only `alpha` of the move, not all of it, so a
persistent gap keeps closing bar over bar) and therefore still moves
`x` -- the mechanism damps NOISE-scale weight churn, not real regime
shifts, by construction (not proof; this is exactly what B3/B4 below
test empirically rather than assume).

**What would make this fail (named now, before evaluation code ran):**

1. **The sharpest named risk, carried over verbatim from R-129's own
   failure-mode #1 (this branch's paper's own Figure 1(b) makes the same
   point independently):** shrinking `p`'s bar-to-bar CHANGE does not
   guarantee `x = p @ a` itself re-targets less often -- the paper's own
   S&P500 experiment found total trade COUNT (L0 norm) does not always
   fall monotonically with the penalty weight even though total
   transacted AMOUNT (L1 norm) does. `hedge_experts`'s existing
   `hysteresis=0.05` already gates `x`'s re-targets; this mechanism could
   leave the realized trade count on `x` close to baseline's regardless
   of how much `p`'s own churn is damped, which would make it cost-aware
   in name only. Reported explicitly below (both `p`'s own realized L1
   turnover and the ENGINE's actual `num_trades`, not just one proxy for
   the other).
2. If `alpha` is large enough to matter, it does not just remove noise:
   it also slows how fast `p` can respond to a genuine regime change
   (every bar it can move by at most `alpha` beyond what shrinkage
   removes), which is structurally the same LAG mechanism this project
   has now measured failing in every one of 11 regime-timing mechanisms
   tried on `kelly_regime_v4` -- if `hedge_experts`'s edge on spot comes
   substantially from timely regime response, a large-enough `alpha`
   could give that back for a turnover saving that does not compensate.
   B3's 4x sweep is the direct test of whether this shows up as a
   plateau or a reversal.
3. Six-plus independent prior constructions on this project (R-109,
   R-113, R-115-conservative, R-125-conservative, R-126 both branches,
   R-128 conservative weakly) have passed a BTC promotion gate and
   INVERTED sign on ETH. This is a real, repeated prior, not a formality;
   B4 is the test built to catch exactly this, and a BTC-only pass here
   is treated as weak evidence regardless of its own size.
4. `alpha = fee_rate` is a single-scalar calibration applied identically
   to all ten experts' weight components, even though the experts have
   very different native timescales (R-129's own `EXPERT_HORIZON_DAYS`
   table) -- a fixed per-bar shrinkage could be structurally too tight
   for the fast (sub-daily) experts' natural churn and too loose for the
   slow (multi-day) ones, since this branch (unlike R-129) intentionally
   does NOT differentiate by expert timescale, in order to isolate the
   OLU mechanism itself from R-129's already-tested per-expert
   partitioning question. If B3's sweep shows the sign flipping rather
   than plateauing, an uneven per-expert fit (not the OLU mechanism
   itself) is the first alternative explanation to rule out before
   concluding the mechanism failed outright.

No bar at or after `OOS_START = 2023-01-01` is read anywhere in this
file.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.strategies.hedge_experts import HedgeExperts  # noqa: E402

import r130_shared as sh  # noqa: E402


class LazyHedgeExperts(HedgeExperts):
    """`hedge_experts` with an OLU-style (Das/Johnson/Banerjee 2013) lazy
    weight update layered onto the Hedge softmax step itself -- see this
    module's own docstring for the mechanism and what was adapted.

    NOT registered (`@register` deliberately omitted): this is an
    experiment, not a strategy this project trades.
    """

    def __init__(self, eta: float = 0.05, fixed_share: float = 1e-4,
                 hysteresis: float = 0.05, fee_rate: float = 0.0005,
                 lazy_alpha_mult: float = 1.0) -> None:
        super().__init__(eta=eta, fixed_share=fixed_share,
                          hysteresis=hysteresis, fee_rate=fee_rate)
        self.lazy_alpha_mult = lazy_alpha_mult
        self.lazy_alpha = lazy_alpha_mult * fee_rate
        # Diagnostics filled by prepare(), read by the eval script below.
        self.diag_l1_turnover_raw = 0.0        # sum |w_raw - p_prev| (pre-shrink)
        self.diag_l1_turnover_realized = 0.0   # sum |p - p_prev| (post-shrink)
        self.diag_p_retarget_bars = 0          # bars where any p_i actually moved
        self.diag_bars_scored = 0

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        r = np.log(df["close"]).diff()
        sig1 = r.ewm(span=288, min_periods=250).std()
        a = self._experts(df, r, sig1)  # (n, N) -- HedgeExperts._experts, unmodified
        r_a = r.to_numpy()
        sig_a = sig1.shift(1).to_numpy()

        n, num = a.shape
        target = np.zeros(n)
        logw = np.zeros(num)
        p_prev = np.full(num, 1.0 / num)  # cold-start convention, see docstring
        pos = 0.0
        alpha = self.lazy_alpha

        l1_raw = 0.0
        l1_realized = 0.0
        retarget_bars = 0
        scored = 0

        for i in range(2, n):
            s = sig_a[i]
            if not np.isfinite(s) or s <= 0:
                target[i] = pos
                continue
            z_t = min(max(r_a[i] / (3.0 * s), -1.0), 1.0)
            fee_n = min(self.fee_rate / (3.0 * s), 0.25)
            g = np.clip(a[i - 1] * z_t - fee_n * np.abs(a[i - 1] - a[i - 2]), -1.0, 1.0)
            logw += self.eta * g
            logw -= logw.max()
            w_raw = np.exp(logw)
            w_raw /= w_raw.sum()

            # --- OLU-style lazy weight update (Das, Johnson & Banerjee 2013) ---
            delta = w_raw - p_prev
            shrunk = np.sign(delta) * np.maximum(np.abs(delta) - alpha, 0.0)
            p = p_prev + shrunk
            p = np.maximum(p, 0.0)
            s_sum = p.sum()
            p = p / s_sum if s_sum > 0 else w_raw
            p = (1.0 - self.fixed_share) * p + self.fixed_share / num

            scored += 1
            l1_raw += float(np.abs(w_raw - p_prev).sum())
            realized_move = float(np.abs(p - p_prev).sum())
            l1_realized += realized_move
            if realized_move > 1e-12:
                retarget_bars += 1

            p_prev = p
            logw = np.log(np.clip(p, 1e-300, None))

            x = float(p @ a[i])
            if abs(x - pos) > self.hysteresis or (x > 0) != (pos > 0) or (x < 0) != (pos < 0):
                pos = x
            target[i] = pos

        self.diag_l1_turnover_raw = l1_raw
        self.diag_l1_turnover_realized = l1_realized
        self.diag_p_retarget_bars = retarget_bars
        self.diag_bars_scored = scored

        df["target"] = target
        return df


# ======================================================================
# Evaluation script (falsification battery). Trains/evaluates ONLY on
# INNER_TRAIN_START .. INNER_VAL_END; never reads OOS_START or later.
# ======================================================================

CONFIGS_EVALUATED = 0  # bumped by every run_period call below, printed at the end


def _run(strat, df, market, start, end, label):
    global CONFIGS_EVALUATED
    CONFIGS_EVALUATED += 1
    m, res = sh.run_strategy(strat, df, market, start, end, label)
    return m, res


def _fmt_ci(pr, name):
    lo, hi = pr.diff.lo, pr.diff.hi
    return (f"{name}: cand_sharpe={pr.stat_a:.3f} base_sharpe={pr.stat_b:.3f} "
            f"d_sharpe={pr.stat_a - pr.stat_b:+.3f} CI95=[{lo:+.3f},{hi:+.3f}] "
            f"p(diff>0)={pr.p_positive:.3f}")


def main() -> None:
    t_start = time.time()
    print("=" * 78)
    print("R-130 CONSERVATIVE: OLU-style lazy Hedge weight update vs hedge_experts")
    print("=" * 78)

    btc, btc_label = sh.load_btc_train("spot")
    eth = sh.load_eth_train()

    candidate = LazyHedgeExperts(lazy_alpha_mult=1.0)

    # ------------------------------------------------------------------
    # B1: candidate vs frozen hedge_experts, BTC spot + futures, full
    # inner-train-through-val AND inner-validation alone.
    # ------------------------------------------------------------------
    print("\n--- B1: BTC, candidate (alpha_mult=1.0) vs frozen hedge_experts ---")
    b1 = {}
    for mkt_name, market in (("spot", sh.SPOT), ("futures", sh.FUTURES)):
        for period_name, (pstart, pend) in (
            ("full", (sh.INNER_TRAIN_START, sh.INNER_VAL_END)),
            ("val_only", (sh.INNER_VAL_START, sh.INNER_VAL_END)),
        ):
            m_base, res_base = _run(HedgeExperts(), btc, market, pstart, pend, btc_label)
            m_cand, res_cand = _run(LazyHedgeExperts(lazy_alpha_mult=1.0), btc, market,
                                     pstart, pend, btc_label)
            pr = sh.sharpe_diff(res_cand, res_base)
            key = f"btc_{mkt_name}_{period_name}"
            b1[key] = (m_base, m_cand, pr)
            print(f"[{key}] base: trades={m_base.num_trades} final={m_base.final_balance:.1f} "
                  f"sharpe={m_base.sharpe:.3f} | cand: trades={m_cand.num_trades} "
                  f"final={m_cand.final_balance:.1f} sharpe={m_cand.sharpe:.3f}")
            print("   " + _fmt_ci(pr, key))

    # ------------------------------------------------------------------
    # B3: plateau check. Sweep lazy_alpha_mult in {0.5, 1, 2, 4} on the
    # primary decision cell (BTC spot, inner-validation).
    # ------------------------------------------------------------------
    print("\n--- B3: plateau sweep, BTC spot inner-validation, "
          "lazy_alpha_mult in {0.5, 1, 2, 4} ---")
    b3 = {}
    # b1 stored only metrics for the val_only baseline, not the result
    # object sharpe_diff needs; re-run once (cheap relative to the sweep)
    # so B3 has its own paired baseline result object.
    m_base_val, res_base_val = _run(HedgeExperts(), btc, sh.SPOT,
                                     sh.INNER_VAL_START, sh.INNER_VAL_END, btc_label)
    for mult in sh.B3_MULTIPLIERS:
        m_cand, res_cand = _run(LazyHedgeExperts(lazy_alpha_mult=mult), btc, sh.SPOT,
                                 sh.INNER_VAL_START, sh.INNER_VAL_END, btc_label)
        pr = sh.sharpe_diff(res_cand, res_base_val)
        b3[mult] = (m_cand, pr)
        print(f"[mult={mult:>4}] trades={m_cand.num_trades} sharpe={m_cand.sharpe:.3f} "
              f"d_sharpe={pr.stat_a - pr.stat_b:+.3f} CI95=[{pr.diff.lo:+.3f},{pr.diff.hi:+.3f}]")

    # ------------------------------------------------------------------
    # B4: pre-registered falsification test. Does the inner-validation
    # sign replicate on ETH spot?
    # ------------------------------------------------------------------
    print("\n--- B4: ETH spot, candidate (alpha_mult=1.0) vs frozen hedge_experts ---")
    b4 = {}
    for period_name, (pstart, pend) in (
        ("full", (None, sh.INNER_VAL_END)),
        ("val_only", (sh.INNER_VAL_START, sh.INNER_VAL_END)),
    ):
        m_base, res_base = _run(HedgeExperts(), eth, sh.SPOT, pstart, pend, "ETH spot")
        m_cand, res_cand = _run(LazyHedgeExperts(lazy_alpha_mult=1.0), eth, sh.SPOT,
                                 pstart, pend, "ETH spot")
        pr = sh.sharpe_diff(res_cand, res_base)
        key = f"eth_spot_{period_name}"
        b4[key] = (m_base, m_cand, pr)
        print(f"[{key}] base: trades={m_base.num_trades} sharpe={m_base.sharpe:.3f} | "
              f"cand: trades={m_cand.num_trades} sharpe={m_cand.sharpe:.3f}")
        print("   " + _fmt_ci(pr, key))

    # ------------------------------------------------------------------
    # B5: 0.40% taker fee tier, BTC spot + futures, inner-validation.
    # ------------------------------------------------------------------
    print("\n--- B5: 0.40% fee tier, BTC, inner-validation ---")
    b5 = {}
    for mkt_name, market in (("spot", sh.SPOT_HIGH_FEE), ("futures", sh.FUTURES_HIGH_FEE)):
        m_base, res_base = _run(HedgeExperts(), btc, market,
                                 sh.INNER_VAL_START, sh.INNER_VAL_END, btc_label)
        m_cand, res_cand = _run(LazyHedgeExperts(lazy_alpha_mult=1.0), btc, market,
                                 sh.INNER_VAL_START, sh.INNER_VAL_END, btc_label)
        pr = sh.sharpe_diff(res_cand, res_base)
        key = f"btc_{mkt_name}_highfee"
        b5[key] = (m_base, m_cand, pr)
        print(f"[{key}] base: trades={m_base.num_trades} sharpe={m_base.sharpe:.3f} | "
              f"cand: trades={m_cand.num_trades} sharpe={m_cand.sharpe:.3f}")
        print("   " + _fmt_ci(pr, key))

    # ------------------------------------------------------------------
    # Causal-truncation probe (mandatory hygiene, not a promotion gate).
    # ------------------------------------------------------------------
    print("\n--- Causal-truncation probe (candidate strategy) ---")
    cand_probe = LazyHedgeExperts(lazy_alpha_mult=1.0)
    m_full, _ = _run(cand_probe, btc, sh.SPOT, sh.INNER_TRAIN_START, sh.INNER_TRAIN_END, btc_label)
    btc_trunc = btc.loc[:sh.INNER_VAL_END]
    cand_probe2 = LazyHedgeExperts(lazy_alpha_mult=1.0)
    m_trunc, _ = _run(cand_probe2, btc_trunc, sh.SPOT, sh.INNER_TRAIN_START,
                       sh.INNER_TRAIN_END, btc_label)
    trunc_ok = np.isclose(m_full.final_balance, m_trunc.final_balance, rtol=1e-9)
    print(f"causal truncation probe: {'PASS' if trunc_ok else 'FAIL'} "
          f"({m_full.final_balance} vs {m_trunc.final_balance})")

    # ------------------------------------------------------------------
    # Weight-level diagnostic: does the mechanism cut p's own churn at all?
    # ------------------------------------------------------------------
    print("\n--- Weight-level diagnostic (candidate, alpha_mult=1.0, BTC spot full) ---")
    diag = LazyHedgeExperts(lazy_alpha_mult=1.0)
    _run(diag, btc, sh.SPOT, sh.INNER_TRAIN_START, sh.INNER_VAL_END, btc_label)
    print(f"bars scored: {diag.diag_bars_scored}")
    print(f"p L1 turnover, pre-shrink (raw softmax proposal): {diag.diag_l1_turnover_raw:.1f}")
    print(f"p L1 turnover, post-shrink (realized):             {diag.diag_l1_turnover_realized:.1f}")
    print(f"bars where p actually moved: {diag.diag_p_retarget_bars} / {diag.diag_bars_scored} "
          f"({100.0*diag.diag_p_retarget_bars/max(diag.diag_bars_scored,1):.1f}%)")

    print(f"\nTotal configurations evaluated (run_period calls): {CONFIGS_EVALUATED}")
    print(f"Wall time: {time.time() - t_start:.1f}s")

    assert trunc_ok, "causal truncation probe FAILED -- candidate reads ahead of its own start"


if __name__ == "__main__":
    main()
