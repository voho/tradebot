"""Shared, read-only pre-registration and engine for the R-177 round (08-28).

DIRECTION, one sentence: give `kelly_regime_v4`'s already-existing 3-anchor
vote a SIGNED range (bear votes act on the negative side instead of
flattening to zero), on the grounds -- new to this ledger, see
`experiments/r177_direction.md` for the full argument -- that no round in
176 prior rounds has ever tested a directional short on this incumbent's
own vote, despite R-37 (08-19) having already MEASURED the all-bearish
vote state's own forward mu/sigma**2 at roughly -62%/yr, a number that
implies a non-trivial negative Kelly fraction that was floored at zero by
construction in every SIZE-axis variant built since, including R-37's own
novel branch.

The conservative branch symmetrizes the existing unsigned vote itself
(`frac_signed = 2*frac - 1`, no new parameters) times v4's own unmodified
`scale`; the novel branch reuses R-37's exact per-vote-state causal
mu_state/sigma_state**2 estimator (see `experiments/kelly_regime_v6_state_kelly.py`,
which this file's `state_kelly_stats` is a direct, undisguised port of) but
removes the `clip(kelly_f, 0.0, None)` floor R-37's own docstring names
explicitly ("a state whose noisy trailing estimate says 'negative expected
return' sizes to flat, exactly v4's own logic for its bear state, never to
a short") -- this round tests exactly the branch that line describes and
declines to run.

Full Step 1/Step 2 design (constraint attacked [SIZE primary], the
non-duplication argument against R-37/R-38/L-22/L-25/R-63/R-76/R-89/R-90,
simulability, named failure modes, and the pre-registered falsification
rule for both branches) is in `experiments/r177_direction.md`.

This module is DELIBERATELY neutral between the two branches: it exposes
`signed_vote_frac` (the conservative branch's sign-symmetric vote) and
`state_kelly_stats` (the novel branch's per-state, UNFLOORED causal
mu/sigma**2 estimator, identical to R-37's construction except for the
floor) as two functionals of the SAME underlying vote, mirroring
r175_shared.py's and r176_shared.py's convention of one shared engine
exposing two functionals neither branch may edit.

Configs evaluated by this file: 0 (shared infrastructure only; each
branch's own count is logged in its own module and summed in the ledger
entry, per R-163/R-168's convention).

Simplifications and risks disclosed up front (per ROUTINE.md's honesty
convention):
- Shorting is only mechanically possible on the `futures_5x` market
  (`MarketSpec.futures(allow_short=True)`; `MarketSpec.spot()` has
  `allow_short=False` and clips the lower bound to 0.0 in `broker.py`).
  Both branches therefore report spot as a LONG-ONLY control (identical
  to unmodified v4 on spot, since a signed vote can never go negative
  there) and reserve the actual test of this direction for futures_5x.
  This is a narrower falsification surface than most rounds get (one
  market, not two), named here rather than discovered mid-round.
- A short on 5x leverage stacks two of this project's own worst-measured
  risks in one place: R-33/R-57's exposure-artifact warning (a short is
  MORE exposure, not less, whenever the vote is wrong) and the standing
  fact that buy-and-hold itself liquidates on 5x futures in the January
  2017 crash. Both branches are pre-registered to report liquidation rate
  across the Monte Carlo stress windows explicitly, not just the point
  estimate, exactly because a single lucky point estimate would hide this.
- The per-state estimator (novel branch) inherits R-37's own already-named
  data-hunger failure mode: an unfloored bear-state mu/sigma**2 needs the
  SAME 2,000-observation minimum and multi-month halflife maturation R-37
  measured as fragile inside a 2-year inner-validation window; this round
  does not get a second chance to discover that problem, it is inherited
  verbatim.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY
STATES = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
STATE_LABELS = {0.0: "0/3", 1.0 / 3.0: "1/3", 2.0 / 3.0: "2/3", 1.0: "3/3"}


# ------------------------------------------------------------- shared vote


def unsigned_vote_frac(close: pd.Series, horizons: tuple[int, ...] = (20, 40, 80),
                        band: float = 0.01) -> np.ndarray:
    """`kelly_regime_v4`'s own 3-anchor latched vote, verbatim, in [0, 1].

    Copied unchanged from `kelly_regime.py`/`kelly_regime_v4.py` -- neither
    branch may alter detection, only the response to a given vote state.
    """
    votes = []
    for days in horizons:
        anchor = close.rolling(int(days * BARS_PER_DAY)).mean()
        v = pd.Series(
            np.where(close > anchor * (1.0 + band), 1.0,
                     np.where(close < anchor * (1.0 - band), 0.0, np.nan)),
            index=close.index,
        )
        votes.append(v.ffill().fillna(0.0))
    return (sum(votes) / len(votes)).to_numpy()


def signed_vote_frac(close: pd.Series, horizons: tuple[int, ...] = (20, 40, 80),
                      band: float = 0.01) -> np.ndarray:
    """Conservative branch primitive: the same vote, remapped to [-1, 1].

    `frac_signed = 2*frac - 1` is the unique affine map that sends v4's
    four discrete states {0, 1/3, 2/3, 1} to {-1, -1/3, +1/3, +1} -- the
    ONLY change from v4 is that the two bearish states now carry negative
    sign instead of collapsing to zero. No new parameter is introduced.
    """
    frac = unsigned_vote_frac(close, horizons=horizons, band=band)
    return 2.0 * frac - 1.0


# ------------------------------------------------ shared per-state Kelly stats


def state_kelly_stats(close: pd.Series, frac: np.ndarray, *, halflife_days: float = 90.0,
                       min_obs: int = 2000, stat_horizon_bars: int = 1,
                       floor_at_zero: bool = False) -> dict[str, np.ndarray]:
    """R-37's exact causal per-vote-state mu/sigma**2 estimator, floor optional.

    This is a direct port of `experiments/kelly_regime_v6_state_kelly.py`'s
    `prepare()` inner loop (see that file's module docstring for the
    9-step causality argument, reproduced in brief in
    `r177_direction.md`), generalized only by making the R-37 floor
    (`kelly_f = clip(kelly_f, 0.0, None)`) an explicit, default-preserving
    keyword rather than a hard-coded line -- `floor_at_zero=True`
    reproduces R-37's novel branch bit-for-bit; `floor_at_zero=False` (the
    novel branch's use here) is the untested alternative this round exists
    to measure.

    Returns arrays aligned to `close.index`: `mu`, `var`, `kelly_f`
    (unfloored `mu/var`, or floored if requested), `count` (occurrences of
    the active state seen so far, lagged identically to `mu`/`var`).
    """
    idx = close.index
    n = len(close)
    h = int(stat_horizon_bars)
    frac_s = pd.Series(frac, index=idx)

    r = np.log(close).diff(h)
    bucket_state = frac_s.shift(h)
    bucket_ret = r

    halflife = pd.Timedelta(days=halflife_days)
    mu_arr = np.full(n, np.nan)
    var_arr = np.full(n, np.nan)
    cnt_arr = np.zeros(n)
    bstate_np = bucket_state.to_numpy(dtype=float)

    for k in STATES:
        mask = np.isclose(bstate_np, k)
        occ = bucket_ret[mask].dropna()
        if len(occ) == 0:
            continue
        ew = occ.ewm(halflife=halflife, times=occ.index, min_periods=min_obs)
        mu_occ = ew.mean()
        mu2_occ = occ.pow(2).ewm(halflife=halflife, times=occ.index,
                                  min_periods=min_obs).mean()
        var_occ = (mu2_occ - mu_occ ** 2).clip(lower=0.0)
        cnt_occ = pd.Series(np.arange(1, len(occ) + 1), index=occ.index)

        mu_full = mu_occ.reindex(idx).ffill().shift(1)
        var_full = var_occ.reindex(idx).ffill().shift(1)
        cnt_full = cnt_occ.reindex(idx).ffill().shift(1).fillna(0.0)

        sel = np.isclose(frac, k)
        mu_arr[sel] = mu_full.to_numpy()[sel]
        var_arr[sel] = var_full.to_numpy()[sel]
        cnt_arr[sel] = cnt_full.to_numpy()[sel]

    with np.errstate(divide="ignore", invalid="ignore"):
        kelly_f = np.where(var_arr > 0, mu_arr / var_arr, np.nan)
    kelly_f = np.where(np.isfinite(kelly_f), kelly_f, 0.0)
    if floor_at_zero:
        kelly_f = np.clip(kelly_f, 0.0, None)

    return {"mu": mu_arr, "var": var_arr, "kelly_f": kelly_f, "count": cnt_arr}


# --------------------------------------------------------------------- self-test


def _self_test() -> None:
    print("r177_shared self-test")

    # 1. signed_vote_frac is an exact affine remap of the unsigned vote --
    #    both must be computed from the SAME anchors on the SAME data.
    rng = np.random.default_rng(0)
    n = 400_000
    idx = pd.date_range("2017-01-01", periods=n, freq="5min")
    price = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.0006, n)))
    close = pd.Series(price, index=idx)

    unsigned = unsigned_vote_frac(close)
    signed = signed_vote_frac(close)
    diff = np.nanmax(np.abs(signed - (2.0 * unsigned - 1.0)))
    print(f"  [1] signed == 2*unsigned-1 exactly: max|diff|={diff:.3e} "
          f"{'PASS' if diff < 1e-12 else 'FAIL'}")
    states_seen = sorted(set(np.round(unsigned[~np.isnan(unsigned)], 6)))
    signed_states = sorted(set(np.round(signed[~np.isnan(signed)], 6)))
    print(f"  [1b] unsigned states={states_seen} -> signed states={signed_states}")

    # 2. state_kelly_stats: floor_at_zero=True must reproduce R-37's own
    #    kelly_regime_v6_state_kelly.py bit-for-bit on the same synthetic
    #    series (regression guard against silently drifting from the port).
    stats_floored = state_kelly_stats(close, unsigned, halflife_days=90.0, min_obs=200,
                                       floor_at_zero=True)
    stats_unfloored = state_kelly_stats(close, unsigned, halflife_days=90.0, min_obs=200,
                                         floor_at_zero=False)
    only_floor_differs = np.array_equal(
        np.isnan(stats_floored["mu"]), np.isnan(stats_unfloored["mu"])
    ) and np.allclose(stats_floored["mu"], stats_unfloored["mu"], equal_nan=True) and \
        np.allclose(stats_floored["var"], stats_unfloored["var"], equal_nan=True)
    neg_kelly_exists = np.any(stats_unfloored["kelly_f"] < 0)
    floor_removes_negatives = not np.any(stats_floored["kelly_f"] < 0)
    print(f"  [2] mu/var identical between floored/unfloored: "
          f"{'PASS' if only_floor_differs else 'FAIL'}; "
          f"unfloored has negative kelly_f values: {neg_kelly_exists}; "
          f"floored has none: {floor_removes_negatives} "
          f"{'PASS' if (neg_kelly_exists and floor_removes_negatives) else 'FAIL'}")

    # 3. causal truncation probe, on REAL BTC data (R-172's own lesson:
    #    a same-day/same-state broadcast lookahead bug needs real, choppy
    #    price data to manifest -- the smooth synthetic generator above
    #    will not catch a mid-day state flip that only real data produces).
    from tradebot.data import load_dataset

    df, _ = load_dataset(ROOT / "data", "spot")
    real_close = df["close"].iloc[-300_000:].copy()
    cut = len(real_close) - 5_000

    up = real_close.copy()
    down = real_close.copy()
    up.iloc[cut:] *= 3.0
    down.iloc[cut:] /= 3.0

    ok = True
    for label, fn in (
        ("unsigned_vote_frac", lambda c: unsigned_vote_frac(c)),
        ("signed_vote_frac", lambda c: signed_vote_frac(c)),
    ):
        a = fn(up)[:cut]
        b = fn(down)[:cut]
        worst = float(np.nanmax(np.abs(a - b)))
        good = worst < 1e-9
        ok &= good
        print(f"  [3] {label}: max|difference before cut|={worst:.3e} "
              f"{'PASS' if good else 'FAIL'}")

    frac_up = unsigned_vote_frac(up)
    frac_down = unsigned_vote_frac(down)
    for floor in (True, False):
        su = state_kelly_stats(up, frac_up, halflife_days=90.0, min_obs=2000,
                                floor_at_zero=floor)
        sd = state_kelly_stats(down, frac_down, halflife_days=90.0, min_obs=2000,
                                floor_at_zero=floor)
        worst = 0.0
        for key in ("mu", "var", "kelly_f", "count"):
            a = su[key][:cut]
            b = sd[key][:cut]
            worst = max(worst, float(np.nanmax(np.abs(a - b))))
        good = worst < 1e-6
        ok &= good
        print(f"  [3] state_kelly_stats(floor_at_zero={floor}): "
              f"max|difference before cut|={worst:.3e} {'PASS' if good else 'FAIL'}")

    print(f"\nself-test {'PASSED' if ok and diff < 1e-12 else 'FAILED'}")


if __name__ == "__main__":
    _self_test()
