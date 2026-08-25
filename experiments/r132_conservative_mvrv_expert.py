"""R-132 CONSERVATIVE branch: append ONE MVRV mean-reversion expert to
`hedge_experts`'s existing ten-expert panel.

See `experiments/r132_shared.py`'s module docstring for the full
pre-registration (direction, literature, mechanism, non-duplicate check,
named failure modes, falsification test, decision rule). This file only
adds the CONSERVATIVE branch's implementation and results.

MECHANISM (frozen; every number below was chosen BEFORE any performance
number was read):

  ``ConservativeMVRVExpert(HedgeExperts)`` overrides ONLY ``_experts()``.
  It calls ``super()._experts(df, r, sig1)`` to get the existing
  bit-for-bit-unmodified 10-column expert matrix, then appends an 11th
  column and returns the concatenation. ``prepare()`` and ``on_bar()``
  are inherited untouched -- the Hedge weight-update loop already derives
  ``num`` from ``a.shape[1]``, so it generalizes to 11 experts with no
  other code change. `eta=0.05`, `fixed_share=1e-4`, `hysteresis=0.05`,
  `fee_rate=0.0005` (`HedgeExperts`'s own defaults) are held fixed.

  The 11th column, the MVRV expert:

  1. Load ``tradebot.data.load_mvrv_ratio(data_dir, asset=...)`` --
     CoinMetrics daily MVRV (market cap / realized cap), the same file
     R-74 used.
  2. Compute a trailing rolling z-score of the RAW daily MVRV series,
     entirely on the daily cadence (not the 5m bar cadence) before any
     alignment onto bars:

         z_t = (mvrv_t - rolling_mean(mvrv, W)_t) / rolling_std(mvrv, W)_t

     with ``min_periods = W`` (no partial-window z-scores; the column is
     NaN, and the expert votes 0, until a full window of trailing daily
     history exists).

     WINDOW ``W = 365`` calendar days (one year), chosen a priori and
     frozen before any backtest was run, on two grounds stated now:
       (a) MVRV / realized-cap valuation cycles are documented in the
           literature (Mahmudov & Puell 2018, the paper that introduced
           the ratio; Puell & Mahmudov's later "MVRV Z-Score") as
           multi-month-to-annual in their local swings, well below the
           ~4-year halving supercycle but well above weekly noise -- one
           year is the shortest window that reliably spans at least one
           full local over/under-valuation swing without needing several
           years of warmup history the dataset does not have for ETH.
       (b) One year of daily observations (365 points) is also enough
           for a rolling std to be a stable denominator without being so
           long (multi-year) that it starts averaging across genuinely
           different market regimes (2018 bear vs. 2021 bull), which
           would blunt the z-score exactly when a regime shift is the
           thing worth detecting.
     BTC MVRV data starts 2016-01-01 (a full year before
     `INNER_TRAIN_START = 2017-01-01`), so the 365-day window is already
     full by the first inner-train bar -- no NaN warmup cost inside the
     train/inner-val period. ETH MVRV starts 2018-01-01, so the window is
     full by ~2019-01-01, well before `INNER_VAL_START = 2021-01-01`
     used for the B4 falsification run.

  3. Map z to a vote via ``vote = -tanh(z / k)`` with ``k = 2.0``, chosen
     a priori (before any performance number) on the standard convention
     that +-2 standard deviations is the threshold a trailing-window
     z-score conventionally treats as "notably stretched" (Bollinger's
     own 2-sigma banding, Bollinger 1992, is the canonical instance of
     this exact convention, just applied here to a valuation ratio
     instead of price-vs-MA). At |z|=2 the vote is +-tanh(1) = +-0.762;
     it saturates toward +-1 only for even more extreme deviations.

     SIGN CONVENTION (frozen before any result): MVRV is a valuation
     ratio. A large POSITIVE z means today's MVRV is far ABOVE its own
     trailing-year mean, i.e. richly valued relative to realized cost
     basis -- the vote should lean SHORT/flat, hence the leading minus
     sign. A large NEGATIVE z means MVRV is far BELOW its trailing mean,
     i.e. cheap -- the vote leans LONG. This is a mean-reversion-on-
     valuation construction, deliberately different in kind from the
     panel's existing momentum/trend experts.

  4. Causal alignment onto the 5-minute bar grid uses
     ``tradebot.data.align_mvrv_causal`` applied to the DAILY z-score
     series (not the raw mvrv column) -- the same day+1 shift-then-ffill
     causal alignment R-74 used for the raw ratio, just applied one step
     downstream to the derived z column. A bar at time T can only see
     the z-score computed from days that closed strictly before T's own
     day. Bars before the first full window get NaN, mapped to a vote of
     0.0 (flat), exactly like the base panel's own NaN-warmup-as-flat
     convention (``np.nan_to_num(a, nan=0.0)`` in `HedgeExperts._experts`).

  Frozen parameters: ``MVRV_WINDOW_DAYS = 365``, ``MVRV_K = 2.0``.
  B3 sweeps ``MVRV_WINDOW_DAYS`` by `r132_shared.B3_MULTIPLIERS`
  (0.5x/1x/2x/4x = 183/365/730/1460 days), holding `k` fixed, since the
  window is the construction's primary a priori-chosen lever.

NON-DUPLICATE / CITATION: see `r132_shared.py`'s "Not a duplicate of"
section (R-74 used the same file as a standalone confirming-vote GATE on
`kelly_regime_v4`, NEGATIVE; here it is one Hedge-weighted VOTE among
eleven on a different object, never gating anything directly).

FALSIFICATION TEST (pre-registered, verbatim from r132_shared.py): B4 --
does the candidate's `d_sharpe` sign (candidate vs `hedge_experts`,
inner-validation) replicate on ETH spot?

DECISION RULE (pre-registered, verbatim from r132_shared.py): PROMOTE-
candidate only if causal-truncation probe AND A2 AND B1 (both markets,
both periods) AND B3 (plateau majority) AND B4 (sign replicates) AND B5
(no sign flip at 0.40% fee) all pass.

No bar at or after `OOS_START = 2023-01-01` is read anywhere below;
`_assert_no_holdout` (imported from r132_shared) is called on every frame
used.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.data import align_mvrv_causal, load_mvrv_ratio  # noqa: E402
from tradebot.strategies.hedge_experts import HedgeExperts  # noqa: E402

from r132_shared import (  # noqa: E402
    B3_MULTIPLIERS,
    FUTURES,
    FUTURES_HIGH_FEE,
    INNER_TRAIN_END,
    INNER_TRAIN_START,
    INNER_VAL_END,
    INNER_VAL_START,
    OOS_START,
    SPOT,
    SPOT_HIGH_FEE,
    a2_non_inertness,
    load_btc_train,
    load_eth_train,
    log_growth_diff,
    run_baseline,
    run_strategy,
    sharpe_diff,
)


def _assert_no_holdout(df: pd.DataFrame) -> None:
    last = df.index[-1]
    assert last < pd.Timestamp(OOS_START, tz=last.tz), (
        f"holdout breach: frame's last bar {last} is at/after {OOS_START}")


class ConservativeMVRVExpert(HedgeExperts):
    """`hedge_experts` plus one MVRV-valuation mean-reversion vote (R-132 CONSERVATIVE).

    Ten inherited experts, bit-for-bit unchanged, plus an 11th: a causal
    trailing-365-day z-score of daily MVRV mapped through -tanh(z/2.0).
    See module docstring for the full frozen construction.
    """

    name = "conservative_mvrv_expert"  # not @register-decorated: experimental only

    MVRV_WINDOW_DAYS = 365
    MVRV_K = 2.0

    def __init__(self, *args, mvrv_window_days: int = 365, mvrv_k: float = 2.0,
                 mvrv_asset: str = "BTC", data_dir: Path | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mvrv_window_days = mvrv_window_days
        self.mvrv_k = mvrv_k
        self.mvrv_asset = mvrv_asset
        self.data_dir = Path(data_dir) if data_dir is not None else ROOT / "data"

    def _mvrv_vote(self, df: pd.DataFrame) -> pd.Series:
        mvrv = load_mvrv_ratio(self.data_dir, asset=self.mvrv_asset)
        assert mvrv is not None, f"MVRV data missing for {self.mvrv_asset}"
        # Guard: never let a same-or-after-OOS row into the z-score computation.
        mvrv = mvrv.loc[mvrv.index < pd.Timestamp(OOS_START, tz=mvrv.index.tz)]
        w = self.mvrv_window_days
        roll_mean = mvrv["mvrv"].rolling(w, min_periods=w).mean()
        roll_std = mvrv["mvrv"].rolling(w, min_periods=w).std()
        z = (mvrv["mvrv"] - roll_mean) / roll_std
        z_df = z.to_frame("mvrv_z")
        aligned = align_mvrv_causal(z_df, df)
        vote = -np.tanh(aligned["mvrv_z"] / self.mvrv_k)
        return vote.fillna(0.0)

    def _experts(self, df: pd.DataFrame, r: pd.Series, sig1: pd.Series) -> np.ndarray:
        base = super()._experts(df, r, sig1)  # (n, 10), bit-for-bit unchanged
        vote = self._mvrv_vote(df).to_numpy(dtype=np.float64)
        vote = np.nan_to_num(vote, nan=0.0)
        return np.column_stack([base, vote])  # (n, 11)

    # -- diagnostic-only copy of prepare()'s loop body, also returning the
    # new (last-column) expert's own Hedge weight share p_i at every bar,
    # for A2. Does not replace prepare()/on_bar(), which stay inherited.
    def _prepare_with_new_expert_weight(self, df: pd.DataFrame):
        r = np.log(df["close"]).diff()
        sig1 = r.ewm(span=288, min_periods=250).std()
        a = self._experts(df, r, sig1)
        r_a = r.to_numpy()
        sig_a = sig1.shift(1).to_numpy()

        n, num = a.shape
        target = np.zeros(n)
        p_new = np.full(n, 1.0 / num)
        logw = np.zeros(num)
        pos = 0.0
        for i in range(2, n):
            s = sig_a[i]
            if not np.isfinite(s) or s <= 0:
                target[i] = pos
                p_new[i] = p_new[i - 1]
                continue
            z_t = min(max(r_a[i] / (3.0 * s), -1.0), 1.0)
            fee_n = min(self.fee_rate / (3.0 * s), 0.25)
            g = np.clip(a[i - 1] * z_t - fee_n * np.abs(a[i - 1] - a[i - 2]), -1.0, 1.0)
            logw += self.eta * g
            logw -= logw.max()
            p = np.exp(logw)
            p /= p.sum()
            p = (1.0 - self.fixed_share) * p + self.fixed_share / num
            logw = np.log(p)
            x = float(p @ a[i])
            if abs(x - pos) > self.hysteresis or (x > 0) != (pos > 0) or (x < 0) != (pos < 0):
                pos = x
            target[i] = pos
            p_new[i] = p[-1]
        return target, p_new


N_CONFIGS_EVALUATED = 0


def _count(n=1):
    global N_CONFIGS_EVALUATED
    N_CONFIGS_EVALUATED += n


# ----------------------------------------------------------------------
# 1. Causal-truncation probe
# ----------------------------------------------------------------------

def causal_truncation_probe() -> bool:
    # Full frame extends well past the tested window's end (through
    # INNER_VAL_END); the truncated frame stops exactly at the tested
    # window's end. If prepare() peeks ahead (a full-series stat, a
    # non-shifted rolling window, etc.) the two runs will disagree.
    df, label = load_btc_train("spot")
    strat = ConservativeMVRVExpert()
    m_full, _ = run_strategy(strat, df, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label)
    _count()

    df_trunc = df.loc[:INNER_TRAIN_END]
    strat2 = ConservativeMVRVExpert()
    m_trunc, _ = run_strategy(strat2, df_trunc, SPOT, INNER_TRAIN_START, INNER_TRAIN_END, label)
    _count()

    ok = np.isclose(m_full.final_balance, m_trunc.final_balance, rtol=1e-9)
    print(f"[causal probe] {'PASS' if ok else 'FAIL'} "
          f"(full={m_full.final_balance:.6f} vs trunc={m_trunc.final_balance:.6f})")
    return bool(ok)


# ----------------------------------------------------------------------
# 2. A2 non-inertness
# ----------------------------------------------------------------------

def a2_gate() -> dict:
    from tradebot.window import prefix_bars

    df, label = load_btc_train("spot")
    _assert_no_holdout(df)
    strat = ConservativeMVRVExpert()
    # Replicate run_period's own windowing exactly (full-inner cell: warmup
    # prefix before INNER_TRAIN_START, through INNER_VAL_END) so the weight
    # trace reflects the same computation B1's "full_inner" cell runs.
    lo = int(df.index.searchsorted(INNER_TRAIN_START))
    hi = int(df.index.searchsorted(INNER_VAL_END, side="right"))
    prefix = prefix_bars(df, lo, strat.warmup)
    frame = df.iloc[lo - prefix: hi]
    target, p_new = strat._prepare_with_new_expert_weight(frame)
    _count()
    val_mask = (frame.index >= pd.Timestamp(INNER_VAL_START, tz=frame.index.tz)) & \
               (frame.index <= pd.Timestamp(INNER_VAL_END, tz=frame.index.tz))
    p_val = p_new[val_mask]
    result = a2_non_inertness(p_val, num_experts=11)
    print(f"[A2] frac_bars_above_2x_uniform={result['frac_bars_above_2x_uniform']:.4f} "
          f"(uniform=1/11={1/11:.4f}, threshold=2x uniform={2/11:.4f}) -> "
          f"{'PASS' if result['pass'] else 'FAIL'}")
    return result


# ----------------------------------------------------------------------
# 3. B1 paired-bootstrap Sharpe + log-growth, 2 markets x 2 periods
# ----------------------------------------------------------------------

def b1_cell(df, market, start, end, label):
    strat = ConservativeMVRVExpert()
    m_cand, res_cand = run_strategy(strat, df, market, start, end, label)
    _count()
    m_base, res_base = run_baseline(df, market, start, end, label)
    _count()
    d_sharpe = sharpe_diff(res_cand, res_base)
    d_growth = log_growth_diff(res_cand, res_base)
    return m_cand, m_base, d_sharpe, d_growth


def b1_gate():
    df, label = load_btc_train("spot")
    _assert_no_holdout(df)
    cells = {}
    for mkt_name, mkt in (("SPOT", SPOT), ("FUTURES", FUTURES)):
        for period_name, (s, e) in (
            ("full_inner", (INNER_TRAIN_START, INNER_VAL_END)),
            ("inner_val", (INNER_VAL_START, INNER_VAL_END)),
        ):
            m_cand, m_base, d_sharpe, d_growth = b1_cell(df, mkt, s, e, label)
            passed = d_sharpe.diff.lo > 0.0
            cells[(mkt_name, period_name)] = dict(
                cand_sharpe=m_cand.sharpe, base_sharpe=m_base.sharpe,
                cand_final=m_cand.final_balance, base_final=m_base.final_balance,
                d_sharpe=d_sharpe, d_growth=d_growth, passed=passed,
            )
            print(f"[B1 {mkt_name}/{period_name}] cand_sharpe={m_cand.sharpe:.3f} "
                  f"base_sharpe={m_base.sharpe:.3f} "
                  f"d_sharpe={d_sharpe.diff.point:.4f} [{d_sharpe.diff.lo:.4f}, {d_sharpe.diff.hi:.4f}] "
                  f"d_growth={d_growth.diff.point:.4f} [{d_growth.diff.lo:.4f}, {d_growth.diff.hi:.4f}] "
                  f"-> {'PASS' if passed else 'FAIL'} (CI-excludes-zero-positive)")
    return cells


# ----------------------------------------------------------------------
# 4. B3 plateau sweep (inner-validation cells only, window x B3_MULTIPLIERS)
# ----------------------------------------------------------------------

def b3_gate():
    df, label = load_btc_train("spot")
    _assert_no_holdout(df)
    base_window = ConservativeMVRVExpert.MVRV_WINDOW_DAYS
    rows = []
    for mult in B3_MULTIPLIERS:
        w = max(2, round(base_window * mult))
        for mkt_name, mkt in (("SPOT", SPOT), ("FUTURES", FUTURES)):
            strat = ConservativeMVRVExpert(mvrv_window_days=w)
            m_cand, res_cand = run_strategy(strat, df, mkt, INNER_VAL_START, INNER_VAL_END, label)
            _count()
            m_base, res_base = run_baseline(df, mkt, INNER_VAL_START, INNER_VAL_END, label)
            _count()
            d_sharpe = sharpe_diff(res_cand, res_base)
            same_sign_positive = d_sharpe.diff.point > 0
            rows.append(dict(mult=mult, window=w, market=mkt_name,
                              d_sharpe_point=d_sharpe.diff.point,
                              d_sharpe_lo=d_sharpe.diff.lo, d_sharpe_hi=d_sharpe.diff.hi,
                              positive=same_sign_positive))
            print(f"[B3 x{mult} window={w}d {mkt_name}] d_sharpe={d_sharpe.diff.point:.4f} "
                  f"[{d_sharpe.diff.lo:.4f}, {d_sharpe.diff.hi:.4f}] "
                  f"-> {'positive' if same_sign_positive else 'non-positive'}")
    return rows


# ----------------------------------------------------------------------
# 5. B4 falsification test: ETH spot, inner-validation, exact frozen construction
# ----------------------------------------------------------------------

def b4_gate():
    eth = load_eth_train()
    _assert_no_holdout(eth)
    strat = ConservativeMVRVExpert(mvrv_asset="ETH")
    m_cand, res_cand = run_strategy(strat, eth, SPOT, INNER_VAL_START, INNER_VAL_END, "ETH spot")
    _count()

    from tradebot.registry import get_strategy
    base_strat = get_strategy("hedge_experts")
    from tradebot.window import run_period
    from tradebot.metrics import compute_metrics
    res_base = run_period(base_strat, eth, start=INNER_VAL_START, end=INNER_VAL_END,
                           market=SPOT, start_balance=1000.0, data_label="ETH spot")
    m_base = compute_metrics(res_base)
    _count()

    d_sharpe = sharpe_diff(res_cand, res_base)
    print(f"[B4 ETH spot inner-val] cand_sharpe={m_cand.sharpe:.3f} base_sharpe={m_base.sharpe:.3f} "
          f"d_sharpe={d_sharpe.diff.point:.4f} [{d_sharpe.diff.lo:.4f}, {d_sharpe.diff.hi:.4f}]")
    return d_sharpe


# ----------------------------------------------------------------------
# 6. B5: 0.40% taker fee tier, inner-validation, both markets
# ----------------------------------------------------------------------

def b5_gate():
    df, label = load_btc_train("spot")
    _assert_no_holdout(df)
    rows = {}
    for mkt_name, mkt in (("SPOT_HIGH_FEE", SPOT_HIGH_FEE), ("FUTURES_HIGH_FEE", FUTURES_HIGH_FEE)):
        strat = ConservativeMVRVExpert()
        m_cand, res_cand = run_strategy(strat, df, mkt, INNER_VAL_START, INNER_VAL_END, label)
        _count()
        m_base, res_base = run_baseline(df, mkt, INNER_VAL_START, INNER_VAL_END, label)
        _count()
        d_sharpe = sharpe_diff(res_cand, res_base)
        rows[mkt_name] = d_sharpe
        print(f"[B5 {mkt_name} inner-val] cand_sharpe={m_cand.sharpe:.3f} base_sharpe={m_base.sharpe:.3f} "
              f"d_sharpe={d_sharpe.diff.point:.4f} [{d_sharpe.diff.lo:.4f}, {d_sharpe.diff.hi:.4f}]")
    return rows


if __name__ == "__main__":
    print("=" * 70)
    print("R-132 CONSERVATIVE: ConservativeMVRVExpert (hedge_experts + 1 MVRV vote)")
    print("=" * 70)

    ok = causal_truncation_probe()
    if not ok:
        print("\nCAUSAL TRUNCATION PROBE FAILED -- branch invalid, stopping.")
        sys.exit(1)

    a2 = a2_gate()

    b1 = b1_gate()
    b1_pass = all(c["passed"] for c in b1.values())

    if not b1_pass:
        print("\nB1 FAILED on at least one cell. Per decision rule this alone is enough "
              "for a NEGATIVE verdict. Running remaining gates anyway for completeness "
              "(diagnostic only).")

    b3 = b3_gate()
    b4 = b4_gate()
    b5 = b5_gate()

    print("\n" + "=" * 70)
    print(f"TOTAL CONFIGURATIONS EVALUATED (this branch): {N_CONFIGS_EVALUATED}")
    print("=" * 70)
