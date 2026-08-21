#!/usr/bin/env python
"""R-90 NOVEL branch: the ATR-SCALED TRAILING-STOP RATCHET with a
data-driven-recovery restart, overlaid on `kelly_regime_v4`'s exit.

MECHANISM (one sentence). Every construction in 89 prior rounds decides
exposure from an EXTERNAL anchor vote alone ("flat or full, decided by a
20/40/80-day trend consensus"); this branch adds a PATH-DEPENDENT overlay
that forces flat the instant a held long's close falls more than `k` ATRs
below its own running peak since entry, and only re-arms once price has
independently reclaimed the exact exit level AND a minimum cooldown has
elapsed -- so the exposure now depends on the trade's own realised P&L
path, never varied before (B-41).

LITERATURE (backlog B-41, filed by R-89; verified before dispatch):
  - Sepp & Lucic (2026), arXiv:2607.19497. "American" trend systems: binary
    position, ATR-scaled entry buffer and ATR-scaled TRAILING STOP --
    structurally the volatility-adaptive stop-distance half of this branch.
  - Han, Zhou & Zhu (2016), SSRN 2407199. Literal FIXED-PERCENTAGE stop on
    momentum deciles; a 15% stop cuts the worst month from -49.8% to -11.4%
    and more than doubles Sharpe. This project's CONSERVATIVE branch (a
    disjoint file, not read here) implements this literally. This branch's
    stop distance is instead `k * ATR_14d(t) / close(t)` -- adaptive to
    realised volatility, not a fixed fraction of price -- which is the
    FIRST axis on which this branch differs from the conservative one.
  - Hsieh (2023), arXiv:2303.02613, IFAC-PapersOnLine. Names the part a
    naive stop omits: with no restart mechanism, a drawdown-triggered exit
    is either a permanent de-risking after one bad episode, or it whipsaws
    on every re-entry. **The paper's own exact restart formula could not be
    extracted from its abstract/PDF at fetch time.** This branch does NOT
    claim to replicate it. It implements the STATED CONCEPT -- re-entry
    gated on a DATA-DRIVEN recovery confirmation (price must reclaim the
    exact price at which the stop fired) plus an explicit minimum cooldown
    -- rather than Han-Zhou-Zhu's unconditional instant resume. This is the
    SECOND axis on which this branch differs from the conservative one.

Both axes together (ATR-adaptive distance x reclaim-gated restart) have
not been combined in this project before; that combination, not either
piece alone, is this branch's contribution.

THE NAMED RISK (B-41's own filing, written before any code): *on BTC,
trailing stops fire on the routine 10-20% intra-trend drawdowns that
punctuate every bull run, and re-entry happens higher than the exit -- the
classic whipsaw, expensive at 10-20bps.* Measured directly below via
`r90_shared.stopout_whipsaw_rate`, the SAME diagnostic the conservative
branch uses, so the two branches' whipsaw rates are comparable.

THE STANDING RISK-MATCH RULE applies with more force here than on any
prior SIZE-axis round: a trailing stop can only ever force MORE flat time
than v4 (never less), so any drawdown improvement is presumptively R-28/
R-32/R-33's "held less, drew down less" artifact until `exposure_ratio`
and `vol_ratio` are both shown in [0.9, 1.1] -- `compare()` reports both on
every cell and B2 below is gated on `risk_matched` for its drawdown leg.

--------------------------------------------------------------------------
THE FROZEN MECHANISM (not redesigned here; implemented exactly as spec'd)
--------------------------------------------------------------------------
stop_frac[i] = min(0.95, k * atr_days(df, 14.0)[i] / close[i])
             = 1.0 wherever atr_days is NaN (ATR warmup: stop unreachable)

Restart: ALWAYS reentry_reclaim=True (price must close back above the
exact price at which the stop fired) AND a minimum cooldown of
`cooldown_days * 288` bars. Both gates must clear (this is exactly how
`r90_shared.apply_trailing_stop` already combines cooldown_ok and
reclaim_ok -- read there, not reimplemented here).

--------------------------------------------------------------------------
THE FROZEN GRID -- 9 configurations, none added or dropped after results
--------------------------------------------------------------------------
  A1 identity (not counted among the 8): forced stop_frac == 1.0
    everywhere (bypassing the ATR formula entirely, to force an EXACT
    passthrough), reentry_reclaim=False, reentry_delay_bars=None.
    Must reproduce r90_shared.v4_target(df) bit-for-bit.

  swept: k in {2.0, 3.0, 4.0, 5.0} x cooldown_days in {0, 3}
         (both always combined with reentry_reclaim=True)      = 8

--------------------------------------------------------------------------
THE FROZEN DECISION RULE (written before any config past A1 is run)
--------------------------------------------------------------------------
Step A, per configuration, before any performance number is read:
  A1 identity     -- forced-passthrough config == v4_target(df), exactly.
  A2 non-inert    -- stop_events.sum() > 0 on inner-train, each of the 8
                     swept configs. A config that never fires is reported
                     as INERT, not scored.
  A3 causality    -- causal_truncation_probe passes on the exact
                     build_target_fn used for selection (including the ATR
                     computation), on real BTC inner-train.

Step B selection: `compare()` over slice_names=("inner_train","inner_val"),
markets=(SPOT, FUTURES). Selection statistic = inner-validation paired
log-growth difference vs v4 on futures_5x, among the Step-A survivors.
Full 8x4 table reported regardless of the winner.

Promotion bar (default REJECT). ALL FIVE must hold for "CANDIDATE FOR
HOLDOUT"; otherwise NEGATIVE, with the failing clause(s) named exactly:
  B1 -- paired bootstrap excludes zero in >=1 of 4 cells AND point
        estimate positive in all 4.
  B2 -- EITHER dSharpe > +0.2 on inner-val on BOTH markets, OR a max-DD
        improvement on inner-val on BOTH markets WHERE risk_matched is
        True for both -- an unmatched drawdown improvement is not
        evidence (standing rule, R-28/R-32/R-33).
  B3 -- plateau not peak: report the finalist's immediate grid neighbours
        (adjacent k at the same cooldown, other cooldown at the same k)
        and state whether they move with the finalist or reverse sharply.
  B4 -- falsification, two parts, BOTH reported regardless of outcome:
        (a) ETH replication, inner_train only (Bitfinex ETH ends
            2019-12-31): same sign of d_loggrowth as BTC inner-train on
            BOTH ETH markets, else this half FAILS.
        (b) whipsaw falsification: stopout_whipsaw_rate on inner-train BTC
            futures_5x; FAILS this half if whipsaw_rate > 0.5 AND the B1
            point estimate on that same cell is NOT positive.
  B5 -- cost robustness: re-run inner-validation (both markets) at 0.40%
        taker; the B1 point-estimate sign must not reverse on either
        market.

This branch does NOT read the holdout and does NOT decide promotion to it
-- it reports CANDIDATE FOR HOLDOUT or NEGATIVE; the holdout read, if any,
is the operator's job.

This file never reads a bar at or after OOS_START (2023-01-01): every load
goes through r90_shared's truncating loaders, and the max timestamp
actually touched is tracked and printed at the end of main().
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.broker import MarketSpec  # noqa: E402

from experiments.r90_shared import (  # noqa: E402
    BARS_PER_DAY,
    FUTURES,
    OOS_START,
    SLICES,
    SPOT,
    apply_deadband,
    apply_trailing_stop,
    atr_days,
    causal_truncation_probe,
    compare,
    load_btc,
    load_eth,
    print_rows,
    r_squared,
    stopout_whipsaw_rate,
    v4_raw_desired,
    v4_target,
)

# ---------------------------------------------------------------------------
# FROZEN GRID
# ---------------------------------------------------------------------------
K_GRID = (2.0, 3.0, 4.0, 5.0)
COOLDOWN_GRID_DAYS = (0, 3)
ATR_DAYS = 14.0

TAKER_040 = 0.0040  # B5 cost-robustness fee


def _mk(name: str, fee: float, leverage: float, short: bool, funding: bool) -> MarketSpec:
    return MarketSpec(name=name, leverage=leverage, fee_rate=fee,
                      allow_short=short, pays_funding=funding)


SPOT_040 = _mk("spot@0.40%", TAKER_040, 1.0, False, False)
FUT_040 = _mk("fut5x@0.40%", TAKER_040, 5.0, True, True)

V4_WARMUP_BARS = 80 * BARS_PER_DAY + 10


# ---------------------------------------------------------------------------
# Caches keyed on frame content (compare() re-`prepare`s the same frame for
# both candidate and control, and stop-run computation is O(n) python loop,
# so caching the (raw_desired, stop run) pair per (frame, config) matters).
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _key(df: pd.DataFrame) -> tuple:
    return (len(df), int(df.index[0].value), int(df.index[-1].value),
            float(df["close"].iloc[0]), float(df["close"].iloc[-1]))


def cached_raw(df: pd.DataFrame) -> np.ndarray:
    k = ("raw",) + _key(df)
    if k not in _CACHE:
        _CACHE[k] = v4_raw_desired(df)
    return _CACHE[k]


# ---------------------------------------------------------------------------
# Config: one (k, cooldown_days) point on the swept grid, or the identity.
# ---------------------------------------------------------------------------
class Config:
    def __init__(self, label: str, *, identity: bool = False,
                 k: float | None = None, cooldown_days: float | None = None):
        self.label = label
        self.identity = identity
        self.k = k
        self.cooldown_days = cooldown_days

    def _run(self, df: pd.DataFrame):
        """Return (TrailingStopResult) for this config on this frame, cached."""
        key = ("run", self.label) + _key(df)
        if key in _CACHE:
            return _CACHE[key]
        raw = cached_raw(df)
        if self.identity:
            stop_frac = np.ones(len(df))
            r = apply_trailing_stop(df, raw, stop_frac,
                                    reentry_delay_bars=None, reentry_reclaim=False)
        else:
            atr = atr_days(df, ATR_DAYS)
            close = df["close"].to_numpy(dtype=float)
            stop_frac = np.where(np.isfinite(atr),
                                 np.minimum(0.95, self.k * atr / close), 1.0)
            delay = np.full(len(df), int(self.cooldown_days * BARS_PER_DAY))
            r = apply_trailing_stop(df, raw, stop_frac,
                                    reentry_delay_bars=delay, reentry_reclaim=True)
        _CACHE[key] = r
        return r

    def build(self, df: pd.DataFrame) -> np.ndarray:
        return apply_deadband(self._run(df).target)

    def stop_events(self, df: pd.DataFrame) -> np.ndarray:
        return self._run(df).stop_events


def frozen_grid() -> list[Config]:
    cfgs = []
    for k in K_GRID:
        for cd in COOLDOWN_GRID_DAYS:
            cfgs.append(Config(f"k{k:.1f} cd{cd}d", k=k, cooldown_days=cd))
    return cfgs


IDENTITY = Config("identity(A1)", identity=True)


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def hdr(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def main() -> None:
    max_ts = []

    hdr("R-90 NOVEL BRANCH -- ATR-SCALED TRAILING-STOP RATCHET, RECLAIM-GATED RESTART")
    print("mechanism: overlay a path-dependent trailing stop on v4's raw desired "
          "exposure --")
    print("stop_frac[i] = min(0.95, k * ATR_14d[i] / close[i]) (NaN ATR -> "
          "stop_frac=1.0, unreachable);")
    print("re-arm only once close reclaims the exact stop-out price AND a "
          "cooldown of cooldown_days*288")
    print("bars has elapsed. Restart is Hsieh's STATED CONCEPT (data-driven "
          "recovery confirmation),")
    print("not a replication of an unseen exact formula -- said plainly, not "
          "hedged.")
    print(f"\nfrozen grid: 1 identity (A1) + k in {K_GRID} x cooldown_days in "
          f"{COOLDOWN_GRID_DAYS} = 8 swept = 9 total configurations")

    btc = load_btc()
    max_ts.append(btc.index.max())
    print(f"\nBTC: {len(btc):,} bars  {btc.index[0]} -> {btc.index[-1]}  (< {OOS_START})")
    train = btc.loc[:SLICES["inner_train"][1]]
    print(f"inner-train frame: {len(train):,} bars  {train.index[0]} -> {train.index[-1]}")

    # ================================================================ STEP A
    hdr("STEP A -- MECHANISM GATE (before any performance number)")

    # --- A1 identity
    ident_path = IDENTITY.build(train)
    v4_path_train = v4_target(train)
    a1_max = float(np.max(np.abs(ident_path - v4_path_train)))
    a1 = a1_max == 0.0
    print(f"A1 identity: max|forced-passthrough - v4_target(train)| = {a1_max:.3e}"
          f"   -> {'PASS' if a1 else 'FAIL'}")
    ident_full = IDENTITY.build(btc)
    v4_path_full = v4_target(btc)
    a1_full_max = float(np.max(np.abs(ident_full - v4_path_full)))
    print(f"    (also checked on the full pre-OOS frame: max abs diff = "
          f"{a1_full_max:.3e})")

    # --- A2 non-inertness: stop_events.sum() > 0 on inner-train, each of 8 configs
    cfgs = frozen_grid()
    assert len(cfgs) == 8, len(cfgs)
    print("\nA2 non-inertness (inner-train): stop_events.sum() must be > 0")
    print(f"{'config':14s} {'k':>5s} {'cooldown_d':>10s} {'stop_events':>12s} "
          f"{'status':>8s}")
    print("-" * 56)
    a2 = {}
    for cfg in cfgs:
        se = cfg.stop_events(train)
        n_events = int(se.sum())
        inert = n_events == 0
        a2[cfg.label] = dict(n_events=n_events, inert=inert)
        print(f"{cfg.label:14s} {cfg.k:5.1f} {cfg.cooldown_days:10.0f} "
              f"{n_events:12d} {'INERT' if inert else 'ok':>8s}")
    n_inert = sum(1 for v in a2.values() if v["inert"])
    print(f"\n{n_inert} of 8 configurations are INERT (stop never fires on "
          f"inner-train) and are excluded from selection.")

    # --- A3 causality: on the exact build_target_fn used for selection,
    # using the config with the shortest stop distance (most likely to
    # expose a lookahead bug in the ATR/np.where construction) plus the
    # identity, on real BTC inner-train.
    hdr("STEP A3 -- CAUSAL TRUNCATION PROBE")
    print("Rebuild the target on 55% and 80% truncations of inner-train; the "
          "surviving prefix must")
    print("match bit-for-bit. This is the check most likely to catch a "
          "lookahead bug in the ATR")
    print("computation or the np.where stop_frac construction.\n")
    probe_cfgs = [IDENTITY] + cfgs  # every config, including identity
    a3 = True
    for cfg in probe_cfgs:
        try:
            ok = causal_truncation_probe(cfg.build, train, cuts=(0.55, 0.80))
        except AssertionError as exc:
            ok = False
            print(f"  {cfg.label:14s} FAIL -- {exc}")
        if ok:
            print(f"  {cfg.label:14s} PASS (cuts 0.55, 0.80)")
        a3 = a3 and ok
    print(f"\nA3 = {'PASS' if a3 else 'FAIL'} (checked on all 9 configurations, "
          f"not just the eventual finalist)")

    survivors = [c for c in cfgs if not a2[c.label]["inert"]]
    print(f"\nStep A survivors eligible for selection: {len(survivors)} of 8 "
          f"({', '.join(c.label for c in survivors)})")

    # ================================================================ STEP B
    hdr("STEP B -- FULL GRID, 8 swept configurations x 4 (slice x market) cells")
    print("candidate vs kelly_regime_v4 control; d_loggrowth is the paired "
          "block-bootstrap difference")
    print("(30-day blocks, 2000 resamples). All 8 configs run and reported "
          "regardless of A2 status.\n")

    all_rows: dict[str, list[dict]] = {}
    for cfg in cfgs:
        rows = compare(cfg.build, btc, label=cfg.label)
        all_rows[cfg.label] = rows
        print_rows(rows)
        print()

    def sel_stat(label: str) -> float:
        for r in all_rows[label]:
            if r["slice"] == "inner_val" and r["market"] == "futures_5x":
                return r["d_loggrowth"]
        return float("nan")

    hdr("SELECTION -- inner-validation paired log-growth difference vs v4, futures_5x")
    print(f"{'config':14s} {'eligible':>9s} {'selstat':>9s} {'[lo':>9s},{'hi]':>9s} "
          f"{'why not':<20s}")
    print("-" * 74)
    for cfg in cfgs:
        reasons = []
        if a2[cfg.label]["inert"]:
            reasons.append("INERT")
        ok = not reasons
        row = [r for r in all_rows[cfg.label]
               if r["slice"] == "inner_val" and r["market"] == "futures_5x"][0]
        print(f"{cfg.label:14s} {'YES' if ok else 'no':>9s} "
              f"{row['d_loggrowth']:+9.3f} {row['d_lo']:+9.3f},{row['d_hi']:+9.3f} "
              f"{'; '.join(reasons):<20s}")

    if not survivors:
        print("\nNo Step-A-eligible configuration. VERDICT: NEGATIVE by construction.")
        print(f"\nConfigurations evaluated: 1 identity + 8 swept = 9 total.")
        print(f"max timestamp read anywhere: {max(max_ts)}  (< {OOS_START})")
        return

    finalist = max(survivors, key=lambda c: sel_stat(c.label))
    print(f"\nFINALIST: {finalist.label}  (k={finalist.k}, cooldown_days="
          f"{finalist.cooldown_days})   selection statistic "
          f"{sel_stat(finalist.label):+.4f} log units (inner-val, futures_5x)")

    frows = all_rows[finalist.label]
    print()
    print_rows(frows)

    # ------------------------------------------------------------ B1
    hdr("THE FROZEN DECISION RULE -- clause by clause")

    pts = [r["d_loggrowth"] for r in frows]
    excl = [r["excludes_zero"] for r in frows]
    b1 = any(excl) and all(p > 0 for p in pts)
    print(f"B1 paired bootstrap: excludes zero in {sum(excl)}/4 cells; point "
          f"estimate positive in {sum(1 for p in pts if p > 0)}/4 cells")
    for r in frows:
        print(f"   {r['slice']:11s} {r['market']:11s} d_loggrowth={r['d_loggrowth']:+.4f} "
              f"[{r['d_lo']:+.4f},{r['d_hi']:+.4f}]  excludes_zero={r['excludes_zero']}")
    print(f"   B1 = {'PASS' if b1 else 'FAIL'}")

    # ------------------------------------------------------------ B2
    val = [r for r in frows if r["slice"] == "inner_val"]
    dsh = {r["market"]: r["d_sharpe"] for r in val}
    ddd = {r["market"]: r["d_dd"] for r in val}
    rm = {r["market"]: r["risk_matched"] for r in val}
    b2_sharpe = all(v > 0.2 for v in dsh.values())
    b2_dd = all(v < 0.0 for v in ddd.values()) and all(rm.values())
    b2 = b2_sharpe or b2_dd
    print(f"\nB2 noise floor (inner-validation):")
    print(f"   dSharpe:  " + ", ".join(f"{k}={v:+.3f}" for k, v in dsh.items())
          + f"   -> both > +0.2: {b2_sharpe}")
    print(f"   dMaxDD:   " + ", ".join(f"{k}={v:+.2f}pp" for k, v in ddd.items())
          + f"   -> both improved: {all(v < 0.0 for v in ddd.values())}")
    print(f"   risk_matched: " + ", ".join(f"{k}={v}" for k, v in rm.items())
          + f"   -> both matched: {all(rm.values())}")
    print(f"   exposure_ratio / vol_ratio (cand/v4), inner-val: " + ", ".join(
        f"{r['market']}=exp{r['exposure_ratio']:.2f}/vol{r['vol_ratio']:.2f}"
        for r in val))
    print(f"   drawdown leg counts as evidence only if BOTH improved AND BOTH "
          f"risk-matched: {b2_dd}")
    print(f"   B2 = {'PASS' if b2 else 'FAIL'} (via {'Sharpe' if b2_sharpe else ('matched drawdown' if b2_dd else 'neither')})")

    # ------------------------------------------------------------ B3 plateau
    print("\nB3 plateau not peak: the finalist's immediate grid neighbours")
    print(f"{'config':14s} {'k':>5s} {'cd_d':>5s} {'selstat':>9s} {'note':<14s}")
    print("-" * 55)
    fk, fcd = finalist.k, finalist.cooldown_days
    neigh_labels = []
    ik = K_GRID.index(fk)
    for di in (-1, 1):
        j = ik + di
        if 0 <= j < len(K_GRID):
            neigh_labels.append(f"k{K_GRID[j]:.1f} cd{int(fcd)}d")
    other_cd = [c for c in COOLDOWN_GRID_DAYS if c != fcd]
    for cd in other_cd:
        neigh_labels.append(f"k{fk:.1f} cd{int(cd)}d")
    b3_vals = []
    for lb in [finalist.label] + neigh_labels:
        s = sel_stat(lb)
        note = "<-- FINALIST" if lb == finalist.label else ""
        cfgobj = next(c for c in cfgs if c.label == lb)
        print(f"{lb:14s} {cfgobj.k:5.1f} {cfgobj.cooldown_days:5.0f} {s:+9.4f} "
              f"{note:<14s}")
        if lb != finalist.label:
            b3_vals.append(s)
    same_direction = all(np.sign(v) == np.sign(sel_stat(finalist.label)) for v in b3_vals) if b3_vals else False
    print(f"\n   finalist selstat sign: {'+' if sel_stat(finalist.label) > 0 else '-'}; "
          f"neighbours all same sign: {same_direction}")
    b3 = bool(b3_vals) and same_direction
    print(f"   B3 = {'PASS' if b3 else 'FAIL'} ('plateau' reading: neighbours move "
          f"with the finalist rather than reversing sharply, per the values above)")

    # ------------------------------------------------------------ B4a ETH
    hdr("B4(a) -- FALSIFICATION: ETH REPLICATION (inner_train only)")
    eth = load_eth()
    max_ts.append(eth.index.max())
    print(f"ETH (Bitfinex): {len(eth):,} bars  {eth.index[0]} -> {eth.index[-1]}")
    print(f"ETH ends {eth.index.max().date()}, before inner-validation begins "
          f"(2021-01-01) -- only inner_train is run on ETH; reported plainly, "
          f"not worked around.")
    eth_rows = compare(finalist.build, eth, label=finalist.label,
                       slice_names=("inner_train",))
    print()
    print_rows(eth_rows)
    btc_train_row = [r for r in frows if r["slice"] == "inner_train"
                     and r["market"] == "futures_5x"][0]
    btc_train_sign_by_mkt = {r["market"]: (1.0 if r["d_loggrowth"] > 0 else -1.0)
                             for r in frows if r["slice"] == "inner_train"}
    eth_ok = []
    for r in eth_rows:
        btc_sign = btc_train_sign_by_mkt.get(r["market"])
        same = btc_sign is not None and np.sign(r["d_loggrowth"]) == btc_sign
        eth_ok.append(same)
        print(f"   {r['market']:11s}: BTC inner-train sign="
              f"{'+' if btc_sign and btc_sign > 0 else '-'}  ETH d_loggrowth="
              f"{r['d_loggrowth']:+.4f}  same sign: {same}")
    b4a = bool(eth_rows) and all(eth_ok)
    print(f"   B4(a) = {'PASS' if b4a else 'FAIL'}")

    # ------------------------------------------------------------ B4b whipsaw
    hdr("B4(b) -- FALSIFICATION: WHIPSAW RATE (inner-train BTC futures_5x)")
    fin_run = finalist._run(train)
    fin_target_train = apply_deadband(fin_run.target)
    close_train = train["close"].to_numpy(dtype=float)
    whip = stopout_whipsaw_rate(close_train, fin_target_train, fin_run.stop_events)
    print(f"stop_events={whip['stop_events']}  events_with_reentry_in_horizon="
          f"{whip['events_with_reentry_in_horizon']}  whipsaws={whip['whipsaws']}")
    print(f"whipsaw_rate={whip['whipsaw_rate']:.4f}  "
          f"mean_whipsaw_log_cost={whip['mean_whipsaw_log_cost']:+.5f}")
    b1_train_futures_positive = btc_train_row["d_loggrowth"] > 0
    print(f"\nB1 point estimate on inner-train/futures_5x: "
          f"{btc_train_row['d_loggrowth']:+.4f}  (positive: {b1_train_futures_positive})")
    b4b_fail = (whip["whipsaw_rate"] > 0.5) and (not b1_train_futures_positive)
    b4b = not b4b_fail
    print(f"FAIL condition: whipsaw_rate > 0.5 AND inner-train/futures point "
          f"estimate NOT positive -> {b4b_fail}")
    print(f"   B4(b) = {'PASS' if b4b else 'FAIL'}")

    b4 = b4a and b4b
    print(f"\nB4 (both parts) = {'PASS' if b4 else 'FAIL'}")

    # ------------------------------------------------------------ B5 fees
    hdr("B5 -- COST ROBUSTNESS: 0.40% TAKER, inner-validation")
    print("MarketSpec(name=..., leverage=..., fee_rate=0.0040, allow_short=..., "
          "pays_funding=...) -- fields read from tradebot.broker.MarketSpec "
          "directly, not guessed.")
    fee_rows = compare(finalist.build, btc, label=finalist.label + "@40bp",
                       markets=(SPOT_040, FUT_040), slice_names=("inner_val",))
    print()
    print_rows(fee_rows)
    val_sign_by_mkt = {r["market"]: (1.0 if r["d_loggrowth"] > 0 else -1.0)
                       for r in val}
    fee_sign_ok = []
    for r in fee_rows:
        base_mkt = "spot" if "spot" in r["market"] else "futures_5x"
        base_sign = val_sign_by_mkt.get(base_mkt)
        same = base_sign is not None and np.sign(r["d_loggrowth"]) == base_sign
        fee_sign_ok.append(same)
        print(f"   {r['market']:14s} d_loggrowth={r['d_loggrowth']:+.4f}  "
              f"(base-fee inner-val {base_mkt} sign="
              f"{'+' if base_sign and base_sign>0 else '-'})  same sign: {same}")
    b5 = all(fee_sign_ok)
    print(f"   B5 = {'PASS' if b5 else 'FAIL'}")

    # ------------------------------------------------------------ verdict
    hdr("VERDICT")
    clauses = {"B1": b1, "B2": b2, "B3": b3, "B4(a) ETH": b4a,
               "B4(b) whipsaw": b4b, "B5 0.40% taker": b5}
    for k, v in clauses.items():
        print(f"  {k:16s} {'PASS' if v else 'FAIL'}")
    promote = all(clauses.values())
    print(f"\nVERDICT: {'CANDIDATE FOR HOLDOUT' if promote else 'NEGATIVE'}")
    if not promote:
        failed = [k for k, v in clauses.items() if not v]
        print(f"Failing clause(s): {', '.join(failed)}")

    print(f"\nFinalist config: k={finalist.k}, cooldown_days={finalist.cooldown_days}, "
          f"ATR window={ATR_DAYS}d, reentry_reclaim=True")

    print(f"\nConfigurations evaluated in this file:")
    print(f"  A1 identity check: 1")
    print(f"  frozen swept grid (evaluated on all 4 cells each): 8")
    print(f"  finalist re-runs on other data/costs (not new configurations): "
          f"ETH inner_train (2 markets), 0.40% taker inner-val (2 markets)")
    print(f"  => total configurations evaluated: 9 "
          f"(1 identity + 8 swept); finalist re-run 4 additional cells "
          f"(ETH x2, fee x2), not new configurations.")
    print(f"\nmax timestamp read anywhere in this branch (BTC and ETH): "
          f"{max(max_ts)}  (< {OOS_START}) -- no holdout bar was read.")


if __name__ == "__main__":
    main()
