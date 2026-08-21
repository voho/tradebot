"""R-78 addendum — sizing the follow-up, not deciding this round.

**Read the label first.** Nothing here feeds R-78's pre-registered
classification, either branch's falsification test, or any verdict. The
round was already decided by
``r78_conservative_b06_horizon.py`` (NOT VIABLE AS SPECIFIED) and
``r78_novel_record_fidelity.py`` (MATERIALLY COSTLY) before this file
existed. This exists so the backlog item R-78 files (**B-38**) is a
*costed* proposal rather than a plausible-sounding guess — because this
ledger's own rule is that an untried idea must go on the backlog as
untried, and an idea whose payoff can be measured cheaply should not be
filed on intuition.

**The question.** The conservative branch found the binding quantity is
not the effect size but the ratio of effect to daily noise. The paired
difference `kelly_regime_v4 − buy_and_hold` carries sd ≈ 3.0%/day, which
is not disagreement about the market — it is mostly *exposure*: v4 holds
roughly 0.4 of the notional a fully-invested hold does, so ~0.6 of BTC's
own daily move sits in the difference as common-mode variance before any
skill is involved. R-33 built the machinery that removes exactly that
term, and its lesson ("before believing any comparison here, check
whether the two arms carry the same risk") has so far been applied only
to backtest comparisons, never to what B-06 records forward.

So: if the forward record paired v4 against a passive long carrying
**v4's own mean notional** instead of against a fully-invested hold, how
much does the daily noise fall, and what does that do to the horizon?

Run::

    python experiments/r78_matched_arm_sizing.py

Holdout: **+0** — the frame comes from ``r78_shared.load_truncated()``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.r78_shared import (  # noqa: E402
    FEE_LIVE,
    W_TRAIN,
    W_VAL,
    load_truncated,
    paired_daily_diff,
)
from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.inference import daily_returns  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402
from tradebot.window import run_period  # noqa: E402


class ConstantExposure(Strategy):
    """A passive long holding a constant fraction of equity — no gate, no
    forecast, no timing. R-33's matched benchmark, as a Strategy.

    Deliberately the dumbest possible arm: it exists to carry a chosen
    exposure and nothing else, so a difference against it is a statement
    about *timing* rather than about how much notional each side holds.
    """

    name = "_r78_constant_exposure"
    warmup = 0

    def __init__(self, fraction: float = 0.4) -> None:
        self.fraction = float(fraction)

    def on_bar(self, ctx: Context) -> None:
        ctx.order_notional(self.fraction)


def mean_notional(df: pd.DataFrame, label: str, window: tuple[str, str],
                  fee: float) -> tuple[float, pd.Series]:
    """v4's own mean notional over ``window``, and its daily returns."""
    market = MarketSpec.spot(fee_rate=fee)
    result = run_period(get_strategy("kelly_regime_v4"), df, window[0],
                        window[1], market=market, data_label=label)
    frame = result.df
    exposure = (frame["target"].clip(lower=0.0, upper=1.0)
                if "target" in frame.columns else None)
    if exposure is None:
        raise AssertionError("kelly_regime_v4 lost its target column")
    return float(exposure.mean()), daily_returns(result.equity)


def main() -> None:
    df, label = load_truncated()
    rows = []
    for wname, window in (("inner-train", W_TRAIN), ("inner-val", W_VAL)):
        frac, v4_daily = mean_notional(df, label, window, FEE_LIVE)

        market = MarketSpec.spot(fee_rate=FEE_LIVE)
        matched = run_period(ConstantExposure(frac), df, window[0], window[1],
                             market=market, data_label=label)
        matched_daily = daily_returns(matched.equity)

        joined = pd.concat([v4_daily, matched_daily], axis=1, join="inner")
        joined.columns = ["v4", "matched"]
        d_matched = (joined["v4"] - joined["matched"]).dropna()
        d_raw = paired_daily_diff(df, label, window, FEE_LIVE)

        for tag, d in (("vs fully-invested hold (what B-06 records)", d_raw),
                       ("vs v4's own mean notional (the B-38 proposal)", d_matched)):
            mu, sd = float(d.mean()), float(d.std(ddof=1))
            n_fixed = (1.96 * sd / abs(mu)) ** 2 if mu != 0 else float("inf")
            rows.append({"window": wname, "arm": tag, "mean_notional": frac,
                         "mean_per_day": mu, "sd_per_day": sd,
                         "abs_t_per_sqrt_day": abs(mu) / sd,
                         "fixed_n_years": n_fixed / 365.0})

    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:,.6g}"))

    print("\nWhat this does and does not say:")
    for wname in out["window"].unique():
        sel = out[out["window"] == wname]
        raw, mat = sel.iloc[0], sel.iloc[1]
        print(f"  [{wname}] daily noise {raw['sd_per_day']:.4f} -> "
              f"{mat['sd_per_day']:.4f} "
              f"({100.0 * (1 - mat['sd_per_day'] / raw['sd_per_day']):.1f}% "
              f"lower); look-once horizon "
              f"{raw['fixed_n_years']:,.1f}y -> {mat['fixed_n_years']:,.1f}y")
    print(
        "\n  Removing the exposure term shrinks the COMMON-MODE variance, "
        "which is\n  the only quantity this file measures. It does not "
        "promise the horizon\n  collapses: the mean moves too, and on a "
        "matched pair it can move toward\n  zero as fast as the noise "
        "does. Whichever way it lands, that is the\n  measurement B-38 "
        "exists to make - pre-registered, on its own round, not\n  read off "
        "this addendum."
    )


if __name__ == "__main__":
    main()
