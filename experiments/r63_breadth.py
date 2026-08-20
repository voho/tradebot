"""R-63, operator-side: how much breadth does this panel actually carry?

Named failure mode (F1) in `r63_shared`'s pre-registration says the round
should fail because Grinold's `IR = IC * sqrt(BR)` counts *independent* bets,
not universe size, and crypto majors are one factor wearing eight tickers.
That is a claim about the data, not about either candidate, so it is measured
here rather than inside a branch -- and it is measured on prices alone, so it
cannot be contaminated by, or contaminate, any strategy verdict.

Three readings of "how many independent bets", because no single one is
canonical and they should agree if the answer is robust:

  mean pairwise correlation      the raw number the claim rests on
  participation ratio           `(sum e)^2 / sum(e^2)` over the correlation
                                matrix's eigenvalues -- the effective number
                                of independent factors, standard in RMT
  equal-correlation breadth     `N / (1 + (N-1) * rho)`, the closed form for
                                the effective number of independent bets in a
                                set of N equally-correlated signals

Run: `python experiments/r63_breadth.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.r63_shared import (  # noqa: E402
    UNIVERSE_6,
    UNIVERSE_8,
    align_frames,
    load_universe,
)


def breadth(corr: pd.DataFrame) -> dict:
    c = corr.to_numpy(dtype=float)
    n = len(c)
    off = c[np.triu_indices(n, 1)]
    ev = np.linalg.eigvalsh(c)
    ev = ev[ev > 0]
    rho = float(off.mean())
    return {
        "n": n,
        "mean_corr": rho,
        "min_corr": float(off.min()),
        "max_corr": float(off.max()),
        "participation_ratio": float(ev.sum() ** 2 / (ev**2).sum()),
        "top_eigen_share": float(ev.max() / ev.sum()),
        "equal_corr_breadth": float(n / (1.0 + (n - 1) * rho)),
    }


def main() -> None:
    aligned = align_frames(load_universe(UNIVERSE_8), ("2020-04-01", None))
    px = pd.DataFrame({t: v["close"] for t, v in aligned.items()})
    daily = px.resample("1D").last().pct_change().dropna()
    print(f"daily returns: {len(daily)} days, {daily.index[0].date()} -> {daily.index[-1].date()}\n")
    print(daily.corr().round(2).to_string(), "\n")

    for label, cols in (("U8", UNIVERSE_8), ("U6", UNIVERSE_6)):
        b = breadth(daily[list(cols)].corr())
        gain = np.sqrt(b["equal_corr_breadth"])
        print(
            f"{label}: N={b['n']}  mean_corr={b['mean_corr']:.3f} "
            f"[{b['min_corr']:.2f}, {b['max_corr']:.2f}]  "
            f"participation_ratio={b['participation_ratio']:.2f}  "
            f"top_eigen_share={b['top_eigen_share']:.1%}  "
            f"equal_corr_breadth={b['equal_corr_breadth']:.2f}  "
            f"=> IR multiple vs one asset = sqrt(BR) = {gain:.2f}x"
        )


if __name__ == "__main__":
    main()
