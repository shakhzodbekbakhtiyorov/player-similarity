"""
Per-position feature weights from reliability.

    ICC = variance between players / (between + within the same player)

A feature is worth weighting inside a position if it separates players there and
stays put when the player moves club. Reports icc_all (inflated by team context)
and icc_cross_club (the honest one), rolled up to group weights.

Caveat the numbers make plain: ICC measures whether a feature is RELIABLE, not
whether it is STYLE. height_cm scores ~0.96 because height does not change.

    python -m experiments.weights_from_icc
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import axes
import config
import data


def one_row_per_club(df):
    moved = df.groupby("player")["team"].transform("nunique") >= 2
    return (df[moved].sort_values("Playing Time_Min", ascending=False)
                     .drop_duplicates(["player", "team"]))


def table(df, cols, col_group, level="bucket"):
    rows = []
    for strat, g in df.groupby(level):
        if len(g) < 200:
            continue
        num = g[cols].apply(pd.to_numeric, errors="coerce")
        lo, hi = num.quantile(config.CLIP_QUANTILES[0]), num.quantile(config.CLIP_QUANTILES[1])
        num = num.clip(lower=lo, upper=hi, axis=1)
        cross = one_row_per_club(g.assign(**{c: num[c] for c in cols}))

        for c in cols:
            a, _ = axes.icc(num[c], g["player"])
            x, _ = axes.icc(cross[c], cross["player"])
            rows.append({level: strat, "group": col_group[c], "feature": c,
                         "identity": c in data.IDENTITY_COLS,
                         "icc_all": a, "icc_cross_club": x, "team_drop": a - x})
    return pd.DataFrame(rows)


def run(df, level="bucket", groups=None, priors=None):
    groups = groups or config.OUTFIELD_GROUPS
    priors = priors or config.OUTFIELD_WEIGHTS
    cols = [c for gc in groups.values() for c in gc if c in df.columns]
    col_group = {c: g for g, gc in groups.items() for c in gc if c in cols}

    tab = table(df, cols, col_group, level)
    rel = (tab.groupby([level, "group"])["icc_cross_club"].mean().unstack("group"))
    rel = rel.div(rel.mean(axis=1), axis=0)

    print(f"\n=== relative ICC by {level} (1.0 = average reliability) ===")
    print(rel.round(2).to_string())
    print("\n=== team-context drop (icc_all - icc_cross_club) ===")
    print(tab.groupby([level, "group"])["team_drop"].mean().unstack("group").round(3).to_string())
    print("\n=== most reliable features (note what tops the list) ===")
    print(tab.nlargest(8, "icc_cross_club")[["feature", "identity", "icc_cross_club"]]
          .round(3).to_string(index=False))
    return tab, rel


if __name__ == "__main__":
    outfield, _ = data.load()
    run(outfield)
