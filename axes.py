"""Per-role style axes, and a test of whether each one describes the player."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

import config
import data
import features
import labels

N_AXES = 6
MIN_GROUPS, MIN_ROWS = 25, 60


def icc(x: pd.Series, gid: pd.Series) -> tuple[float, int]:
    """
    One-way random-effects ICC on unbalanced groups: the share of variance that
    is between players rather than within a player.

    1.0 = every observation of a player lands in the same place.
    0.0 = knowing the player tells you nothing.
    """
    m = x.notna()
    x, gid = x[m], gid[m]
    keep = gid.map(gid.value_counts()) >= 2
    x, gid = x[keep], gid[keep]
    if gid.nunique() < MIN_GROUPS or len(x) < MIN_ROWS:
        return np.nan, gid.nunique()

    n, group_mean = x.groupby(gid).size(), x.groupby(gid).mean()
    N, k = len(x), len(n)
    msb = float((n * (group_mean - x.mean()) ** 2).sum()) / (k - 1)
    msw = float(((x - x.groupby(gid).transform("mean")) ** 2).sum()) / (N - k)
    n0 = (N - float((n ** 2).sum()) / N) / (k - 1)

    var_between = (msb - msw) / n0
    total = var_between + msw
    return (0.0 if total <= 0 else max(float(var_between / total), 0.0)), k


def style_columns(df: pd.DataFrame, groups: dict) -> tuple[list[str], dict]:
    cols = [c for gc in groups.values() for c in gc
            if c in df.columns and c not in data.IDENTITY_COLS]
    col_group = {c: g for g, gc in groups.items() for c in gc if c in cols}
    return cols, col_group


def prepare(outfield: pd.DataFrame, groups=None):
    """Residualise team context once, then hand out role slices."""
    groups = groups or config.OUTFIELD_GROUPS
    cols, col_group = style_columns(outfield, groups)
    return features.residualize(outfield, cols), cols, col_group


def role_matrix(df: pd.DataFrame, role: str, cols: list[str]):
    d = df[df["role"] == role].reset_index(drop=True)
    X = features.winsorize(d[cols].apply(pd.to_numeric, errors="coerce"))
    X = X.fillna(X.median())
    return d, (X - X.mean()) / (X.std(ddof=0) + 1e-9)


def factors(Z: pd.DataFrame, k=N_AXES, seed=0, max_iter=250):
    fa = FactorAnalysis(n_components=k, rotation="varimax",
                        random_state=seed, max_iter=max_iter)
    scores = fa.fit_transform(Z.to_numpy())
    names = [f"F{i + 1}" for i in range(k)]
    loadings = pd.DataFrame(fa.components_.T, index=Z.columns, columns=names)
    scores = pd.DataFrame(scores, columns=names)

    # factor signs are arbitrary: make "more of the dominant concept" positive
    for f in names:
        loadings[f], scores[f] = labels.orient(loadings[f], scores[f])

    # varimax does not order factors; put the biggest first
    share = (loadings ** 2).sum() / len(Z.columns)
    order = share.sort_values(ascending=False).index
    loadings, scores, share = loadings[order], scores[order], share[order]
    loadings.columns = scores.columns = share.index = names
    return loadings, scores, share


def stability(d: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-club ICC per axis: does a player keep his position after a transfer?

    Only players who moved, one row per club spell, so within-player variance is
    variance across systems. Two seasons at the same club share tactics and
    would make a team detector look like a style detector.
    """
    t = d[["player", "team", "Playing Time_Min"]].join(scores)
    movers = t[t.groupby("player")["team"].transform("nunique") >= 2]
    spells = (movers.sort_values("Playing Time_Min", ascending=False)
                    .drop_duplicates(["player", "team"]))
    rows = [dict(zip(("icc_cross_club", "n_movers"), icc(spells[f], spells["player"])))
            for f in scores.columns]
    return pd.DataFrame(rows, index=scores.columns)


def fit(outfield: pd.DataFrame, groups=None, k=N_AXES, verbose=True):
    """Fit the axes for every role. Returns {role: (loadings, scores, table)}."""
    resid, cols, col_group = prepare(outfield, groups)
    out = {}
    for role in data.ROLES:
        d, Z = role_matrix(resid, role, cols)
        if len(d) < 400:
            continue
        loadings, scores, share = factors(Z, k)
        table = stability(d, scores)
        table["var_share"] = share
        out[role] = (d, loadings, scores, table, col_group)
        if verbose:
            best = table["icc_cross_club"].max()
            print(f"[axes] {role}: {len(d):,} rows, "
                  f"{int(table['n_movers'].max())} movers, best icc {best:.2f}")
    return out
