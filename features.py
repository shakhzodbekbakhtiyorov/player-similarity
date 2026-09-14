"""Feature matrix: scale, weight, and optionally strip out team context."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config


@dataclass
class FeatureMatrix:
    X: np.ndarray             # scaled and weighted, ready for a distance
    Z: np.ndarray             # scaled only, for diagnostics and explanations
    columns: list[str]
    col_group: list[str]
    col_weight: np.ndarray
    meta: pd.DataFrame


def resolve(df: pd.DataFrame, groups: dict) -> tuple[list[str], list[str]]:
    cols, col_group, missing = [], [], []
    for group, group_cols in groups.items():
        for c in group_cols:
            if c in df.columns:
                cols.append(c)
                col_group.append(group)
            else:
                missing.append(f"{group}:{c}")
    if missing:
        print(f"[features] {len(missing)} configured columns missing: {missing[:5]}")
    return cols, col_group


def winsorize(X: pd.DataFrame) -> pd.DataFrame:
    lo, hi = X.quantile(config.CLIP_QUANTILES[0]), X.quantile(config.CLIP_QUANTILES[1])
    return X.clip(lower=lo, upper=hi, axis=1)


def zscore(X: pd.DataFrame, by: pd.Series) -> pd.DataFrame:
    """Within position, so a defender's 3 tackles and a winger's 3 tackles
    stop meaning the same thing."""
    g = X.groupby(by)
    return (X - g.transform("mean")) / (g.transform("std", ddof=0) + 1e-9)


def build(df: pd.DataFrame, groups: dict, weights: dict, meta_cols=None) -> FeatureMatrix:
    cols, col_group = resolve(df, groups)
    meta_cols = meta_cols or [c for c in config.META_COLS if c in df.columns]

    raw = winsorize(df[cols].apply(pd.to_numeric, errors="coerce"))
    raw = raw.fillna(raw.groupby(df["bucket"]).transform("median")).fillna(raw.median())
    Z = zscore(raw, df["bucket"])

    # A group's weight is split across its columns, so a 12-column concept does
    # not quietly outweigh a 2-column one.
    sizes = pd.Series(col_group).value_counts().to_dict()
    w = np.array([weights[g] / sizes[g] for g in col_group], dtype=float)

    print(f"[features] {len(df):,} rows x {len(cols)} cols")
    return FeatureMatrix(
        X=Z.to_numpy(dtype=float) * np.sqrt(w),   # sqrt: squared distance picks up w
        Z=Z.to_numpy(dtype=float),
        columns=cols, col_group=col_group, col_weight=w,
        meta=df[["player_key"] + meta_cols].reset_index(drop=True),
    )


# --------------------------------------------------------------------------
# team context
# --------------------------------------------------------------------------
def team_volume(df: pd.DataFrame, col="Per 90 Minutes_Total_Att") -> pd.Series:
    """Squad-mean passes per 90: a stand-in for how much a team keeps the ball."""
    return df.groupby(["team", "season"])[col].transform("mean")


def residualize(df: pd.DataFrame, cols, by="bucket") -> pd.DataFrame:
    """
    Regress each feature on squad passing volume and keep the residual, so what
    remains is the player relative to what his team's style predicts.

    Removes real signal too: a player whose identity is playing for a
    possession side loses part of it. That is the trade.
    """
    df = df.copy()
    proxy = pd.to_numeric(team_volume(df), errors="coerce")
    cols = [c for c in cols if c in df.columns]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce").astype("float64")

    for _, idx in df.groupby(by).groups.items():
        idx = pd.Index(idx)
        pv = proxy.loc[idx]
        ok = np.isfinite(pv)
        if ok.sum() < 30:
            continue
        for c in cols:
            y = df.loc[idx, c]
            m = ok & np.isfinite(y)
            if m.sum() < 30:
                continue
            X = np.column_stack([np.ones(m.sum()), pv[m].to_numpy()])
            beta = np.linalg.pinv(X.T @ X) @ X.T @ y[m].to_numpy()
            df.loc[y[m].index, c] = y[m].to_numpy() - X @ beta
    return df
