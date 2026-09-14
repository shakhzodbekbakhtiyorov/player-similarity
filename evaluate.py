"""How good is it? Two metrics that need no labels.

    python evaluate.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import data
import features


def unit(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def same_player_recall(fm, k=10, by="bucket", verbose=True) -> dict:
    """
    A player is similar to himself. For everyone with two or more seasons: does
    another of his own seasons appear in his top k?

    recall_transfer is the number that matters. Consecutive seasons at one club
    share tactics and teammates, so a volume-driven engine scores well on
    recall_all for the wrong reason; requiring a different club strips that out.

        recall_all high, recall_transfer low  -> a team detector
        both high                             -> a style detector
    """
    meta, XN = fm.meta, unit(fm.X)
    group = meta[by].astype(str).to_numpy()
    player = meta["player"].astype(str).to_numpy()
    team = meta["team"].astype(str).to_numpy()

    counts = pd.Series(player).value_counts()
    eligible = np.flatnonzero(pd.Series(player).isin(counts[counts >= 2].index))

    hits, transfer_hits, transfers = [], [], 0
    for i in eligible:
        cand = np.flatnonzero((group == group[i]) & (np.arange(len(meta)) != i))
        if not len(cand):
            continue
        top = cand[np.argsort(-(XN[cand] @ XN[i]))[:k]]
        same = player[top] == player[i]
        hits.append(bool(same.any()))

        others = (player == player[i]) & (np.arange(len(meta)) != i)
        if (team[others] != team[i]).any():
            transfers += 1
            transfer_hits.append(bool((same & (team[top] != team[i])).any()))

    out = {"n": len(hits), "k": k,
           "recall_all": float(np.mean(hits)),
           "recall_transfer": float(np.mean(transfer_hits)) if transfer_hits else np.nan,
           "n_transfer": transfers,
           "chance": k / max(len(meta) - 1, 1)}
    if verbose:
        print(f"[eval] same-player recall@{k} over {out['n']:,} player-seasons")
        print(f"       recall_all      = {out['recall_all']:.3f}")
        print(f"       recall_transfer = {out['recall_transfer']:.3f} "
              f"(n={out['n_transfer']:,})  <- the one that matters")
        print(f"       chance          = {out['chance']:.5f}")
        if out["recall_all"] - out["recall_transfer"] > 0.15:
            print("       WARNING: team context is doing the work, not style.")
    return out


def quality_leak(fm, k=10, sample=500, col="overall", seed=0, verbose=True) -> dict:
    """Are neighbours systematically better rated than the population? A style
    engine should be near zero; a big gap means it learned 'good', not 'similar'."""
    q = pd.to_numeric(fm.meta[col], errors="coerce")
    XN = unit(fm.X)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(fm.meta), size=min(sample, len(fm.meta)), replace=False)

    gaps, pairs = [], []
    for i in idx:
        if np.isnan(q.iloc[i]):
            continue
        near = np.argsort(-(XN @ XN[i]))[1:k + 1]
        gaps.append(q.iloc[near].mean() - q.mean())
        pairs.append((q.iloc[i], q.iloc[near].mean()))

    c = pd.DataFrame(pairs, columns=["self", "neighbours"]).dropna()
    out = {"mean_gap": float(np.nanmean(gaps)),
           "gap_in_sd": float(np.nanmean(gaps) / (q.std() + 1e-9)),
           "corr": float(c["self"].corr(c["neighbours"]))}
    if verbose:
        print(f"\n[eval] quality leak on '{col}' (k={k})")
        print(f"       neighbour gap = {out['mean_gap']:+.2f} ({out['gap_in_sd']:+.2f} SD)")
        print(f"       corr(player quality, neighbour quality) = {out['corr']:.3f}")
        if out["corr"] > 0.5:
            print("       WARNING: partly ranking by how good players are.")
    return out


if __name__ == "__main__":
    outfield, _ = data.load()
    fm = features.build(outfield, config.OUTFIELD_GROUPS, config.OUTFIELD_WEIGHTS)
    same_player_recall(fm)
    quality_leak(fm)
