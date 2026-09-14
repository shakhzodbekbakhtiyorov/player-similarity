"""
Does any of the weight tuning actually retrieve better?

Weights are fit on seasons <= 2223 and scored on 2324/2425, one index per
position. Run twice: with the FIFA identity columns in the matrix, and without.

The contrast is the point. With them, ICC weights look spectacular -- because
height never changes, so the engine becomes a height lookup and same-player
recall rewards it. Without them, every arm lands inside noise of every other.

    python -m experiments.ab_weights
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import data
import evaluate
import features
from experiments.weights_from_icc import table as icc_table

TRAIN = [1718, 1819, 1920, 2021, 2122, 2223]
TEST = [2324, 2425]


def icc_weights(df, groups, priors):
    cols = [c for gc in groups.values() for c in gc if c in df.columns]
    col_group = {c: g for g, gc in groups.items() for c in gc if c in cols}
    tab = icc_table(df[df.season.isin(TRAIN)], cols, col_group, "bucket")
    rel = tab.groupby(["bucket", "group"])["icc_cross_club"].mean().unstack("group")
    rel = rel.div(rel.mean(axis=1), axis=0)
    mixed = rel.mul(pd.Series(priors).reindex(rel.columns), axis=1)
    return rel, mixed.mul(sum(priors.values()) / mixed.sum(axis=1), axis=0)


def score(test, groups, weights_by_bucket, k=10):
    rows = []
    for bucket, g in test.groupby("bucket"):
        w = weights_by_bucket.get(bucket)
        if w is None or len(g) < 300:
            continue
        fm = features.build(g.reset_index(drop=True), groups, w)
        r = evaluate.same_player_recall(fm, k=k, verbose=False)
        rows.append({"bucket": bucket, "recall_transfer": r["recall_transfer"],
                     "n": r["n_transfer"]})
    return pd.DataFrame(rows)


def run(drop_identity: bool, k=10):
    outfield, _ = data.load()
    groups = {g: [c for c in cols if not (drop_identity and c in data.IDENTITY_COLS)]
              for g, cols in config.OUTFIELD_GROUPS.items()}
    groups = {g: c for g, c in groups.items() if c}
    priors = {g: w for g, w in config.OUTFIELD_WEIGHTS.items() if g in groups}

    rel, mixed = icc_weights(outfield, groups, priors)
    test = outfield[outfield.season.isin(TEST)].reset_index(drop=True)
    buckets = list(mixed.index)

    arms = {"hand": {b: dict(priors) for b in buckets},
            "icc_pure": {b: rel.loc[b].to_dict() for b in buckets},
            "prior_x_icc": {b: mixed.loc[b].to_dict() for b in buckets}}

    out = {}
    for name, w in arms.items():
        r = score(test, groups, w, k)
        out[name] = float((r.recall_transfer * r.n).sum() / r.n.sum())
    return out


if __name__ == "__main__":
    for drop in (False, True):
        label = "identity columns REMOVED" if drop else "identity columns present"
        print(f"\n=== {label} ===")
        for arm, v in run(drop).items():
            print(f"  {arm:<14} pooled recall_transfer = {v:.4f}")
