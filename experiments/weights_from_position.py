"""
The original idea: learn group weights from permutation importance for
predicting a player's position.

Kept because it is the wrong question, and that is worth showing. Within-position
z-scoring has already removed the between-position signal this measures, and
"which feature predicts position" has no per-position version.

    python -m experiments.weights_from_position
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

import config
import data


def run(df, groups=None, target="pos_primary", min_class=40, seed=0):
    groups = groups or config.OUTFIELD_GROUPS
    cols = [c for gc in groups.values() for c in gc if c in df.columns]
    col_group = {c: g for g, gc in groups.items() for c in gc if c in cols}

    y = df[target].astype(str)
    keep = y.map(y.value_counts()) >= min_class
    X, y = df.loc[keep, cols].apply(pd.to_numeric, errors="coerce"), y[keep]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=seed, stratify=y)
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(Xtr, ytr)
    print(f"holdout accuracy {clf.score(Xte, yte):.3f} "
          f"(baseline {yte.value_counts(normalize=True).max():.3f})")

    imp = permutation_importance(clf, Xte, yte, n_repeats=5, random_state=seed, n_jobs=-1)
    per_col = pd.DataFrame({"group": [col_group[c] for c in cols],
                            "importance": np.clip(imp.importances_mean, 0, None)})
    per_group = (per_col.groupby("group")["importance"].sum().to_frame("importance")
                 .assign(weight=lambda d: d.importance / d.importance.mean())
                 .sort_values("weight", ascending=False))
    print(per_group.round(3).to_string())
    return per_group


if __name__ == "__main__":
    outfield, _ = data.load()
    run(outfield)
