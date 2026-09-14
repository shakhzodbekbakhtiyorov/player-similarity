# Experiments

Scripts behind the claims in the main README. Each is standalone:

```bash
python -m experiments.weights_from_position
python -m experiments.weights_from_icc
python -m experiments.ab_weights
```

### `weights_from_position.py`
The first approach: learn group weights from permutation importance for
predicting a player's position. Kept because it is the **wrong question** —
within-position z-scoring has already removed the signal it measures, and
"which feature predicts position" has no per-position version.

### `weights_from_icc.py`
Weights from reliability: the share of a feature's variance that is between
players rather than within one player across clubs. Prints the per-group
relative ICC, the team-context drop, and the most "reliable" features — which
turn out to be height, weight and pace, because they never change. Reliability
is necessary for a style feature, not sufficient.

### `ab_weights.py`
The test that settled it. Weights fit on seasons ≤ 2223, scored on 2324/2425,
one index per position, run twice:

| arm | identity columns present | identity columns removed |
|---|---|---|
| hand priors | 0.2257 | 0.1771 |
| pure ICC | **0.3966** | 0.1771 |
| prior × ICC | 0.2680 | 0.1818 |

Pooled `recall_transfer`, n ≈ 630, so SE ≈ 0.015.

With height in the matrix, ICC weights look like a breakthrough. They aren't —
the engine became a height lookup and the metric rewarded it. Remove six FIFA
body columns and every arm is inside noise of every other.

Two conclusions: group weights are not the bottleneck, and same-player recall
cannot be used for model selection while identity features are in the matrix.
