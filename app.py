"""Flask API and the single page over the precomputed index.

    GET /api/search?q=              resolve a name to player-seasons
    GET /api/similar?key=&k=        neighbours, with filters
    GET /api/explain?a=&b=          why those two came out close
    GET /api/meta                   roles, axes, seasons
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

import labels

ARTIFACT = "data/processed/app_index.pkl"
PORT = 5001

app = Flask(__name__, static_folder="static", static_url_path="")

with open(ARTIFACT, "rb") as fh:
    ART = pickle.load(fh)

META = ART["meta"]
X, Z = ART["X"], ART["Z"]
XN = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)   # cosine, once
AXIS, AXES_META = ART["axis_scores"], ART["axes_meta"]
COLUMNS, COL_GROUP, PRETTY = ART["columns"], ART["col_group"], ART["pretty"]

# Identity columns stay in the distance but out of the feature lists: "alike
# because both weigh 72kg" is not an answer to why they play the same way.
STYLE_IDX = [n for n, c in enumerate(COLUMNS) if c not in set(ART["identity_cols"])]

ROW = pd.Series(np.arange(len(META)), index=META["player_key"].to_numpy())
MINS = pd.to_numeric(META["Playing Time_90s"], errors="coerce").fillna(0).to_numpy()
SEASON = META["season"].astype(str).to_numpy()
ROLE = META["role"].astype(str).to_numpy()
BUCKET = META["bucket"].astype(str).to_numpy()
PLAYER = META["player"].astype(str).to_numpy()
SEARCH = np.char.lower(META["player_key"].astype(str).to_numpy().astype("U"))


def row_of(key: str) -> int:
    i = ROW.get(key)
    if i is None:
        raise KeyError(key)
    return int(i)


def card(i: int) -> dict:
    m = META.iloc[i]
    return {
        "key": m["player_key"], "player": m["player"], "team": m.get("team"),
        "season": str(m.get("season")), "role": m.get("role"),
        "bucket": m.get("bucket"), "pos": m.get("pos"),
        "league": m.get("league") or m.get("league_name"),
        "nineties": round(float(MINS[i]), 1),
        "age": None if pd.isna(m.get("age_fbref")) else int(float(m.get("age_fbref"))),
    }


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/meta")
def api_meta():
    return jsonify({
        "n_rows": int(len(META)),
        "seasons": sorted(set(SEASON.tolist())),
        "roles": sorted(r for r in set(ROLE.tolist()) if r != "OTHER"),
        "axes": AXES_META,
    })


@app.get("/api/search")
def api_search():
    q = (request.args.get("q") or "").strip().lower()
    if len(q) < 2:
        return jsonify({"results": []})
    limit = min(int(request.args.get("limit", 25)), 100)
    hit = np.flatnonzero(np.char.find(SEARCH, q) >= 0)
    hit = hit[np.argsort(-MINS[hit])][:limit]        # the row meant is usually the one played
    return jsonify({"results": [card(int(i)) for i in hit]})


@app.get("/api/similar")
def api_similar():
    try:
        i = row_of(request.args.get("key", ""))
    except KeyError:
        return jsonify({"error": "unknown player_key"}), 404

    k = min(int(request.args.get("k", 10)), 50)
    scope = request.args.get("scope", "role")
    season = request.args.get("season") or None

    mask = np.ones(len(META), dtype=bool)
    mask[i] = False
    mask &= PLAYER != PLAYER[i]                       # not his own other seasons
    if scope == "role" and ROLE[i] != "OTHER":
        mask &= ROLE == ROLE[i]
    else:
        mask &= BUCKET == BUCKET[i]
    mask &= MINS >= float(request.args.get("min90s", 8))
    if season:
        mask &= SEASON == season

    cand = np.flatnonzero(mask)
    if not len(cand):
        return jsonify({"query": card(i), "results": [], "note": "no candidates"})

    sim = XN[cand] @ XN[i]
    top = np.argsort(-sim)[:k]
    results = [dict(card(int(cand[t])), rank=n + 1, similarity=round(float(sim[t]), 4))
               for n, t in enumerate(top)]
    return jsonify({"query": card(i), "scope": scope,
                    "n_candidates": int(len(cand)), "results": results})


@app.get("/api/explain")
def api_explain():
    try:
        i, j = row_of(request.args.get("a", "")), row_of(request.args.get("b", ""))
    except KeyError:
        return jsonify({"error": "unknown player_key"}), 404

    role = ROLE[i]
    axes_out = []
    if role == ROLE[j] and role in AXES_META and not np.isnan(AXIS[i, 0]):
        na, nb = META.at[i, "player"], META.at[j, "player"]
        for n, spec in enumerate(AXES_META[role]["axes"]):
            a, b = float(AXIS[i, n]), float(AXIS[j, n])
            conf, note = labels.confidence(spec.get("icc_cross_club"))
            axes_out.append({
                **spec, "a": round(a, 2), "b": round(b, 2), "gap": round(abs(a - b), 2),
                "a_level": labels.level(a), "b_level": labels.level(b),
                "sentence": labels.phrase(a, b, na, nb, role),
                "confidence": conf, "confidence_note": note,
            })
        axes_out.sort(key=lambda d: d["gap"])

    contrib = (X[i] - X[j]) ** 2
    total = float(contrib.sum()) + 1e-12
    gap = np.abs(Z[i] - Z[j])
    shared = (Z[i] + Z[j]) / 2.0

    # Ranking agreement by smallest gap fills the list with ties on near-constant
    # columns -- 88% of attacking midfielders share the same red-card value.
    # Agreement only means something at a value that is unusual for the position.
    alike = sorted(STYLE_IDX, key=lambda n: gap[n] - abs(shared[n]))
    apart = sorted(STYLE_IDX, key=lambda n: -gap[n])

    def feat(n: int) -> dict:
        return {"feature": PRETTY[COLUMNS[n]], "group": COL_GROUP[n].replace("_", " "),
                "z_a": round(float(Z[i, n]), 2), "z_b": round(float(Z[j, n]), 2),
                "gap": round(float(gap[n]), 2), "share": round(float(contrib[n] / total), 4)}

    per_group = pd.Series(contrib).groupby(COL_GROUP).sum() / total
    return jsonify({
        "a": card(i), "b": card(j),
        "similarity": round(float(XN[i] @ XN[j]), 4),
        "role_axes": axes_out, "axes_available": bool(axes_out),
        "by_group": [{"group": g, "share": round(float(v), 4)}
                     for g, v in per_group.sort_values(ascending=False).items()],
        "most_alike": [feat(int(n)) for n in alike[:6]],
        "most_different": [feat(int(n)) for n in apart[:6]],
    })


if __name__ == "__main__":
    print(f"[app] {len(META):,} player-seasons, {X.shape[1]} features")
    app.run(host="127.0.0.1", port=PORT, debug=False)
