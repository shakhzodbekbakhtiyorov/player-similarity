"""Export the index as static files so the demo runs entirely in the browser.

    python build.py && python export_static.py

Writes to docs/, which GitHub Pages can serve directly. No server: every query
is one mat-vec against a 20,625 x 47 matrix, which JavaScript does instantly.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np

ARTIFACT = "data/processed/app_index.pkl"
OUT = "docs/data"

# Z is only ever displayed, so 1/25 of a standard deviation is plenty of
# precision and costs a quarter of the bytes. X stays float32 so the browser
# reproduces the Python ranking exactly.
Z_SCALE = 25.0


def main():
    with open(ARTIFACT, "rb") as fh:
        art = pickle.load(fh)
    os.makedirs(OUT, exist_ok=True)

    meta, X = art["meta"], art["X"].astype("float32")
    XN = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    XN.astype("<f4").tofile(f"{OUT}/index.bin")

    z = np.clip(np.round(art["Z"] * Z_SCALE), -127, 127).astype("<i1")
    z.tofile(f"{OUT}/z.bin")

    axis = np.nan_to_num(art["axis_scores"], nan=-999).astype("<f4")
    axis.tofile(f"{OUT}/axes.bin")

    # intern the repeated strings; names dominate the payload otherwise
    def codes(values):
        uniq = sorted(set(values))
        lookup = {v: i for i, v in enumerate(uniq)}
        return uniq, [lookup[v] for v in values]

    teams, team_id = codes(meta["team"].astype(str).tolist())
    roles, role_id = codes(meta["role"].astype(str).tolist())
    buckets, bucket_id = codes(meta["bucket"].astype(str).tolist())

    players = {
        "n": int(len(meta)),
        "player": meta["player"].astype(str).tolist(),
        "season": meta["season"].astype(str).tolist(),
        "teams": teams, "team_id": team_id,
        "roles": roles, "role_id": role_id,
        "buckets": buckets, "bucket_id": bucket_id,
        "nineties": [round(float(v), 1) for v in meta["Playing Time_90s"]],
    }
    with open(f"{OUT}/players.json", "w") as fh:
        json.dump(players, fh, separators=(",", ":"))

    schema = {
        "n_rows": int(len(meta)),
        "n_cols": int(X.shape[1]),
        "n_axes": len(art["axis_cols"]),
        "z_scale": Z_SCALE,
        "columns": [art["pretty"][c] for c in art["columns"]],
        "col_group": [g.replace("_", " ") for g in art["col_group"]],
        "style_idx": [n for n, c in enumerate(art["columns"])
                      if c not in set(art["identity_cols"])],
        "axes": art["axes_meta"],
        "seasons": sorted(set(meta["season"].astype(str))),
    }
    with open(f"{OUT}/schema.json", "w") as fh:
        json.dump(schema, fh, separators=(",", ":"))

    # Generate the page from the same markup the Flask app serves, swapping the
    # API calls for static/demo.js. One source of layout, so the two cannot drift.
    shell = open("static/index.html").read()
    head = shell[:shell.index("<script>")].replace(
        '<p class="sub">Neighbours are ranked',
        '<p class="sub" id="loading">Loading the index (~6 MB) &hellip;</p>\n'
        '  <p class="sub">Neighbours are ranked')
    with open("docs/index.html", "w") as fh:
        fh.write(head + "<script>\n" + open("static/demo.js").read()
                 + "</script>\n</body>\n</html>\n")

    total = sum(os.path.getsize(f"{OUT}/{f}") for f in os.listdir(OUT))
    print(f"[export] docs/index.html + {OUT}/")
    for f in sorted(os.listdir(OUT)):
        print(f"   {f:<14} {os.path.getsize(f'{OUT}/{f}') / 1e6:6.2f} MB")
    print(f"   {'total':<14} {total / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
