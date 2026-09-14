"""Build the single artifact the web app loads. Run this before app.py."""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

import axes
import config
import data
import features
import labels

ARTIFACT = "data/processed/app_index.pkl"


def plain(obj):
    """
    Strip pandas extension dtypes out of the artifact.

    pandas 3.0 makes `str` the default text dtype and it is pyarrow-backed when
    pyarrow is installed -- for column values, and for the columns Index. An
    artifact built with it loads on the build machine and raises
    ModuleNotFoundError anywhere without pyarrow.
    """
    if isinstance(obj, pd.DataFrame):
        out = obj.copy()
        out.columns = pd.Index([str(c) for c in out.columns], dtype=object)
        out.index = pd.RangeIndex(len(out))
        for c in out.columns:
            if pd.api.types.is_extension_array_dtype(out[c].dtype):
                out[c] = out[c].astype(object)
        return out
    if isinstance(obj, pd.Series):
        out = (obj.astype(object)
               if pd.api.types.is_extension_array_dtype(obj.dtype) else obj.copy())
        out.index = pd.RangeIndex(len(out))
        return out
    if isinstance(obj, pd.Index):
        return pd.Index([str(v) for v in obj], dtype=object)
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, str):
        return str(obj)
    return obj


def axis_specs(loadings, table, col_group) -> list[dict]:
    specs = []
    for f in loadings.columns:
        icc = table.at[f, "icc_cross_club"]
        specs.append({
            "id": f,
            **labels.axis_spec(loadings[f], col_group),
            "var_share": round(float(table.at[f, "var_share"]), 4),
            "icc_cross_club": None if not np.isfinite(icc) else round(float(icc), 3),
        })

    # two factors in one role can land on the same concept; identical titles in
    # the panel look like a bug
    seen = pd.Series([s["title"] for s in specs])
    for s in specs:
        if s["title"] in set(seen[seen.duplicated(keep=False)]) and s["high"]:
            s["title"] = f"{s['title']} ({s['high'][0]})"
    return specs


def main():
    outfield, _ = data.load()

    meta_cols = [c for c in config.META_COLS if c in outfield.columns]
    if "role" not in meta_cols:
        meta_cols.append("role")
    fm = features.build(outfield, config.OUTFIELD_GROUPS, config.OUTFIELD_WEIGHTS,
                        meta_cols=meta_cols)

    fitted = axes.fit(outfield)
    axis_cols = [f"F{i + 1}" for i in range(axes.N_AXES)]
    axes_meta, score_frames = {}, []

    for role, (d, loadings, scores, table, col_group) in fitted.items():
        axes_meta[role] = {"axes": axis_specs(loadings, table, col_group),
                           "n_movers": int(table["n_movers"].max())}
        z = (scores - scores.mean()) / (scores.std(ddof=0) + 1e-9)
        z.insert(0, "player_key", d["player_key"].to_numpy())
        score_frames.append(z)

    scores = pd.concat(score_frames, ignore_index=True).set_index("player_key")
    aligned = scores.reindex(fm.meta["player_key"].to_numpy())

    art = plain({
        "X": fm.X.astype("float32"),
        "Z": fm.Z.astype("float32"),
        "columns": fm.columns,
        "col_group": fm.col_group,
        "col_weight": fm.col_weight,
        "meta": fm.meta,
        "axis_cols": axis_cols,
        "axis_scores": aligned[axis_cols].to_numpy(dtype="float32"),
        "axes_meta": axes_meta,
        "pretty": {c: labels.feature(c) for c in fm.columns},
        "identity_cols": sorted(c for c in fm.columns if c in data.IDENTITY_COLS),
    })

    os.makedirs(os.path.dirname(ARTIFACT), exist_ok=True)
    blob = pickle.dumps(art, protocol=4)
    if b"Arrow" in blob:
        bad = [k for k, v in art.items() if b"Arrow" in pickle.dumps(v, protocol=4)]
        raise RuntimeError(f"pyarrow-backed array under {bad}; extend plain()")
    with open(ARTIFACT, "wb") as fh:
        fh.write(blob)

    print(f"\n[build] pandas {pd.__version__}, text dtype '{pd.Series(['x']).dtype}'")
    print(f"[build] {ARTIFACT}: {len(fm.meta):,} player-seasons, "
          f"{len(fm.columns)} features, {len(blob) / 1e6:.1f} MB")
    print(f"[build] axis models: {list(axes_meta)}")


if __name__ == "__main__":
    main()
