"""Loading, cleaning and position labelling."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

# fbref's coarse position -> our bucket
POS_BUCKET = {
    "GK": "GK",
    "DF": "DF", "CB": "DF", "LB": "DF", "RB": "DF", "WB": "DF",
    "MF": "MF", "DM": "MF", "CM": "MF", "AM": "MF",
    "FW": "FW", "LW": "FW", "RW": "FW", "ST": "FW",
}

# FIFA's primary position -> the seven fine roles the axes are fitted on
ROLE_MAP = {
    "CB": "CB",
    "RB": "FB", "LB": "FB", "RWB": "FB", "LWB": "FB",
    "CDM": "DM", "CM": "CM", "CAM": "AM",
    "RM": "W", "LM": "W", "RW": "W", "LW": "W",
    "ST": "ST", "CF": "ST",
}
ROLES = ["CB", "FB", "DM", "CM", "AM", "W", "ST"]

# FIFA ratings that barely move between seasons. They identify a body, not a
# playing style, so the axes are fitted without them.
IDENTITY_COLS = {
    "height_cm", "weight_kg", "pace", "physic", "dribbling", "skill_moves",
}


def load_raw(path=None) -> pd.DataFrame:
    path = path or config.RAW_CSV
    df = pd.read_csv(path, low_memory=False)
    print(f"[data] {len(df):,} rows x {df.shape[1]} cols from {path}")

    # A few per-90 columns carry +inf. Left in place they poison every
    # team-season mean and any least-squares fit that touches such a row.
    num = df.select_dtypes(include=[np.number]).columns
    n_inf = int(np.isinf(df[num].to_numpy(dtype="float64", na_value=np.nan)).sum())
    if n_inf:
        df[num] = df[num].replace([np.inf, -np.inf], np.nan)
        print(f"[data] replaced {n_inf} inf cells with NaN")
    return df


def add_positions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    primary = (df["pos"].astype("string").fillna("")
                 .str.split(",").str[0].str.strip().str.upper())
    df["pos_primary"] = primary
    df["bucket"] = primary.map(POS_BUCKET).fillna(config.DEFAULT_BUCKET)

    fifa = (df["player_positions"].astype("string").fillna("")
              .str.split(",").str[0].str.strip().str.upper())
    df["role"] = fifa.map(ROLE_MAP).fillna("OTHER")
    return df


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """One fbref season can match two fifa_version rows. Keep the most minutes."""
    sort_cols = [c for c in (config.DEDUPE_PREFER, "fifa_version") if c in df.columns]
    before = len(df)
    df = (df.sort_values(sort_cols, ascending=[False] + [True] * (len(sort_cols) - 1),
                         na_position="last")
            .drop_duplicates(config.DEDUPE_KEYS, keep="first")
            .reset_index(drop=True))
    print(f"[data] dropped {before - len(df)} duplicate rows")
    return df


def add_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["player_key"] = (df["player"].astype("string").str.strip()
                        + " (" + df["season"].astype(str)
                        + ", " + df["team"].astype("string").str.strip() + ")")
    return df


def load(path=None):
    """Returns (outfield, goalkeepers). Keepers share almost no columns with
    outfielders, so they never go in the same matrix."""
    df = add_key(dedupe(add_positions(load_raw(path))))

    keep = df["Playing Time_90s"].astype(float) >= config.MIN_90S
    print(f"[data] keeping {keep.sum():,}/{len(df):,} rows with >= {config.MIN_90S} 90s")
    df = df.loc[keep].reset_index(drop=True)

    gk = df[df["bucket"] == "GK"].reset_index(drop=True)
    out = df[df["bucket"] != "GK"].reset_index(drop=True)
    print(f"[data] outfield={len(out):,} gk={len(gk):,}")
    return out, gk


if __name__ == "__main__":
    outfield, gk = load()
    print(outfield[["player_key", "bucket", "role"]].head())
