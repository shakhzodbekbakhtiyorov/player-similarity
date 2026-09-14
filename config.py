"""Every knob in one place: what counts as a feature, and how much."""

from pathlib import Path

RAW_CSV = Path("fifa_fbref_merged.csv")

# Row filtering
MIN_90S = 8.0                       # below ~720 minutes, per-90 rates are noise
DEDUPE_KEYS = ["player", "team", "season"]
DEDUPE_PREFER = "Playing Time_Min"  # on a duplicate, keep the row with most minutes
DEFAULT_BUCKET = "MF"
CLIP_QUANTILES = (0.01, 0.99)       # winsorise before scaling

# These encode how GOOD or how expensive a player is, not how he plays. They are
# carried as metadata for display and evaluation, never as features.
LEAKAGE_COLS = [
    "overall", "potential", "value_eur", "wage_eur", "international_reputation",
]

META_COLS = [
    "player", "short_name", "season", "fifa_version", "team", "club_name",
    "bucket", "pos_primary", "role",
    "league", "league_name", "nation", "nationality_name", "pos",
    "player_positions", "age_fbref", "age_fifa", "height_cm", "weight_kg",
    "preferred_foot", "Playing Time_MP", "Playing Time_Min", "Playing Time_90s",
] + LEAKAGE_COLS

# Grouping does two jobs: it lets you weight a concept instead of 60 columns, and
# it kills block-size bias, since a group's weight is split across its columns.

OUTFIELD_GROUPS = {
    "finishing": [
        "Per 90 Minutes_npxG",
        "Standard_Sh/90",
        "Standard_SoT/90",
        "Standard_SoT%",
        "Expected_npxG/Sh",
        "Standard_Dist",
    ],
    "creation": [
        "Per 90 Minutes_xAG",
        "Per 90 Minutes_KP",
        "SCA_SCA90",
        "GCA_GCA90",
        "Per 90 Minutes_PPA",
        "Per 90 Minutes_CrsPA",
        "Per 90 Minutes_Expected_xA",
    ],
    "passing_volume": [
        "Per 90 Minutes_Total_Att",
        "Per 90 Minutes_Total_TotDist",
        "Per 90 Minutes_Short_Att",
        "Per 90 Minutes_Medium_Att",
    ],
    "passing_risk": [
        "Total_Cmp%",
        "Short_Cmp%",
        "Medium_Cmp%",
        "Long_Cmp%",
        "Per 90 Minutes_Long_Att",
        "Per 90 Minutes_Total_PrgDist",
        "Per 90 Minutes_PrgP",
        "Per 90 Minutes_1/3",
    ],
    "carrying": [
        "Per 90 Minutes_PrgC",
        "Per 90 Minutes_PrgR",
        "Per 90 Minutes_SCA Types_TO",
        "Per 90 Minutes_SCA Types_Fld",
        "skill_moves",
        "dribbling",
    ],
    "defending_volume": [
        "Per 90 Minutes_Tackles_Tkl",
        "Per 90 Minutes_Int",
        "Per 90 Minutes_Blocks_Blocks",
        "Per 90 Minutes_Clr",
    ],
    "defending_quality": [
        "Challenges_Tkl%",
        "Per 90 Minutes_Challenges_Lost",
        "Per 90 Minutes_Err",
    ],
    "defending_zone": [
        "Per 90 Minutes_Tackles_Def 3rd",
        "Per 90 Minutes_Tackles_Mid 3rd",
        "Per 90 Minutes_Tackles_Att 3rd",
    ],
    "discipline": [
        "Per 90 Minutes_CrdY",
        "Per 90 Minutes_CrdR",
    ],
    "physical": [
        "height_cm",
        "weight_kg",
        "pace",
        "physic",
    ],
}

# The dial you actually turn. physical is held low on purpose: left
# alone, height drives the results.
OUTFIELD_WEIGHTS = {
    "finishing": 1.0,
    "creation": 1.3,
    "passing_volume": 1.0,
    "passing_risk": 1.1,
    "carrying": 1.2,
    "defending_volume": 1.0,
    "defending_quality": 0.6,
    "defending_zone": 0.8,
    "discipline": 0.3,
    "physical": 0.15,
}

GK_GROUPS = {
    "shot_stopping": [
        "Performance_Save%",
        "Per 90 Minutes_Performance_Saves",
        "Per 90 Minutes_Performance_SoTA",
        "Performance_GA90",
        "goalkeeping_reflexes",
        "goalkeeping_diving",
    ],
    "handling": [
        "goalkeeping_handling",
        "goalkeeping_positioning",
    ],
    "distribution": [
        "goalkeeping_kicking",
        "Per 90 Minutes_Total_Att",
        "Per 90 Minutes_Long_Att",
        "Long_Cmp%",
        "Per 90 Minutes_Total_PrgDist",
    ],
    "sweeping": [
        "goalkeeping_speed",
        "Per 90 Minutes_Tackles_Def 3rd",
    ],
    "penalties": [
        "Penalty Kicks_Save%",
        "Per 90 Minutes_Penalty Kicks_PKA",
    ],
    "physical": [
        "height_cm",
        "weight_kg",
    ],
}

GK_WEIGHTS = {
    "shot_stopping": 1.5,
    "handling": 1.0,
    "distribution": 1.2,
    "sweeping": 0.8,
    "penalties": 0.4,
    "physical": 0.2,
}
