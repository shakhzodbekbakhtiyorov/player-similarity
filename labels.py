"""Turning column names and factor loadings into English."""

from __future__ import annotations

FEATURE = {
    "Per 90 Minutes_npxG": "non-penalty xG",
    "Standard_Sh/90": "shots",
    "Standard_SoT/90": "shots on target",
    "Standard_SoT%": "shot accuracy",
    "Expected_npxG/Sh": "chance quality per shot",
    "Standard_Dist": "average shot distance",
    "Per 90 Minutes_xAG": "expected assists",
    "Per 90 Minutes_KP": "key passes",
    "SCA_SCA90": "shot-creating actions",
    "GCA_GCA90": "goal-creating actions",
    "Per 90 Minutes_PPA": "passes into the box",
    "Per 90 Minutes_CrsPA": "crosses into the box",
    "Per 90 Minutes_Expected_xA": "expected assists (xA)",
    "Per 90 Minutes_Total_Att": "passes attempted",
    "Per 90 Minutes_Total_TotDist": "total passing distance",
    "Per 90 Minutes_Short_Att": "short passes",
    "Per 90 Minutes_Medium_Att": "medium passes",
    "Total_Cmp%": "pass completion",
    "Short_Cmp%": "short-pass completion",
    "Medium_Cmp%": "medium-pass completion",
    "Long_Cmp%": "long-pass completion",
    "Per 90 Minutes_Long_Att": "long passes",
    "Per 90 Minutes_Total_PrgDist": "progressive passing distance",
    "Per 90 Minutes_PrgP": "progressive passes",
    "Per 90 Minutes_1/3": "passes into the final third",
    "Per 90 Minutes_PrgC": "progressive carries",
    "Per 90 Minutes_PrgR": "progressive passes received",
    "Per 90 Minutes_SCA Types_TO": "take-ons leading to a shot",
    "Per 90 Minutes_SCA Types_Fld": "fouls won leading to a shot",
    "Per 90 Minutes_Tackles_Tkl": "tackles",
    "Per 90 Minutes_Int": "interceptions",
    "Per 90 Minutes_Blocks_Blocks": "blocks",
    "Per 90 Minutes_Clr": "clearances",
    "Challenges_Tkl%": "duels won",
    "Per 90 Minutes_Challenges_Lost": "duels lost",
    "Per 90 Minutes_Err": "errors leading to a shot",
    "Per 90 Minutes_Tackles_Def 3rd": "tackles in his own third",
    "Per 90 Minutes_Tackles_Mid 3rd": "tackles in midfield",
    "Per 90 Minutes_Tackles_Att 3rd": "tackles high up the pitch",
    "Per 90 Minutes_CrdY": "yellow cards",
    "Per 90 Minutes_CrdR": "red cards",
    "height_cm": "height", "weight_kg": "weight", "pace": "pace",
    "physic": "physicality", "dribbling": "dribbling rating",
    "skill_moves": "skill moves",
}

CONCEPT = {
    "finishing": "Shooting",
    "creation": "Chance creation",
    "passing_volume": "Passing volume",
    "passing_risk": "Passing ambition",
    "carrying": "Carrying & running",
    "defending_volume": "Defensive actions",
    "defending_quality": "Duels",
    "defending_zone": "Where he tackles",
    "discipline": "Cards",
    "physical": "Physique",
}

# columns that mean something sharper than their group name
CONCEPT_OVERRIDE = {
    "Per 90 Minutes_Tackles_Tkl": "Tackling",
    "Per 90 Minutes_Tackles_Def 3rd": "Tackling deep",
    "Per 90 Minutes_Tackles_Att 3rd": "Pressing high",
    "Total_Cmp%": "Pass accuracy", "Short_Cmp%": "Pass accuracy",
    "Medium_Cmp%": "Pass accuracy", "Long_Cmp%": "Pass accuracy",
    "Expected_npxG/Sh": "Shot quality",
    "Standard_Dist": "Shot distance",
    "Per 90 Minutes_PrgR": "Running in behind",
    "Per 90 Minutes_Clr": "Clearing danger",
}


def feature(col: str) -> str:
    return FEATURE.get(col, col)


def concept(col: str, group: str) -> str:
    return CONCEPT_OVERRIDE.get(col) or CONCEPT.get(group, group.replace("_", " "))


def orient(loadings, scores):
    """Flip an axis so its dominant concept is the positive end. Otherwise half
    the axes read 'shooting down', which everyone misreads."""
    pos = loadings[loadings > 0].abs().sum()
    neg = loadings[loadings < 0].abs().sum()
    return (-loadings, -scores) if neg > pos else (loadings, scores)


def axis_spec(loadings, col_group: dict, thresh=0.35) -> dict:
    """
    Name an axis from its loadings, so the label follows the maths on a rebuild
    instead of drifting away from it. The group with the most loading mass wins
    each end -- a plain count would hand the name to whichever group has the
    most columns in config.
    """
    ordered = loadings.sort_values()
    pos = ordered[ordered >= thresh].sort_values(ascending=False)
    neg = ordered[ordered <= -thresh]

    def end(v):
        if not len(v):
            return None, []
        mass = v.abs().groupby([col_group[c] for c in v.index]).sum()
        return (concept(v.index[0], mass.idxmax()),
                [feature(c) for c in v.index[:4]])

    high_name, high = end(pos)
    low_name, low = end(neg)

    if high_name and low_name and high_name == low_name:
        title = f"{high[0]} ↑ / {low[0]} ↓"      # same concept at both ends
    elif high_name and low_name:
        title = f"{high_name} vs {low_name}"
    else:
        title = high_name or low_name or "Mixed"
    return {"title": title, "high": high, "low": low}


def level(z: float) -> str:
    if z >= 1.25:
        return "far above average"
    if z >= 0.45:
        return "above average"
    if z > -0.45:
        return "around average"
    if z > -1.25:
        return "below average"
    return "far below average"


def phrase(a: float, b: float, name_a: str, name_b: str, role: str,
           agree=0.45) -> str:
    """
    One sentence instead of a gap in standard deviations. The bands decide the
    shape: two players 0.6 apart can still both be 'around average', and calling
    one clearly higher would contradict the levels printed next to it.
    """
    short_a, short_b = name_a.split()[-1], name_b.split()[-1]
    la, lb, gap = level(a), level(b), abs(a - b)

    if la == lb:
        base = (f"Both sit close to the average {role} here."
                if la == "around average" else f"Both are {la} for a {role} here.")
        return base + (f" {short_a if a > b else short_b} a little more so."
                       if gap > agree else "")
    if gap <= agree:
        return f"Very close — on the line between {lb} and {la} for a {role}."

    hi, lo = (short_a, short_b) if a > b else (short_b, short_a)
    word = "clearly" if gap >= 1.0 else "somewhat"
    return (f"{hi} is {word} higher than {lo} — "
            f"{level(max(a, b))} vs {level(min(a, b))} for a {role}.")


def confidence(icc: float | None) -> tuple[str, str]:
    if icc is None:
        return "unknown", "no cross-club estimate for this axis"
    if icc >= 0.45:
        return "holds up", "players tend to keep this trait after a transfer"
    if icc >= 0.30:
        return "partly holds", "about a third of this survives a transfer"
    return "season-specific", "this mostly describes the season, not the player"
