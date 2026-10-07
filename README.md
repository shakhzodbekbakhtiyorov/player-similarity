# Player style similarity

Find footballers who **play like** a given player — not ones who are equally
good. Given "Rodri, 2022/23", it returns Højbjerg, Brozović, Fernandinho,
Locatelli, Henderson, and then explains what makes them alike in six named
dimensions rather than fifty raw columns.

Built on FBref per-90 stats joined to FIFA attributes: ~20,600 player-seasons
(after a minutes filter) from ten leagues — the European big five plus the
Eredivisie, Primeira Liga, Belgian Pro League, Süper Lig and MLS — 2017/18 to
2024/25. Data: [Football Player Data Set on Kaggle](https://www.kaggle.com/datasets/uday9081/football-plyer-data-set).

**[Try it →](https://shakhzodbekbakhtiyorov.github.io/player-similarity/)**

The demo runs entirely in your browser — the index ships as ~6 MB of typed
arrays and every query is a mat-vec in JavaScript. No server, so nothing to
cold-start.

---

## Run it

Python 3.10+.

```bash
pip install -r requirements.txt
python build.py      # builds data/processed/app_index.pkl, ~1 min
python app.py        # http://127.0.0.1:5001
```

`build.py` is the only slow step. Re-run it when the data or `config.py`
changes; `app.py` only reads the artifact it produces.

### Data

The source table is not committed. Download it from Kaggle:
**[Football Player Data Set](https://www.kaggle.com/datasets/uday9081/football-plyer-data-set)**
by uday9081 (Apache 2.0) — a single file, `fifa_fbref_merged.csv` (~30 MB).

```bash
# via the Kaggle CLI (needs an API token in ~/.kaggle/kaggle.json),
# or download the zip from the page and unzip it here
kaggle datasets download -d uday9081/football-plyer-data-set --unzip
```

Put `fifa_fbref_merged.csv` in the project root (`config.RAW_CSV`).

What is in it: 28,436 rows × 135 columns, one row per player-season-club.
FBref per-90 tables (standard, shooting, passing, possession, defensive actions,
goal/shot creation, keeping) joined to FIFA 18–25 player attributes (ratings,
position, height/weight, value, wage) on player and season. The merge was done
by the dataset author, not in this repo. Any table with the same columns works;
`config.py` lists exactly which ones are used.

Underlying sources: match statistics from [FBref](https://fbref.com) (Opta data),
player attributes from EA Sports FIFA. This project is not affiliated with
either. The demo in `docs/` publishes derived per-player scores, not the raw
table.

### Publishing the demo

```bash
python export_static.py        # writes docs/ — page + ~6 MB of typed arrays
git add docs && git commit -m "Update demo" && git push
```

Then in the repo: **Settings → Pages → Source: deploy from branch → main → /docs**.

`export_static.py` builds `docs/index.html` from `static/index.html`, swapping
the API calls for `static/demo.js`, so the hosted page and the local one share
their markup and cannot drift apart.

The browser reproduces the Python ranking: top-10 order is identical for every
query tested, and across 400 random queries the float32/float64 difference
(~5e-5) never once changed an ordering.

---

## How it works

### 1. Every stat becomes "how unusual is this, for his position"

A per-90 number means nothing on its own — three tackles is a lot for a winger
and nothing for a holding midfielder. Every feature is z-scored **within
position**, so 0 means "typical here" and +1 means "one standard deviation
above". Extremes are winsorised first and gaps filled with the position median.

47 features, grouped into 10 concepts in `config.py`. A group's weight is split
across its columns, so a 12-column concept does not quietly outweigh a 2-column
one. `overall`, `potential`, `value_eur` and `wage_eur` are held out of the
matrix entirely — they measure how good a player is, which is the thing this is
trying not to answer.

### 2. Neighbours are cosine distance in that space

Filtered to the same fine role (CB / FB / DM / CM / AM / W / ST, from FIFA's
primary position), with a minutes floor, excluding the player's own other
seasons. That is a single mat-vec against ~2,000 candidate rows: **2.5 ms**.

No approximate-nearest-neighbour index. At 20k rows, building one costs more
than it saves and gives up exact results.

### 3. The explanation comes from a different space

Six **style axes per position**, fitted separately for each of the seven roles
(`axes.py`):

- Regress out team context first — squad passing volume — so a Manchester City
  midfielder is not simply "high volume".
- Drop the six FIFA body attributes (height, weight, pace, physicality,
  dribbling, skill moves). They identify a body, not a style.
- Factor-analyse with a varimax rotation. Each axis is a weighted blend of all
  41 remaining stats: for central midfielders the biggest one loads +0.96 on
  total passing distance and −0.40 on progressive passes received — the
  deep-metronome-versus-box-crasher spectrum.
- Flip every axis so "more" is always the positive end, then name it from its
  own loadings (`labels.py`), so the label follows the maths on a rebuild
  instead of drifting away from it.

Retrieval and explanation deliberately use different spaces. Axis instability
is a reason not to *rank* on an axis; it is not a reason not to *describe* with
one. "Both of these seasons sit high on passing volume" is true of those
seasons either way.

### 4. Each axis is tested, and the result is shown to the user

For every axis: if a player changes club, does he keep his position on it?

```
ICC = variance between players / (between + within the same player across clubs)
```

Only players who actually moved, one row per club spell — two seasons at one
club share tactics, which would make a team detector look like a style
detector. The app prints the result on every row as *holds up* (≥0.45),
*partly holds* (0.30–0.45) or *season-specific* (<0.30).

---

## What the evaluation actually shows

`python evaluate.py` — two metrics, neither needing labels.

**Same-player recall.** A player is similar to himself. For everyone with two or
more seasons, does another of his own seasons appear in his top 10?

```
recall_all      = 0.343
recall_transfer = 0.142   (n = 11,843)
chance          = 0.0005
```

`recall_transfer` requires the retrieved season to be at a *different club*.
The gap between the two is team context, and it is large. 0.142 is ~300× chance
and still modest in absolute terms.

**Quality leak.** Neighbours are not systematically better rated than the
population (+0.06 SD), but neighbour quality does track query quality
(r = 0.71). Per-90 counting stats entangle style with level, and holding
`overall` out of the matrix does not fully separate them.

**Axis stability, by position:**

| role | mean axis ICC | best axis |
|---|---|---|
| CM | 0.45 | 0.60 |
| AM | 0.39 | 0.47 |
| ST | 0.34 | 0.48 |
| DM | 0.34 | 0.48 |
| W | 0.31 | 0.41 |
| FB | 0.31 | 0.40 |
| CB | 0.26 | 0.39 |

Midfield axes are worth reading. Centre-back and winger axes are interpretable
but mostly describe the season rather than the player — a centre-back's
counting stats are dominated by how much his team defends, and residualising on
squad passing volume only partly removes that.

Factor axes are consistently **8–21% more reliable** than the individual
features they are built from; averaging correlated stats cancels sampling noise.

---

## Things that did not work

Kept in [`experiments/`](experiments/) so they can be re-run rather than taken
on trust.

**Learned group weights do nothing.** Weights from position-prediction
importance, weights from reliability, hand-tuned weights, and dropping the
low-reliability groups all land within noise of each other on held-out seasons.

**Same-player recall is gameable.** With the FIFA body columns in the matrix,
ICC-derived weights nearly doubled `recall_transfer` — by weighting height at
2.3× and becoming a height lookup. Height is a perfect player fingerprint and
says nothing about how anyone plays. Remove six columns and the entire gain
disappears. The metric cannot select models while identity features are present.

**Finer position buckets are mostly an illusion.** Splitting DF into CB/FB and
MF into DM/CM/AM raises `recall_transfer` from 0.177 to 0.248 — but lift over
chance *falls*, because the candidate pool shrank. Better for the interface,
not a better representation.

---

## Limits

- FBref per-90s are counting stats: **what** a player does and **where**, never
  **how**. No off-ball positioning, no pressing triggers, no body orientation,
  no timing of runs. "Deep distributor vs. box-crasher" is recoverable; the
  things that separate two deep distributors are not.
- Team control is one variable and a straight line. It catches "my team passes
  a lot" and misses "my team presses high" or "sits deep".
- Roles come from FIFA's primary position that season, so a player can change
  role between seasons — De Bruyne is AM in some years and CM in others. Pairs
  in different roles get no axis panel.
- There is no ground truth for "similar style", so nothing here proves the axes
  match what a coach means. The honest claim is three negatives: stable across a
  club change, not explained by team volume, not a proxy for `overall`.

---

## Layout

```
config.py     feature groups, weights, row filters
data.py       load, clean, dedupe, position and role labels
features.py   scaling, weighting, team-context residualisation
axes.py       per-role factor axes + the cross-club ICC test
labels.py     column and axis names in English
build.py      bakes everything into one pickle
app.py        Flask API over that pickle
evaluate.py   recall and quality-leak metrics
export_static.py  exports the index as typed arrays + a self-contained page
static/       the page, and the browser build of the same logic
docs/         generated: what GitHub Pages serves
experiments/  the negative results, reproducible
```

---

## License

Code is released under the [MIT License](LICENSE). The dataset is not part of
this repository and keeps its own terms (Apache 2.0 on Kaggle; see
[Data](#data) for the underlying sources).
