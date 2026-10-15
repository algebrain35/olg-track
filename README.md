# OLG Player Risk Pipeline

ThunderHacks 2026 · OLG "Intelligent Player Protection & Fraud Detection" challenge.

A pipeline that flags players for **anti-money laundering (AML)** and **safer-gambling (harm)** review, explains every flag in plain English, and gives investigators a prioritised queue. OLG provides no data, so the project includes a calibrated synthetic data generator with labelled scenarios to test against.

## How it works

```
simulate.py  →  features.py  →  score.py  →  app.py
synthetic DB     59 features     scores +      Streamlit
(+ labels)       per player      reasons       dashboard
```

1. **Simulate**: 5,000 patrons over 180 days. Normal play is calibrated to published gambling research, then harm escalation, 10 AML typologies (structuring, minimal play, mule rings, chip dumping, …) and legitimate-but-unusual "hard negatives" are injected.
2. **Features**: affordability vs income band, comparison with similar players, change vs the player's own baseline, location, money flow, threshold behaviour, network links and harm markers. Reads observable tables only.
3. **Score**: each player gets an AML score and a harm score (0–100) from a transparent 28-rule scorecard, plus a LightGBM model confidence. A lens is routed to its team at 45 scorecard points or 70% model confidence.
4. **Dashboard**: priority queue, Player 360 case view, linked-player networks, the full rule set, and validation against the synthetic scenarios.

## Quick start

Requires Python 3.10+.

```bash
pip install -r requirements.txt

# data to score, and a separate labelled run to train the model on
python simulate.py --db synth.db
python simulate.py --seed 7 --db train.db
python features.py --db synth.db
python features.py --db train.db

python score.py --db synth.db --train-db train.db --eval
streamlit run app.py -- --db synth.db
```

Without `--train-db`, scoring uses the scorecard only.

## Project layout

| Path | Purpose |
| --- | --- |
| `simulate.py` | Command-line entry point for the generator |
| `olgsim/` | Generator package: `config.py` (all parameters), `models.py`, `reference.py`, `factory.py`, `behaviours.py`, `scenarios.py`, `engine.py`, `db.py`, `calibration.py` |
| `features.py` | Builds `player_features` and `feature_meta` |
| `score.py` | Builds `risk_scores`, `risk_reasons`, `scorecard_rules` |
| `app.py` | Streamlit dashboard; writes investigator decisions to `case_actions` |
| `.streamlit/config.toml` | Dashboard theme |
| `OLG_Risk_Pipeline_Specifications.pdf` | Full specifications: simulator, features, scoring, dashboard |

## Command-line options

| Script | Option | Default | Meaning |
| --- | --- | --- | --- |
| `simulate.py` | `--players` / `--days` | 5000 / 180 | Population and window |
| | `--seed` | 42 | Reproducible run |
| | `--start` | 2026-01-05 | First day |
| | `--aml-scale` | 1.0 | Multiply laundering prevalence (robustness tests) |
| `features.py` | `--as-of` | day after last transaction | Scoring date |
| | `--recent-days` | 30 | "Recent" window for change features |
| | `--check` | off | Print feature separation vs labels |
| `score.py` | `--train-db` | none | Labelled run for the model layer |
| | `--eval` | off | Print precision/recall vs labels |
| | `--capacity` | 50 | Cases per lens for precision@k |
| all | `--db` | synth.db | SQLite file |

## Results (synthetic data)

Main run (seed 42) / held-out run (seed 11, half the laundering prevalence); model trained on seed 7.

| Lens | Signal | Precision | Recall |
| --- | --- | --- | --- |
| AML | Scorecard | 0.88 / 0.82 | 0.84 / 0.91 |
| AML | Model | 1.00 / 1.00 | 0.71 / 0.86 |
| Harm | Scorecard | 0.96 / 1.00 | 0.25 / 0.29 |
| Harm | Model | 0.95 / 0.86 | 0.69 / 0.66 |

No hard negatives were routed on the main run (1% on held-out). These figures measure how well the pipeline finds the simulator's patterns, not real-world performance; laundering behaviour is designed from FATF and FINTRAC typologies because no public casino laundering data exists.

## Demo players

- **P004210**: heavy harm player whose escalating spend also trips AML rules; routed to both teams.
- **P003003**: member of a 7-account mule ring (see the Network page).

## Notes

- `labels` and `truth_players` are ground truth: only `--check`, `--eval` and the dashboard's Validation page read them.
- Demographics only choose each player's peer group; they never add risk points.
- Re-run `features.py` after updating it; `score.py` stops with a message if expected features are missing.
- Regenerating a database with `simulate.py` erases its case decisions.
