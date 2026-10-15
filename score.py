#!/usr/bin/env python3
"""Scores players on two lenses (AML, harm): a transparent points scorecard plus an
optional LightGBM model confidence trained on a separate labelled run.

    python score.py --db synth.db [--train-db train.db] [--eval] [--capacity 50]
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from features import FEATURES

FLAG = 45                                     # scorecard points that route a lens to its team
MODEL_FLAG = 70                               # model confidence that routes on its own
TIERS = [(70, "Critical"), (45, "High"), (25, "Medium"), (0, "Low")]


@dataclass(frozen=True)
class Rule:
    name: str
    lens: str
    feature: str
    lo: float            # no points at or below lo
    hi: float            # full points at or above hi
    points: int
    tag: str
    reason: str          # {v} = formatted value of `show`
    fmt: str = "num"
    show: str = ""       # column displayed in the reason (default: feature)
    min_sessions: int = 0
    gate: tuple = ()     # (feature, minimum) required before the rule scores


RULES = [
    # AML: a typology's signature rule can reach FLAG alone; supporting rules add less
    Rule("structuring_24h", "aml", "agg24_threshold_days", 0, 2, 45, "structuring",
         "{v} day(s) with several sub-$10k cash buy-ins totalling $10k+ within 24h", "int",
         gate=("near_threshold_share", 0.9)),
    Rule("near_threshold_cash", "aml", "near_threshold_cash_count", 0, 4, 15, "structuring",
         "{v} cash buy-ins between $8,000 and $9,999", "int", gate=("near_threshold_share", 0.9)),
    Rule("low_play", "aml", "low_play_sessions", 0, 5, 45, "minimal_play",
         "{v} sessions with $1k+ buy-in but under half of it wagered", "int"),
    Rule("refining", "aml", "refining_index", 0.5, 1.0, 10, "refining",
         "Large buy-ins in $20s, cash-outs in $100s or cheques (index {v})", "num"),
    Rule("affordability", "aml", "affordability_buy_in", 1.0, 5.0, 45, "profile_mismatch",
         "Annual buy-ins {v} the income-band midpoint ({income_band}, {occupation})", "mult"),
    Rule("peer_buy_in", "aml", "peer_z_buy_in", 2.5, 6.0, 10, "profile_mismatch",
         "Annual buy-ins {v} the median of similar players ({peer})", "mult", show="buy_in_vs_peer_median"),
    Rule("away_play", "aml", "away_share", 0.15, 0.5, 15, "location_anomaly",
         "{v} of play at venues 300+ km from home or from foreign IPs", "pct"),
    Rule("impossible_travel", "aml", "impossible_travel_count", 1, 3, 45, "location_anomaly",
         "{v} venue visit(s) too far apart for the time between them", "int"),
    Rule("third_party", "aml", "third_party_total", 0, 5000, 45, "third_party_payout",
         "{v} paid by cheque to third-party payees", "money"),
    Rule("unseen_withdrawals", "aml", "withdraw_unseen_total", 0, 5000, 25, "third_party_payout",
         "{v} withdrawn to instruments never used to deposit", "money"),
    Rule("prize_claims", "aml", "large_prize_claims", 0, 3, 15, "prize_claimant",
         "{v} lottery prize claims of $1,000+", "int"),
    Rule("prize_ratio", "aml", "prize_to_purchase_ratio", 2, 20, 45, "prize_claimant",
         "Prizes claimed are {v} own lottery purchases", "mult", gate=("large_prize_claims", 2)),
    Rule("sudden_escalation", "aml", "change_buy_in_log2", 2, 4, 45, "sudden_escalation",
         "Weekly buy-ins up {v} vs own baseline", "log2"),
    Rule("shared_device", "aml", "shared_device_players", 0, 3, 45, "linked_accounts",
         "Device shared with {v} other player(s)", "int"),
    Rule("shared_instrument", "aml", "shared_instrument_players", 0, 3, 20, "linked_accounts",
         "Payment instrument shared with {v} other player(s)", "int"),
    Rule("chip_dumping", "aml", "chip_dump_matches", 0, 2, 45, "chip_dumping",
         "{v} table session(s) where one player lost what another won", "int"),
    # Harm
    Rule("chasing", "harm", "chase_rate", 0.08, 0.35, 25, "chasing",
         "Re-bought within an hour after {v} of heavy losses", "pct", min_sessions=8),
    Rule("chase_count", "harm", "chase_count", 2, 10, 10, "chasing",
         "{v} quick re-buys after heavy losses", "int"),
    Rule("loss_affordability", "harm", "affordability_loss", 0.07, 0.3, 15, "unaffordable_losses",
         "Annualised losses {v} of income-band midpoint ({income_band})", "pct"),
    Rule("peer_losses", "harm", "peer_z_net_loss", 1.2, 3.5, 10, "unaffordable_losses",
         "Losses far above similar players ({peer}, z = {v})", "num"),
    Rule("spend_escalation", "harm", "change_buy_in_log2", 1.0, 3.0, 20, "escalating_spend",
         "Weekly buy-ins up {v} vs own baseline", "log2"),
    Rule("spend_trend", "harm", "buy_in_trend", 0.06, 0.2, 15, "escalating_spend",
         "Buy-ins growing {v} per week", "pct", min_sessions=8),
    Rule("late_night_shift", "harm", "night_share_change", 0.2, 0.5, 10, "late_night",
         "Late-night share up {v} points in the last 30 days", "pct", min_sessions=10),
    Rule("late_night_peer", "harm", "peer_z_night", 2.8, 6, 10, "late_night",
         "Plays late at night far more than similar players (z = {v})", "num", min_sessions=8),
    Rule("cancelled_withdrawals", "harm", "cancel_rate", 0.05, 0.3, 15, "withdrawal_reversal",
         "Cancelled {v} of withdrawals to keep playing", "pct", min_sessions=5),
    Rule("limit_increases", "harm", "limit_increases", 0, 3, 15, "limit_increases",
         "{v} deposit-limit increases", "int"),
    Rule("stake_variability", "harm", "stake_cv", 0.85, 1.6, 10, "erratic_stakes",
         "Highly variable buy-ins (CV {v})", "num", min_sessions=8),
    Rule("game_hopping", "harm", "game_diversity_change", 0, 2, 5, "game_hopping",
         "{v} new game type(s) in the last 30 days", "int"),
]

ACTIONS = {
    "structuring": "Review for suspicious transaction report (24-hour rule)",
    "chip_dumping": "Review table play and linked patrons; consider STR",
    "linked_accounts": "Review linked accounts and shared devices/instruments",
    "minimal_play": "Review buy-in/cash-out pattern; consider STR",
    "third_party_payout": "Verify payee relationship; hold third-party payouts",
    "profile_mismatch": "Request source-of-funds documentation",
    "sudden_escalation": "Enhanced due diligence; request source of funds",
    "prize_claimant": "Review prize-claim history and ticket provenance",
    "refining": "Review currency exchange pattern; consider STR",
    "location_anomaly": "Verify identity and account access locations",
}
HARM_ACTIONS = {"Critical": "Immediate safer-gambling intervention (PlaySmart outreach, limit review)",
                "High": "Safer-gambling outreach; offer limits and self-exclusion options",
                "Medium": "Monitor; send responsible-gambling messaging"}


def fmt(v: float, kind: str) -> str:
    if kind == "int":
        return f"{int(round(v))}"
    if kind == "pct":
        return f"{v:.0%}"
    if kind == "money":
        return f"${v:,.0f}"
    if kind == "mult":
        return f"{v:.1f}x"
    if kind == "log2":
        return f"{2 ** v:.1f}x"
    return f"{v:.2f}"


CHANNEL_WORDS = {"VENUE": "venue players", "ONLINE": "online players", "BOTH": "venue and online players"}


def num(v: float) -> str:
    return f"{v:,.0f}" if float(v).is_integer() or abs(v) >= 100 else f"{v:,.2f}"


def peer_label(key) -> str:
    if not isinstance(key, str) or key == "ALL":
        return "all players"
    *rest, channel = key.split("|")
    words = CHANNEL_WORDS.get(channel, channel.lower())
    if len(rest) == 2:
        return f"age {rest[0]}, {rest[1]} income, {words}"
    if len(rest) == 1:
        return f"{rest[0]} income, {words}"
    return words


def tier(score: pd.Series) -> pd.Series:
    out = pd.Series("Low", index=score.index)
    for cut, name in reversed(TIERS):
        out[score >= cut] = name
    return out


def rule_points(F: pd.DataFrame) -> pd.DataFrame:
    pts = {}
    for r in RULES:
        frac = ((F[r.feature] - r.lo) / (r.hi - r.lo)).clip(0, 1)
        ok = F["n_sessions"] >= r.min_sessions
        if r.gate:
            ok &= F[r.gate[0]] >= r.gate[1]
        pts[r.name] = (frac * r.points).where(ok, 0.0)
    return pd.DataFrame(pts, index=F.index)


def scorecard(F: pd.DataFrame):
    P = rule_points(F)
    by_lens = {lens: [r.name for r in RULES if r.lens == lens] for lens in ("aml", "harm")}
    out = pd.DataFrame(index=F.index)
    for lens, names in by_lens.items():
        out[f"{lens}_score"] = P[names].sum(axis=1).clip(upper=100).round(1)
        out[f"{lens}_tier"] = tier(out[f"{lens}_score"])

    strong = P.ge(pd.Series({r.name: 0.5 * r.points for r in RULES}))
    tag_of = {r.name: r.tag for r in RULES}
    lens_of = {r.name: r.lens for r in RULES}
    for lens in ("aml", "harm"):
        cols = by_lens[lens]
        ordered = P[cols].apply(lambda row: [tag_of[c] for c in row.sort_values(ascending=False).index
                                             if strong.at[row.name, c]], axis=1)
        out[f"{lens}_tags"] = ordered.apply(lambda tags: ",".join(dict.fromkeys(tags)))
    return out, P, lens_of


def reasons(F: pd.DataFrame, P: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    rules = {r.name: r for r in RULES}
    long = P.stack().rename("points").reset_index()
    long.columns = ["player_id", "rule", "points"]
    long = long[long["points"] > 0]
    long["lens"] = long["rule"].map(lambda n: rules[n].lens)
    long = long.sort_values(["player_id", "lens", "points"], ascending=[True, True, False])
    long["rank"] = long.groupby(["player_id", "lens"]).cumcount() + 1
    long = long[long["rank"] <= top]

    def sentence(row):
        r = rules[row.rule]
        f = F.loc[row.player_id]
        v = fmt(f[r.show or r.feature], r.fmt)
        return r.reason.format(v=v, peer=peer_label(f["peer_group"]), income_band=f["income_band"],
                               occupation=f["occupation"].replace("_", " "))

    long["feature"] = long["rule"].map(lambda n: rules[n].feature)
    long["value"] = [float(F.at[p, rules[n].feature]) for p, n in zip(long["player_id"], long["rule"])]
    long["reason"] = [sentence(r) for r in long.itertuples()]
    long["source"] = "scorecard"
    long["points"] = long["points"].round(1)
    return long[["player_id", "lens", "rank", "source", "rule", "feature", "value", "points", "reason"]]


# --- model layer ---------------------------------------------------------------
MODEL_FEATURES = [f for f in FEATURES]


def train_models(train_db: str) -> dict:
    import lightgbm as lgb

    con = sqlite3.connect(train_db)
    F = pd.read_sql("SELECT * FROM player_features", con).set_index("player_id")
    check_features(F, train_db)
    labels = pd.read_sql("SELECT player_id, lens FROM labels", con)
    con.close()
    models = {}
    for lens in ("aml", "harm"):
        y = F.index.isin(labels.loc[labels["lens"] == lens.upper(), "player_id"]).astype(int)
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               scale_pos_weight=(len(y) - y.sum()) / max(1, y.sum()),
                               random_state=0, verbose=-1)
        m.fit(F[MODEL_FEATURES], y)
        models[lens] = m
    return models


def apply_models(models: dict, F: pd.DataFrame, top: int = 3):
    probs = pd.DataFrame(index=F.index)
    rows = []
    X = F[MODEL_FEATURES]
    for lens, m in models.items():
        probs[f"{lens}_model"] = (m.predict_proba(X)[:, 1] * 100).round(1)
        contrib = pd.DataFrame(m.predict(X, pred_contrib=True)[:, :-1], index=F.index, columns=MODEL_FEATURES)
        flagged = probs.index[probs[f"{lens}_model"] >= 50]
        for pid in flagged:
            c = contrib.loc[pid].nlargest(top)
            for rank, (feat, val) in enumerate(c[c > 0].items(), 1):
                rows.append((pid, lens, rank, "model", "model", feat, float(F.at[pid, feat]), round(float(val), 3),
                             f"{FEATURES[feat][2]}: {num(F.at[pid, feat])}"))
    cols = ["player_id", "lens", "rank", "source", "rule", "feature", "value", "points", "reason"]
    return probs, pd.DataFrame(rows, columns=cols)


# --- assemble -------------------------------------------------------------------
def flags(R: pd.DataFrame, lens: str) -> pd.Series:
    """'scorecard', 'model', 'both' or '' for one lens."""
    card = R[f"{lens}_score"] >= FLAG
    model = R[f"{lens}_model"] >= MODEL_FLAG if f"{lens}_model" in R else pd.Series(False, index=R.index)
    return pd.Series(np.select([card & model, card, model], ["both", "scorecard", "model"], ""), index=R.index)


def route(R: pd.DataFrame) -> pd.Series:
    aml, harm = R["aml_flag"] != "", R["harm_flag"] != ""
    return pd.Series(np.select([aml & harm, aml, harm], ["BOTH", "AML", "HARM"], "NONE"), index=R.index)


def action(R: pd.DataFrame) -> pd.Series:
    def one(row):
        parts = []
        if row.route in ("AML", "BOTH"):
            first = row.aml_tags.split(",")[0] if row.aml_tags else ""
            parts.append("AML: " + ACTIONS.get(first, "Investigator review"))
        if row.route in ("HARM", "BOTH"):
            parts.append("Safer gambling: " + HARM_ACTIONS.get(row.harm_tier, "Monitor"))
        if not parts and max(row.aml_score, row.harm_score) >= 25:
            parts.append("Monitor")
        return " | ".join(parts)
    return R.apply(one, axis=1)


def check_features(F: pd.DataFrame, where: str):
    needed = {r.feature for r in RULES} | {r.gate[0] for r in RULES if r.gate} | set(MODEL_FEATURES)
    missing = sorted(needed - set(F.columns))
    if missing:
        raise SystemExit(f"{where}: player_features is missing {', '.join(missing)}. "
                         f"Re-run the latest features.py on this database.")


def score(con: sqlite3.Connection, train_db: str | None = None):
    F = pd.read_sql("SELECT * FROM player_features", con).set_index("player_id")
    check_features(F, "scored db")
    R, P, _ = scorecard(F)
    why = reasons(F, P)
    if train_db:
        probs, model_why = apply_models(train_models(train_db), F)
        R = R.join(probs)
        why = pd.concat([why, model_why], ignore_index=True)
    R["aml_flag"], R["harm_flag"] = flags(R, "aml"), flags(R, "harm")
    R["route"] = route(R)
    # model-only alerts rank below traceable scorecard alerts
    models = [c for c in ("aml_model", "harm_model") if c in R]
    model_pts = (R[models].max(axis=1) * 0.6) if models else 0
    R["priority"] = np.maximum(R[["aml_score", "harm_score"]].max(axis=1), model_pts).round(1)
    tie = R[["aml_score", "harm_score"]].sum(axis=1) + (R[models].sum(axis=1) / 1000 if models else 0)
    R["queue_rank"] = (R.assign(_t=tie).sort_values(["priority", "_t"], ascending=False)
                       .reset_index().reset_index().set_index("player_id")["index"].reindex(R.index) + 1)
    R["recommended_action"] = action(R)
    R["as_of"] = F["as_of"]
    return R.reset_index(), why


def write(con: sqlite3.Connection, R: pd.DataFrame, why: pd.DataFrame):
    R.to_sql("risk_scores", con, if_exists="replace", index=False)
    why.to_sql("risk_reasons", con, if_exists="replace", index=False)
    rules = pd.DataFrame([asdict(r) for r in RULES])
    rules["gate"] = [f"{g[0]} >= {g[1]}" if g else "" for g in rules["gate"]]
    rules.to_sql("scorecard_rules", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_rr_player ON risk_reasons(player_id)")
    con.commit()


# --- evaluation (reads ground truth) ---------------------------------------------
def evaluate(con: sqlite3.Connection, R: pd.DataFrame, k: int):
    from sklearn.metrics import average_precision_score

    labels = pd.read_sql("SELECT player_id, lens, typology, is_hard_negative FROM labels", con)
    R = R.set_index("player_id")
    print(f"\n=== Evaluation vs ground truth (capacity k={k} per lens, flag at {FLAG}) ===")
    print(f"  {'lens':<6}{'signal':<12}{'PR-AUC':>8}{'prec@k':>8}{'recall@k':>10}{'flagged':>9}{'precision':>11}{'recall':>8}")
    for lens in ("aml", "harm"):
        y = pd.Series(R.index.isin(labels.loc[labels["lens"] == lens.upper(), "player_id"]), index=R.index)
        for col in (f"{lens}_score", f"{lens}_model"):
            if col not in R:
                continue
            top = R[col].nlargest(k).index
            flag = R[col] >= (FLAG if col.endswith("score") else MODEL_FLAG)
            print(f"  {lens:<6}{col:<12}{average_precision_score(y, R[col]):>8.2f}{y[top].mean():>8.2f}"
                  f"{y[top].sum() / y.sum():>10.2f}{flag.sum():>9}{y[flag].mean() if flag.any() else 0:>11.2f}"
                  f"{y[flag].sum() / y.sum():>8.2f}")

    print("\n  Recall by typology (routed to the right team)")
    for typ, grp in labels[labels["lens"] != "NONE"].groupby("typology"):
        lens = grp["lens"].iloc[0]
        ok = R.loc[grp["player_id"], "route"].isin([lens, "BOTH"]).mean()
        card = (R.loc[grp["player_id"], f"{lens.lower()}_score"] >= FLAG).mean()
        print(f"    {typ:<20}{len(grp):>4}  routed {ok:>4.0%}  by scorecard {card:>4.0%}")
    hard = labels.loc[labels["is_hard_negative"] == 1, "player_id"]
    clean = R.index.difference(labels["player_id"])
    print(f"\n  Hard negatives routed to a team: {(R.loc[hard, 'route'] != 'NONE').mean():.0%} of {len(hard)}")
    print(f"  Unlabelled players routed to a team: {(R.loc[clean, 'route'] != 'NONE').sum()} of {len(clean):,}")


def main():
    ap = argparse.ArgumentParser(description="Score players for AML and harm risk.")
    ap.add_argument("--db", default="synth.db")
    ap.add_argument("--train-db", default=None, help="separate labelled run for the model layer")
    ap.add_argument("--eval", action="store_true", help="compare with ground-truth labels")
    ap.add_argument("--capacity", type=int, default=50)
    args = ap.parse_args()

    t0 = time.time()
    con = sqlite3.connect(args.db)
    R, why = score(con, args.train_db)
    write(con, R, why)
    print(f"Scored {len(R):,} players in {time.time() - t0:.1f}s")
    print(R.groupby(["route", "aml_flag", "harm_flag"]).size().rename("players").to_string())
    if args.eval:
        evaluate(con, R, args.capacity)
    con.close()


if __name__ == "__main__":
    main()
