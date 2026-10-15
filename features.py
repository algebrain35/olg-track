#!/usr/bin/env python3
"""Player-level features for AML and harm scoring. Reads observable tables only.

    python features.py --db synth.db [--check]
"""
from __future__ import annotations

import argparse
import sqlite3
import time

import numpy as np
import pandas as pd

THRESHOLD = 10_000      # FINTRAC reporting threshold
NEAR_LOW = 8_000
FAR_KM = 300
TRAVEL_KMH = 200
CHASE_MINUTES = 60
MIN_PEER_GROUP = 30
MIN_BASELINE_DAYS = 28
LARGE_SESSION = 1_000
CHIP_DUMP_MIN = 2_000
CHIP_DUMP_MINUTES = 10
NIGHT_HOURS = 6         # 00:00-05:59

INCOME_MID = {"<25k": 20_000, "25-50k": 37_500, "50-75k": 62_500,
              "75-100k": 87_500, "100-150k": 125_000, "150k+": 200_000}

PEER_LEVELS = [["age_band", "income_band", "channel"], ["income_band", "channel"], ["channel"]]

# name -> (lens, family, description)
FEATURES = {
    "n_sessions":               ("both", "activity", "Gaming sessions in the window"),
    "sessions_per_week":        ("both", "activity", "Sessions per week"),
    "annual_buy_in":            ("both", "activity", "Annualised buy-ins / deposits ($)"),
    "annual_net_loss":          ("both", "activity", "Annualised net losses ($)"),
    "mean_buy_in":              ("both", "activity", "Average buy-in per session ($)"),
    "affordability_buy_in":     ("both", "affordability", "Annualised buy-ins as a multiple of income-band midpoint"),
    "affordability_loss":       ("both", "affordability", "Annualised net losses as a share of income-band midpoint"),
    "peer_z_buy_in":            ("both", "peer", "Annual buy-ins vs peer group (robust z)"),
    "peer_z_net_loss":          ("both", "peer", "Annual net losses vs peer group (robust z)"),
    "peer_z_sessions":          ("both", "peer", "Session frequency vs peer group (robust z)"),
    "peer_z_stake":             ("both", "peer", "Average buy-in vs peer group (robust z)"),
    "peer_z_night":             ("harm", "peer", "Late-night share vs peer group (robust z)"),
    "buy_in_vs_peer_median":    ("both", "peer", "Annual buy-ins as a multiple of the peer median"),
    "change_buy_in_log2":       ("both", "change", "Weekly buy-ins, last 30 days vs own baseline (log2 ratio)"),
    "change_sessions_log2":     ("both", "change", "Weekly sessions, last 30 days vs own baseline (log2 ratio)"),
    "change_stake_log2":        ("both", "change", "Average buy-in, last 30 days vs own baseline (log2 ratio)"),
    "buy_in_trend":             ("harm", "change", "Relative weekly growth of buy-ins over the window"),
    "far_venue_share":          ("aml", "location", "Share of venue sessions 300+ km from home"),
    "max_venue_distance_km":    ("aml", "location", "Furthest venue played from home (km)"),
    "regions_played":           ("aml", "location", "Distinct regions with venue play"),
    "impossible_travel_count":  ("aml", "location", "Consecutive venue visits too far apart for the time between them"),
    "foreign_ip_share":         ("aml", "location", "Share of online transactions from a foreign IP"),
    "away_share":               ("aml", "location", "Share of play away from home: distant venues or foreign IPs"),
    "turnover_ratio_median":    ("aml", "money_flow", "Median amount bet per $ bought in"),
    "low_play_sessions":        ("aml", "money_flow", "Sessions with $1k+ buy-in and under 0.5x turnover"),
    "low_play_share":           ("aml", "money_flow", "Share of sessions that are large with little play"),
    "cash_out_to_buy_in":       ("aml", "money_flow", "Total cash-out as a share of total buy-in"),
    "near_threshold_cash_count": ("aml", "threshold", "Cash buy-ins between $8,000 and $9,999"),
    "near_threshold_share":     ("aml", "threshold", "Share of $8,000+ cash buy-ins kept under $10,000"),
    "agg24_threshold_days":     ("aml", "threshold", "Days with 2+ sub-$10k cash buy-ins summing to $10k+ within 24h"),
    "max_cash_24h":             ("aml", "threshold", "Largest cash buy-in total within any 24 hours ($)"),
    "small_notes_in_share":     ("aml", "refining", "Share of large cash buy-ins made in $20 bills"),
    "large_notes_out_share":    ("aml", "refining", "Share of large cash-outs taken in $100s or by cheque"),
    "refining_index":           ("aml", "refining", "Small bills in x large bills / cheques out"),
    "cheque_count":             ("aml", "payout", "Cash-outs paid by cheque"),
    "cheque_share":             ("aml", "payout", "Share of cash-outs paid by cheque"),
    "third_party_payees":       ("aml", "payout", "Distinct third-party cheque payees"),
    "third_party_total":        ("aml", "payout", "Amount paid to third-party payees ($)"),
    "withdraw_unseen_total":    ("aml", "payout", "Online withdrawals to an instrument never used to deposit ($)"),
    "large_prize_claims":       ("aml", "lottery", "Lottery prize claims of $1,000+"),
    "prize_claim_total":        ("aml", "lottery", "Total lottery prizes claimed ($)"),
    "prize_to_purchase_ratio":  ("aml", "lottery", "Prizes claimed per $ of own lottery purchases"),
    "shared_device_players":    ("aml", "network", "Other players seen on the same device"),
    "shared_instrument_players": ("aml", "network", "Other players using the same payment instrument"),
    "linked_players":           ("aml", "network", "Distinct players linked by device, instrument or chip-dumping"),
    "network_size":             ("aml", "network", "Size of the connected group of linked players"),
    "chip_dump_matches":        ("aml", "network", "Same-table sessions where one player lost what another won"),
    "night_share":              ("harm", "harm", "Share of sessions starting midnight-6am"),
    "night_share_change":       ("harm", "harm", "Change in late-night share, last 30 days vs baseline"),
    "chase_count":              ("harm", "harm", "Re-buys within 60 min after losing over half a buy-in"),
    "chase_rate":               ("harm", "harm", "Share of heavy losses followed by a quick re-buy"),
    "cancelled_withdrawals":    ("harm", "harm", "Online withdrawals reversed to keep playing"),
    "cancel_rate":              ("harm", "harm", "Share of withdrawals later cancelled"),
    "limit_increases":          ("harm", "harm", "Deposit-limit increases"),
    "stake_cv":                 ("harm", "harm", "Variability of buy-in size (coefficient of variation)"),
    "distinct_games":           ("harm", "harm", "Distinct game types played"),
    "game_diversity_change":    ("harm", "harm", "Extra game types in the last 30 days vs baseline"),
    "payment_methods":          ("harm", "harm", "Distinct deposit instruments"),
    "deposits_per_active_day":  ("harm", "harm", "Online deposits per day with play"),
}


def load(con: sqlite3.Connection, as_of: str | None):
    patrons = pd.read_sql("SELECT * FROM patrons", con, parse_dates=["signup_date"])
    venues = pd.read_sql("SELECT * FROM venues", con)
    regions = pd.read_sql("SELECT * FROM regions", con)
    sessions = pd.read_sql("SELECT * FROM sessions", con, parse_dates=["start_ts", "end_ts"])
    txns = pd.read_sql("SELECT * FROM transactions", con, parse_dates=["ts"])

    as_of_ts = pd.Timestamp(as_of) if as_of else (txns["ts"].max().normalize() + pd.Timedelta(days=1))
    sessions = sessions[sessions["start_ts"] < as_of_ts].copy()
    txns = txns[txns["ts"] < as_of_ts].copy()
    window_start = min(sessions["start_ts"].min(), txns["ts"].min()).normalize()
    return patrons, venues, regions, sessions, txns, as_of_ts, window_start


def region_distances(regions: pd.DataFrame) -> dict:
    lat = np.radians(regions["lat"].to_numpy())
    lon = np.radians(regions["lon"].to_numpy())
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    h = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    km = 6371 * 2 * np.arcsin(np.sqrt(h))
    codes = regions["region_code"].tolist()
    return {(a, b): float(km[i, j]) for i, a in enumerate(codes) for j, b in enumerate(codes)}


def activity_features(F, s, t, as_of, recent_start):
    g = s.groupby("player_id")
    out = pd.DataFrame({
        "n_sessions": g.size(),
        "total_buy_in": g["buy_in"].sum(),
        "total_turnover": g["turnover"].sum(),
        "total_cash_out": g["cash_out"].sum(),
        "net_loss": -g["net_result"].sum(),
        "mean_buy_in": g["buy_in"].mean(),
        "stake_cv": (g["buy_in"].std() / g["buy_in"].mean()),
        "distinct_games": g["game_type"].nunique(),
        "active_days": s.assign(d=s["start_ts"].dt.normalize()).groupby("player_id")["d"].nunique(),
    })
    out = out.reindex(F.index).fillna(0.0)
    days = F["obs_days"]
    out["sessions_per_week"] = out["n_sessions"] / days * 7
    out["annual_buy_in"] = out["total_buy_in"] / days * 365
    out["annual_net_loss"] = out["net_loss"].clip(lower=0) / days * 365
    income = F["income_band"].map(INCOME_MID)
    out["affordability_buy_in"] = out["annual_buy_in"] / income
    out["affordability_loss"] = out["annual_net_loss"] / income

    lottery = t[t["txn_type"] == "LOTTERY_PURCHASE"].groupby("player_id")["amount"].sum()
    out["lottery_spend"] = lottery.reindex(F.index).fillna(0.0)
    return out


def change_features(F, s, recent_start, recent_days):
    recent = s[s["start_ts"] >= recent_start]
    base = s[s["start_ts"] < recent_start]
    rw = recent_days / 7
    bw = (F["baseline_days"] / 7).where(F["baseline_days"] >= MIN_BASELINE_DAYS)

    def per(df, col, how):
        return getattr(df.groupby("player_id")[col], how)().reindex(F.index).fillna(0.0)

    out = pd.DataFrame(index=F.index)
    r_buy, b_buy = per(recent, "buy_in", "sum") / rw, per(base, "buy_in", "sum") / bw
    r_n, b_n = per(recent, "buy_in", "count") / rw, per(base, "buy_in", "count") / bw
    r_stake, b_stake = per(recent, "buy_in", "mean"), per(base, "buy_in", "mean")
    out["change_buy_in_log2"] = np.log2((r_buy + 25) / (b_buy + 25))
    out["change_sessions_log2"] = np.log2((r_n + 0.25) / (b_n + 0.25))
    has_both = (r_stake > 0) & (b_stake > 0)
    out["change_stake_log2"] = np.log2((r_stake + 20) / (b_stake + 20)).where(has_both, 0.0)

    night = s["start_ts"].dt.hour < NIGHT_HOURS
    r_night = s[night & (s["start_ts"] >= recent_start)].groupby("player_id").size().reindex(F.index).fillna(0)
    b_night = s[night & (s["start_ts"] < recent_start)].groupby("player_id").size().reindex(F.index).fillna(0)
    r_share = r_night / per(recent, "buy_in", "count").replace(0, np.nan)
    b_share = b_night / per(base, "buy_in", "count").replace(0, np.nan)
    out["night_share_change"] = (r_share - b_share)

    out["game_diversity_change"] = per(recent, "game_type", "nunique") - per(base, "game_type", "nunique")
    out = out.fillna(0.0)
    no_base = F["baseline_days"] < MIN_BASELINE_DAYS
    out.loc[no_base, ["change_buy_in_log2", "change_sessions_log2", "change_stake_log2",
                      "night_share_change", "game_diversity_change"]] = 0.0
    out["has_baseline"] = (~no_base).astype(int)
    return out


def trend_features(F, s, window_start):
    """Weekly buy-in slope relative to the mean."""
    n_weeks = int((s["start_ts"].max() - window_start).days // 7) + 1
    week = ((s["start_ts"] - window_start).dt.days // 7).clip(upper=n_weeks - 1)
    pivot = (s.assign(week=week).pivot_table(index="player_id", columns="week", values="buy_in",
                                             aggfunc="sum", fill_value=0.0)
             .reindex(index=F.index, columns=range(n_weeks), fill_value=0.0))
    y = pivot.to_numpy(float)
    first_week = ((F["obs_start"] - window_start).dt.days // 7).clip(lower=0).to_numpy()
    w = np.arange(n_weeks)[None, :].astype(float)
    mask = w >= first_week[:, None]
    y = np.where(mask, y, np.nan)
    wm = np.where(mask, w, np.nan)
    w_mean = np.nanmean(wm, axis=1, keepdims=True)
    y_mean = np.nanmean(y, axis=1, keepdims=True)
    cov = np.nansum((wm - w_mean) * (y - y_mean), axis=1)
    var = np.nansum((wm - w_mean) ** 2, axis=1)
    slope = np.divide(cov, var, out=np.zeros_like(cov), where=var > 0)
    rel = slope / (y_mean[:, 0] + 10.0)
    return pd.DataFrame({"buy_in_trend": np.nan_to_num(rel)}, index=F.index)


def peer_features(F, cols: dict):
    """Robust z-scores within peer groups; small groups fall back to broader ones."""
    gamblers = F.index[F["n_sessions"] > 0]
    peer = pd.Series(pd.NA, index=F.index, dtype="object")
    for level in PEER_LEVELS:
        key = F.loc[gamblers, level].astype(str).agg("|".join, axis=1)
        size = key.map(key.value_counts())
        free = peer.loc[gamblers].isna() & (size >= MIN_PEER_GROUP)
        peer.loc[free[free].index] = key[free]
    peer.loc[gamblers[peer.loc[gamblers].isna().to_numpy()]] = "ALL"

    out = pd.DataFrame(index=F.index)
    out["peer_group"] = peer
    out["peer_group_size"] = peer.map(peer.value_counts()).fillna(0).astype(int)
    G = F.loc[gamblers].assign(peer_group=peer.loc[gamblers])
    for name, (col, log) in cols.items():
        x = np.log1p(G[col].clip(lower=0)) if log else G[col]
        grp = x.groupby(G["peer_group"])
        med = grp.transform("median")
        mad = (x - med).abs().groupby(G["peer_group"]).transform("median") * 1.4826
        sd = grp.transform("std")
        scale = mad.where(mad > 1e-6, sd).fillna(0)
        scale = scale.where(scale > 1e-6, 1.0)
        out[name] = ((x - med) / scale).clip(-10, 30).reindex(F.index).fillna(0.0)
    peer_median = F.loc[gamblers, "annual_buy_in"].groupby(G["peer_group"]).transform("median")
    out["buy_in_vs_peer_median"] = (F.loc[gamblers, "annual_buy_in"] / (peer_median + 1.0)).reindex(F.index).fillna(0.0)
    return out


def location_features(F, s, t, venues, dist):
    home = F["home_region"]
    vs = s[s["channel"] == "VENUE"].merge(venues[["venue_id", "region"]], on="venue_id", how="left")
    vs["home"] = vs["player_id"].map(home)
    vs["km"] = [dist.get((h, r), 0.0) for h, r in zip(vs["home"], vs["region"])]
    g = vs.groupby("player_id")
    out = pd.DataFrame({
        "far_venue_share": g["km"].apply(lambda k: (k >= FAR_KM).mean()),
        "max_venue_distance_km": g["km"].max(),
        "regions_played": g["region"].nunique(),
    })

    vs = vs.sort_values(["player_id", "start_ts"])
    prev_region = vs.groupby("player_id")["region"].shift()
    prev_end = vs.groupby("player_id")["end_ts"].shift()
    gap_h = ((vs["start_ts"] - prev_end).dt.total_seconds() / 3600).clip(lower=0.25)
    hop_km = pd.Series([dist.get((a, b), 0.0) if isinstance(a, str) else 0.0
                        for a, b in zip(prev_region, vs["region"])], index=vs.index)
    impossible = (hop_km >= 200) & (hop_km / gap_h > TRAVEL_KMH)
    out["impossible_travel_count"] = impossible.groupby(vs["player_id"]).sum()

    online = t[t["ip_country"].notna()]
    out["foreign_ip_share"] = (online["ip_country"] != "CA").groupby(online["player_id"]).mean()
    out = out.reindex(F.index).fillna(0.0)
    out["away_share"] = out[["far_venue_share", "foreign_ip_share"]].max(axis=1)
    return out


def money_flow_features(F, s):
    ratio = s["turnover"] / s["buy_in"].replace(0, np.nan)
    low = (s["buy_in"] >= LARGE_SESSION) & (ratio < 0.5)
    g = s.assign(ratio=ratio, low=low).groupby("player_id")
    out = pd.DataFrame({
        "turnover_ratio_median": g["ratio"].median(),
        "low_play_sessions": g["low"].sum(),
        "low_play_share": g["low"].mean(),
    }).reindex(F.index).fillna(0.0)
    out["cash_out_to_buy_in"] = (F["total_cash_out"] / F["total_buy_in"].replace(0, np.nan)).fillna(0.0)
    return out


def threshold_features(F, t):
    cash = t[(t["txn_type"] == "BUY_IN") & (t["method"] == "CASH")].sort_values(["player_id", "ts"])
    near = cash["amount"].between(NEAR_LOW, THRESHOLD - 0.01)
    big = cash["amount"] >= NEAR_LOW
    out = pd.DataFrame({"near_threshold_cash_count": near.groupby(cash["player_id"]).sum(),
                        "near_threshold_share": near[big].groupby(cash.loc[big, "player_id"]).mean()})

    multi = cash[cash.groupby("player_id")["amount"].transform("size") >= 2]
    if len(multi):
        roll = multi.set_index("ts").groupby("player_id")["amount"].rolling("24h")
        r = pd.DataFrame({"sum": roll.sum(), "count": roll.count(), "max": roll.max()}).reset_index()
        hit = (r["sum"] >= THRESHOLD) & (r["count"] >= 2) & (r["max"] < THRESHOLD)
        out["agg24_threshold_days"] = r[hit].assign(d=r.loc[hit, "ts"].dt.normalize()).groupby("player_id")["d"].nunique()
        out["max_cash_24h"] = r.groupby("player_id")["sum"].max()
    single = cash.groupby("player_id")["amount"].max()
    out = out.reindex(F.index)
    out["max_cash_24h"] = out.get("max_cash_24h", pd.Series(index=F.index, dtype=float)).fillna(single.reindex(F.index))
    return out.fillna(0.0)


def payout_features(F, t):
    big_in = t[(t["txn_type"] == "BUY_IN") & (t["method"] == "CASH") & (t["amount"] >= LARGE_SESSION)]
    outs = t[t["txn_type"] == "CASH_OUT"]
    big_out = outs[outs["amount"] >= LARGE_SESSION]
    out = pd.DataFrame({
        "small_notes_in_share": (big_in["denomination"] == "20s").groupby(big_in["player_id"]).mean(),
        "large_notes_out_share": ((big_out["denomination"] == "100s") | (big_out["method"] == "CHEQUE"))
                                 .groupby(big_out["player_id"]).mean(),
        "cheque_count": (outs["method"] == "CHEQUE").groupby(outs["player_id"]).sum(),
        "cheque_share": (outs["method"] == "CHEQUE").groupby(outs["player_id"]).mean(),
    }).reindex(F.index).fillna(0.0)
    out["refining_index"] = out["small_notes_in_share"] * out["large_notes_out_share"]

    tp = t[t["payee_name"].notna()]
    out["third_party_payees"] = tp.groupby("player_id")["payee_name"].nunique().reindex(F.index).fillna(0)
    out["third_party_total"] = tp.groupby("player_id")["amount"].sum().reindex(F.index).fillna(0.0)

    dep = t[t["txn_type"] == "DEPOSIT"]
    wd = t[t["txn_type"] == "WITHDRAWAL"]
    own = set(zip(dep["player_id"], dep["instrument_id"]))
    unseen = [(p, i) not in own for p, i in zip(wd["player_id"], wd["instrument_id"])]
    out["withdraw_unseen_total"] = wd.loc[unseen].groupby("player_id")["amount"].sum().reindex(F.index).fillna(0.0)

    claims = t[t["txn_type"] == "PRIZE_CLAIM"]
    out["large_prize_claims"] = (claims["amount"] >= 1000).groupby(claims["player_id"]).sum().reindex(F.index).fillna(0)
    out["prize_claim_total"] = claims.groupby("player_id")["amount"].sum().reindex(F.index).fillna(0.0)
    out["prize_to_purchase_ratio"] = out["prize_claim_total"] / (F["lottery_spend"] + 50.0)
    return out


def chip_dump_pairs(s) -> pd.DataFrame:
    cand = s[s["game_type"].isin(["table", "baccarat"]) & (s["net_result"].abs() >= CHIP_DUMP_MIN)]
    losers = cand[cand["net_result"] <= -CHIP_DUMP_MIN][["player_id", "venue_id", "start_ts"]]
    winners = cand[cand["net_result"] >= CHIP_DUMP_MIN][["player_id", "venue_id", "start_ts"]]
    m = losers.merge(winners, on="venue_id", suffixes=("_l", "_w"))
    m = m[(m["player_id_l"] != m["player_id_w"])
          & ((m["start_ts_l"] - m["start_ts_w"]).abs() <= pd.Timedelta(minutes=CHIP_DUMP_MINUTES))]
    return m[["player_id_l", "player_id_w"]]


def network_features(F, t, s):
    edges = []
    shared = {}
    for col in ("device_id", "instrument_id"):
        x = t[t[col].notna()][["player_id", col]].drop_duplicates()
        n = x.groupby(col)["player_id"].transform("size")
        x = x[n >= 2]
        shared[col] = (x.groupby(col)["player_id"].transform("size") - 1).groupby(x["player_id"]).max()
        for _, grp in x.groupby(col)["player_id"]:
            ids = grp.tolist()
            edges += [(ids[0], other) for other in ids[1:]]
            edges += [(a, b) for a, b in zip(ids, ids[1:])]
    pairs = chip_dump_pairs(s)
    edges += list(zip(pairs["player_id_l"], pairs["player_id_w"]))

    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    neighbours: dict = {}
    for a, b in edges:
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    comp = pd.Series({p: find(p) for p in parent}, dtype="object")
    out = pd.DataFrame(index=F.index)
    out["shared_device_players"] = shared["device_id"].reindex(F.index).fillna(0)
    out["shared_instrument_players"] = shared["instrument_id"].reindex(F.index).fillna(0)
    out["linked_players"] = pd.Series({p: len(n) for p, n in neighbours.items()}).reindex(F.index).fillna(0)
    out["network_component"] = comp.reindex(F.index)
    out["network_size"] = comp.map(comp.value_counts()).reindex(F.index).fillna(1).astype(int)
    out["chip_dump_matches"] = (pairs["player_id_l"].value_counts().reindex(F.index).fillna(0)
                                + pairs["player_id_w"].value_counts().reindex(F.index).fillna(0))
    return out


def harm_features(F, s, t):
    s = s.sort_values(["player_id", "start_ts"])
    g = s.groupby("player_id")
    prev_end, prev_net, prev_buy = g["end_ts"].shift(), g["net_result"].shift(), g["buy_in"].shift()
    gap_min = (s["start_ts"] - prev_end).dt.total_seconds() / 60
    heavy_loss_before = prev_net < -0.5 * prev_buy
    chase = heavy_loss_before & gap_min.between(0, CHASE_MINUTES)
    out = pd.DataFrame({
        "night_share": (s["start_ts"].dt.hour < NIGHT_HOURS).groupby(s["player_id"]).mean(),
        "chase_count": chase.groupby(s["player_id"]).sum(),
        "heavy_losses": (s["net_result"] < -0.5 * s["buy_in"]).groupby(s["player_id"]).sum(),
    }).reindex(F.index).fillna(0.0)
    out["chase_rate"] = (out["chase_count"] / out["heavy_losses"].replace(0, np.nan)).fillna(0.0)

    wd = t[t["txn_type"] == "WITHDRAWAL"].groupby("player_id").size().reindex(F.index).fillna(0)
    cx = t[t["txn_type"] == "WITHDRAWAL_CANCELLED"].groupby("player_id").size().reindex(F.index).fillna(0)
    out["cancelled_withdrawals"] = cx
    out["cancel_rate"] = (cx / wd.replace(0, np.nan)).fillna(0.0)

    lim = t[t["txn_type"] == "LIMIT_CHANGE"].sort_values(["player_id", "ts"])
    up = lim["amount"] > lim.groupby("player_id")["amount"].shift()
    out["limit_increases"] = up.groupby(lim["player_id"]).sum().reindex(F.index).fillna(0)

    dep = t[t["txn_type"] == "DEPOSIT"]
    out["payment_methods"] = dep.groupby("player_id")["instrument_id"].nunique().reindex(F.index).fillna(0)
    active = F["active_days"].replace(0, np.nan)
    out["deposits_per_active_day"] = (dep.groupby("player_id").size().reindex(F.index) / active).fillna(0.0)
    return out.drop(columns=["heavy_losses"])


def build_features(con: sqlite3.Connection, as_of: str | None = None, recent_days: int = 30) -> pd.DataFrame:
    patrons, venues, regions, s, t, as_of_ts, window_start = load(con, as_of)
    recent_start = as_of_ts - pd.Timedelta(days=recent_days)
    dist = region_distances(regions)

    F = patrons.set_index("player_id")[["age_band", "occupation", "income_band", "home_region",
                                        "channel", "loyalty_tier", "signup_date"]].copy()
    F["obs_start"] = F["signup_date"].clip(lower=window_start)
    F["obs_days"] = ((as_of_ts - F["obs_start"]).dt.days).clip(lower=1)
    F["baseline_days"] = ((recent_start - F["obs_start"]).dt.days).clip(lower=0)

    F = F.join(activity_features(F, s, t, as_of_ts, recent_start))
    F = F.join(change_features(F, s, recent_start, recent_days))
    F = F.join(trend_features(F, s, window_start))
    F = F.join(location_features(F, s, t, venues, dist))
    F = F.join(money_flow_features(F, s))
    F = F.join(threshold_features(F, t))
    F = F.join(payout_features(F, t))
    F = F.join(network_features(F, t, s))
    F = F.join(harm_features(F, s, t))
    F = F.join(peer_features(F, {
        "peer_z_buy_in": ("annual_buy_in", True),
        "peer_z_net_loss": ("annual_net_loss", True),
        "peer_z_sessions": ("sessions_per_week", False),
        "peer_z_stake": ("mean_buy_in", True),
        "peer_z_night": ("night_share", False),
    }))

    F["as_of"] = as_of_ts.date().isoformat()
    F["signup_date"] = F["signup_date"].dt.date.astype(str)
    F["obs_start"] = F["obs_start"].dt.date.astype(str)
    return F.reset_index()


def write_features(con: sqlite3.Connection, F: pd.DataFrame):
    F.to_sql("player_features", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_pf_player ON player_features(player_id)")
    meta = pd.DataFrame([(k, *v) for k, v in FEATURES.items()],
                        columns=["feature", "lens", "family", "description"])
    meta.to_sql("feature_meta", con, if_exists="replace", index=False)
    con.commit()


CHECK = {
    "structuring": ["agg24_threshold_days", "near_threshold_cash_count"],
    "minimal_play": ["low_play_sessions", "cash_out_to_buy_in"],
    "refining": ["refining_index", "low_play_sessions"],
    "profile_mismatch": ["affordability_buy_in", "peer_z_buy_in"],
    "location_anomaly": ["away_share", "impossible_travel_count"],
    "third_party_payout": ["third_party_total", "withdraw_unseen_total"],
    "prize_claimant": ["large_prize_claims", "prize_to_purchase_ratio"],
    "sudden_escalation": ["change_buy_in_log2", "peer_z_buy_in"],
    "mule_ring": ["shared_device_players", "withdraw_unseen_total"],
    "chip_dumping": ["chip_dump_matches", "linked_players"],
    "harm_escalation": ["chase_rate", "night_share_change", "cancel_rate", "change_buy_in_log2", "buy_in_trend"],
}


def auc(pos: pd.Series, neg: pd.Series) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = pd.concat([pos, neg]).rank()
    return float((r.iloc[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# evaluation only: reads ground-truth labels
def separation_report(con: sqlite3.Connection, F: pd.DataFrame):
    labels = pd.read_sql("SELECT player_id, typology, is_hard_negative FROM labels", con)
    clean = F[~F["player_id"].isin(labels["player_id"]) & (F["n_sessions"] > 0)]
    hard = F[F["player_id"].isin(labels.loc[labels["is_hard_negative"] == 1, "player_id"])]
    print("\n=== Feature separation vs ground truth (evaluation only) ===")
    print(f"  baseline: {len(clean):,} unlabelled players with sessions; {len(hard)} hard negatives")
    print(f"  {'typology':<20}{'feature':<28}{'n':>4}{'median pos':>12}{'median clean':>14}{'AUC':>7}{'AUC vs hard':>13}")
    for typ, feats in CHECK.items():
        pos = F[F["player_id"].isin(labels.loc[labels["typology"] == typ, "player_id"])]
        for f in feats:
            print(f"  {typ:<20}{f:<28}{len(pos):>4}{pos[f].median():>12.2f}{clean[f].median():>14.2f}"
                  f"{auc(pos[f], clean[f]):>7.2f}{auc(pos[f], hard[f]):>13.2f}")
            typ = ""


def main():
    ap = argparse.ArgumentParser(description="Build the player-level feature table.")
    ap.add_argument("--db", default="synth.db")
    ap.add_argument("--as-of", default=None, help="score as of this date (default: day after last transaction)")
    ap.add_argument("--recent-days", type=int, default=30)
    ap.add_argument("--check", action="store_true", help="print separation vs ground-truth labels")
    args = ap.parse_args()

    t0 = time.time()
    con = sqlite3.connect(args.db)
    F = build_features(con, args.as_of, args.recent_days)
    write_features(con, F)
    print(f"Wrote player_features: {len(F):,} players x {F.shape[1]} columns in {time.time() - t0:.1f}s "
          f"(as of {F['as_of'].iloc[0]})")
    if args.check:
        separation_report(con, F)
    con.close()


if __name__ == "__main__":
    main()
