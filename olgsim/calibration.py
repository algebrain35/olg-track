"""Calibration report: compares the generated data with published stylised facts
(Deng et al. 2021 concentration; Williams & Wood 2007 problem-gambler revenue share)."""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import numpy as np


def top_share(values, frac: float) -> float:
    v = np.sort(np.asarray(values, float))[::-1]
    if v.sum() <= 0:
        return float("nan")
    k = max(1, int(round(len(v) * frac)))
    return float(v[:k].sum() / v.sum())


def calibration_report(con: sqlite3.Connection, cfg: dict):
    q = lambda sql, *a: con.execute(sql, a).fetchall()
    start = date.fromisoformat(cfg["start_date"])
    d30 = (start + timedelta(days=30)).isoformat()

    print("\n=== Calibration report ===")
    for tbl in ("patrons", "sessions", "transactions", "labels"):
        print(f"  {tbl:<14}{q(f'SELECT COUNT(*) FROM {tbl}')[0][0]:>10,}")

    full = [r[0] for r in q("SELECT SUM(turnover) FROM sessions GROUP BY player_id")]
    m1 = [r[0] for r in q("SELECT SUM(turnover) FROM sessions WHERE start_ts < ? GROUP BY player_id", d30)]
    clean = [r[0] for r in q("""SELECT SUM(turnover) FROM sessions WHERE player_id NOT IN
                                (SELECT player_id FROM labels) GROUP BY player_id""")]
    print("\n  Concentration of bets (turnover), unlabelled players only (baseline realism)")
    print(f"    top 20%, full window   : {top_share(clean, .2):6.1%}   target ~80-90%")
    print(f"    top 1%,  full window   : {top_share(clean, .01):6.1%}   target ~19-24%")
    print("\n  Concentration of bets (turnover), all players with sessions (includes injected behaviours)")
    print(f"    top 20%, first 30 days : {top_share(m1, .2):6.1%}   target ~78-82%")
    print(f"    top 20%, full window   : {top_share(full, .2):6.1%}   target rising toward ~90% over a year")
    print(f"    top 1%,  first 30 days : {top_share(m1, .01):6.1%}   target ~19%")
    print(f"    top 1%,  full window   : {top_share(full, .01):6.1%}   target ~19-24%")

    rev = dict(q("""SELECT player_id, SUM(-net_result) FROM sessions GROUP BY player_id"""))
    for pid, amt in q("SELECT player_id, SUM(amount) FROM transactions WHERE txn_type='LOTTERY_PURCHASE' GROUP BY player_id"):
        rev[pid] = rev.get(pid, 0) + amt
    for pid, amt in q("""SELECT t.player_id, SUM(t.amount) FROM transactions t
                         WHERE t.txn_type='PRIZE_CLAIM' AND t.player_id NOT IN
                         (SELECT player_id FROM labels WHERE typology='prize_claimant')
                         GROUP BY t.player_id"""):
        rev[pid] = rev.get(pid, 0) - amt
    harm = {r[0] for r in q("SELECT DISTINCT player_id FROM labels WHERE lens='HARM'")}
    gamblers = {r[0] for r in q("SELECT DISTINCT player_id FROM sessions")}
    total = sum(rev.values())
    harm_rev = sum(v for k, v in rev.items() if k in harm)
    print("\n  Harm group")
    print(f"    share of players with sessions : {len(harm & gamblers) / max(1, len(gamblers)):6.1%}   (Williams & Wood: 4.8% of gamblers)")
    print(f"    share of net revenue           : {harm_rev / total:6.1%}   target ~30-36%")

    print("\n  Labels")
    for lens, typ, n in q("SELECT lens, typology, COUNT(*) FROM labels GROUP BY lens, typology ORDER BY lens, typology"):
        print(f"    {lens:<5}{typ:<22}{n:>5}")

    thr = cfg["reporting_threshold"]
    sub = q(f"""SELECT COUNT(*) FROM transactions WHERE txn_type='BUY_IN' AND method='CASH'
                AND amount BETWEEN {thr * 0.8} AND {thr - 0.01}""")[0][0]
    print(f"\n  Cash buy-ins between 80% and 100% of ${thr:,}: {sub:,}")
