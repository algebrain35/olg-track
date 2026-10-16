"""SQLite schema and batched writer. Downstream code should import SCHEMA from
here rather than duplicating table definitions."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import date, datetime

from .reference import REGIONS

SCHEMA = """
CREATE TABLE regions (region_code TEXT PRIMARY KEY, name TEXT, lat REAL, lon REAL);
CREATE TABLE venues (venue_id TEXT PRIMARY KEY, name TEXT, region TEXT, lat REAL, lon REAL, venue_type TEXT);
CREATE TABLE patrons (player_id TEXT PRIMARY KEY, age_band TEXT, occupation TEXT, income_band TEXT,
                      home_region TEXT, channel TEXT, loyalty_tier TEXT, signup_date TEXT);
CREATE TABLE sessions (session_id INTEGER PRIMARY KEY, player_id TEXT, venue_id TEXT, channel TEXT,
                       start_ts TEXT, end_ts TEXT, game_type TEXT, buy_in REAL, turnover REAL,
                       net_result REAL, cash_out REAL);
CREATE TABLE transactions (txn_id INTEGER PRIMARY KEY, player_id TEXT, session_id INTEGER, ts TEXT,
                           txn_type TEXT, amount REAL, method TEXT, denomination TEXT, instrument_id TEXT,
                           device_id TEXT, venue_id TEXT, ip_region TEXT, ip_country TEXT, payee_name TEXT);
-- ground truth: evaluation only
CREATE TABLE labels (player_id TEXT, lens TEXT, typology TEXT, severity TEXT, onset_date TEXT,
                     group_id TEXT, is_hard_negative INTEGER);
CREATE TABLE truth_players (player_id TEXT PRIMARY KEY, archetype TEXT, income REAL,
                            activity_rate REAL, typical_stake REAL);
CREATE TABLE sim_meta (key TEXT PRIMARY KEY, value TEXT);
"""
INDEXES = """
CREATE INDEX ix_tx_player_ts ON transactions(player_id, ts);
CREATE INDEX ix_tx_instrument ON transactions(instrument_id);
CREATE INDEX ix_tx_device ON transactions(device_id);
CREATE INDEX ix_tx_type ON transactions(txn_type);
CREATE INDEX ix_sess_player_ts ON sessions(player_id, start_ts);
CREATE INDEX ix_labels_player ON labels(player_id);
"""


class Writer:
    FLUSH_EVERY = 50_000

    def __init__(self, path: str):
        if os.path.exists(path):
            os.remove(path)
        self.con = sqlite3.connect(path)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=OFF")
        self.con.executescript(SCHEMA)
        self.sessions, self.txns = [], []

    def write_static(self, venues, players):
        self.con.executemany("INSERT INTO regions VALUES (?,?,?,?)",
                             [(k, v[0], v[2], v[3]) for k, v in REGIONS.items()])
        self.con.executemany("INSERT INTO venues VALUES (?,?,?,?,?,?)", [v.row() for v in venues])
        self.con.executemany("INSERT INTO patrons VALUES (?,?,?,?,?,?,?,?)", [p.profile.row() for p in players])

    def add(self, sessions, txns):
        self.sessions += [s.row() for s in sessions]
        self.txns += [t.row() for t in txns]
        if len(self.txns) >= self.FLUSH_EVERY:
            self.flush()

    def flush(self):
        self.con.executemany("INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?)", self.sessions)
        self.con.executemany("INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", self.txns)
        self.sessions, self.txns = [], []

    def write_truth(self, players, start: date, cfg: dict):
        labels = [b.label(p, start).row() for p in players for b in p.behaviours]
        self.con.executemany("INSERT INTO labels VALUES (?,?,?,?,?,?,?)", labels)
        self.con.executemany("INSERT INTO truth_players VALUES (?,?,?,?,?)",
                             [(p.profile.player_id, p.traits.archetype, round(p.traits.income, 2),
                               round(p.traits.activity_rate, 4), round(p.traits.typical_stake, 2))
                              for p in players])
        meta = {"config": json.dumps(cfg, default=str), "generated_at": datetime.now().isoformat(timespec="seconds")}
        self.con.executemany("INSERT INTO sim_meta VALUES (?,?)", list(meta.items()))

    def close(self):
        self.flush()
        self.con.executescript(INDEXES)
        self.con.commit()
