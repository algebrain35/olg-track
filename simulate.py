#!/usr/bin/env python3
"""
simulate.py - command-line entry point for the synthetic patron & transaction generator for OLG-style AML and
safer-gambling detection (ThunderHacks 2026).

Generates an SQLite database with:
  OBSERVABLE (what an operator would see; detection code may read these)
    regions, venues, patrons, sessions, transactions
  GROUND TRUTH (evaluation only; detection code must NEVER read these)
    labels, truth_players, sim_meta

Model overview
  Layer 1  PlayerFactory      profile (age, occupation, income band, region) + hidden traits
  Layer 2  Simulator          normal play: Gamma-Poisson session counts, lognormal stakes
                              scaled to income, per-game house edge, slow random-walk drift
  Layer 3  Behaviour overlays harm escalation, AML typologies, multi-player group scenarios,
                              and hard negatives (legitimate but unusual behaviour)

Calibration targets (checked by the report printed at the end)
  - Top 20% of players ~80% of bets in a month, ~90% over a year; top 1% ~19-24%
    (Deng, Lesch & Clark 2021, BC PlayNow eCasino data)
  - Problem gamblers ~4.8% of gamblers but ~36% of revenue
    (Williams & Wood 2007, Ontario)
  Harm markers follow Challet-Bouju et al. 2020 (chasing, involvement) and the 2022
  J Gambling Studies self-exclusion study (limit changes, payment methods, cancelled
  withdrawals). AML typologies follow FATF (2009) "Vulnerabilities of Casinos and
  Gaming Sector" and FINTRAC's $10,000 / 24-hour aggregation rule. Laundering
  prevalence is unknown in reality, so it is a parameter: vary it to test robustness.

Code lives in the olgsim/ package:
  config.py       CONFIG dict (all tunable parameters)
  models.py       enums + dataclasses (Profile, Traits, Player, Session, Transaction, ...)
  reference.py    fixed domain data + sampling helpers
  factory.py      PlayerFactory
  behaviours.py   harm / AML / group / hard-negative overlays and registries
  scenarios.py    assign_behaviours (incl. mule rings, chip-dumping pairs)
  engine.py       Simulator
  db.py           SCHEMA + Writer
  calibration.py  calibration report

Usage
  python simulate.py --players 5000 --days 180 --seed 42 --db synth.db
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np

from olgsim.calibration import calibration_report
from olgsim.config import CONFIG
from olgsim.db import Writer
from olgsim.engine import Simulator
from olgsim.factory import PlayerFactory
from olgsim.scenarios import assign_behaviours


def run(cfg: dict, db_path: str):
    t0 = time.time()
    sim = Simulator(cfg)
    factory = PlayerFactory(cfg, sim)
    seeds = np.random.SeedSequence(cfg["seed"]).spawn(cfg["n_players"] + 1)
    master = np.random.default_rng(seeds[0])
    rngs = [np.random.default_rng(s) for s in seeds[1:]]

    players = [factory.create(i, rngs[i]) for i in range(cfg["n_players"])]
    assign_behaviours(players, master, cfg, sim)
    for p in players:
        for b in p.behaviours:
            b.on_create(factory, p, master)
        factory.finalize(p)

    w = Writer(db_path)
    w.write_static(sim.venues, players)
    for i, p in enumerate(players):
        w.add(*sim.run_player(p, rngs[i]))
    w.write_truth(players, sim.start, cfg)
    w.close()
    print(f"Wrote {db_path} in {time.time() - t0:.1f}s")
    calibration_report(w.con, cfg)
    w.con.close()


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic OLG-style patron and transaction data.")
    ap.add_argument("--players", type=int, default=CONFIG["n_players"])
    ap.add_argument("--days", type=int, default=CONFIG["n_days"])
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--start", default=CONFIG["start_date"])
    ap.add_argument("--aml-scale", type=float, default=1.0, help="multiply all AML prevalences (sensitivity tests)")
    ap.add_argument("--db", default="synth.db")
    args = ap.parse_args()

    cfg = copy.deepcopy(CONFIG)
    cfg.update(n_players=args.players, n_days=args.days, seed=args.seed, start_date=args.start)
    cfg["aml"] = {k: v * args.aml_scale for k, v in cfg["aml"].items()}
    run(cfg, args.db)


if __name__ == "__main__":
    main()
