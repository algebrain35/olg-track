"""PlayerFactory: samples observable profiles and hidden behavioural traits."""
from __future__ import annotations

import math
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np

from .models import Channel, Player, Profile, Traits
from .reference import (AGE_BANDS, ARCHETYPES, OCC_BY_AGE, ONLINE_METHODS, REGIONS, income_band,
                        lognormal_median, rand_id, sample_income, weighted_choice)

if TYPE_CHECKING:
    from .engine import Simulator


class PlayerFactory:
    def __init__(self, cfg: dict, sim: "Simulator"):
        self.cfg = cfg
        self.sim = sim

    def create(self, idx: int, rng) -> Player:
        cfg, sim = self.cfg, self.sim
        age = weighted_choice(AGE_BANDS, rng)
        occ = weighted_choice(OCC_BY_AGE[age], rng)
        income = sample_income(occ, rng)
        region = weighted_choice({k: v[1] for k, v in REGIONS.items()}, rng)
        if rng.random() < cfg["new_account_share"]:
            first_day = int(rng.integers(0, sim.n_days - 30))
            signup = sim.start + timedelta(days=first_day)
        else:
            first_day = 0
            signup = sim.start - timedelta(days=int(rng.integers(30, 2500)))
        profile = Profile(f"P{idx:06d}", age, occ, income_band(income), region, Channel.VENUE, signup)
        traits = Traits(archetype="", income=income)
        p = Player(profile, traits, first_day=first_day)
        self.set_archetype(p, self._sample_archetype(income, rng), rng)
        steps = rng.normal(0, cfg["drift_sd_per_30d"] / math.sqrt(30), sim.n_days)
        traits.drift = np.exp(np.cumsum(steps))
        return p

    @staticmethod
    def _sample_archetype(income: float, rng) -> str:
        w = {k: v["weight"] for k, v in ARCHETYPES.items()}
        if income < 120_000:
            w["vip"] *= 0.01          # VIPs are overwhelmingly high-income
        elif income >= 150_000:
            w["vip"] *= 5
        return weighted_choice(w, rng)

    def set_archetype(self, p: Player, arch: str, rng):
        """(Re)derive behavioural traits for an archetype from the player's income."""
        cfg, t, a = self.cfg, p.traits, ARCHETYPES[arch]
        t.archetype = arch
        channel = a["channel"]
        t.venue_games, t.online_games = {}, {}
        if channel == Channel.VENUE:
            t.venue_games = dict(a["venue_games"])
        else:
            t.online_games = dict(a["online_games"])
        if arch != "lottery_only" and rng.random() < cfg["both_channel_prob"]:
            channel = Channel.BOTH
            t.venue_games, t.online_games = dict(a["venue_games"]), dict(a["online_games"])
            t.online_share = 0.3 if a["channel"] == Channel.VENUE else 0.7
        p.profile.channel = channel

        if a["sessions_wk"] > 0:
            k = cfg["activity_gamma_shape"]
            t.activity_rate = float(rng.gamma(k, a["sessions_wk"] / k))
            scale = (t.income / cfg["income_ref"]) ** cfg["income_stake_elasticity"]
            t.typical_stake = lognormal_median(a["stake"] * scale, cfg["between_player_stake_sigma"], rng)
            t.turnover_mean = float(a["turnover"] * rng.lognormal(0, 0.25))
        else:
            t.activity_rate, t.typical_stake, t.turnover_mean = 0.0, 0.0, 0.0
        t.stake_cv = float(cfg["within_player_stake_cv"] * rng.lognormal(0, 0.2))
        base_night = 0.07 if channel == Channel.ONLINE else 0.04
        t.night_share = float(min(0.3, base_night * rng.lognormal(0, 0.5)))
        t.lottery_rate = float(rng.gamma(0.8, a["lottery_wk"] / 0.8))

        t.preferred_venues = []
        if t.venue_games:
            fav_game = max(t.venue_games, key=t.venue_games.get)
            cands = self.sim.venues_for(p.profile.home_region, fav_game)
            n = min(len(cands), 1 + int(rng.random() < 0.4))
            t.preferred_venues = [cands[i] for i in rng.choice(len(cands), n, replace=False)]
        t.debit_card = rand_id("INS", rng)
        t.instruments, t.devices = [], []
        if channel != Channel.VENUE:
            for _ in range(1 + int(rng.random() < 0.3)):
                t.instruments.append((rand_id("INS", rng), weighted_choice(ONLINE_METHODS, rng)))
            t.devices = [rand_id("DEV", rng) for _ in range(1 + int(rng.random() < 0.3))]

    @staticmethod
    def finalize(p: Player):
        """Loyalty tier is observable and reflects expected spend at creation."""
        t = p.traits
        monthly = t.activity_rate * 4.33 * t.typical_stake
        tier = "Bronze" if monthly < 300 else "Silver" if monthly < 2000 else "Gold" if monthly < 10000 else "Platinum"
        p.profile.loyalty_tier = tier
