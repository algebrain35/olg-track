"""Behaviour overlays: gambling harm, AML typologies, group typologies and hard
negatives, plus the registries used to assign them. Every overlay implements the
same four hooks (on_create, session_params, after_session, daily_extra)."""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, timedelta
from typing import TYPE_CHECKING, Optional

from .models import Channel, Label, Player, Session, SessionParams, Txn
from .reference import (ARCHETYPES, GAMBLING_ARCH, OCC_BY_AGE, OCCUPATIONS, REGIONS, VENUE_ARCH,
                        income_band, lognormal_median, rand_id, rand_name, sample_income)

if TYPE_CHECKING:
    from .factory import PlayerFactory


Out = tuple  # (list[Session], list[Transaction])


class Behaviour:
    lens = "NONE"
    typology = "base"
    is_hard_negative = 0

    def __init__(self, onset_day: int, rng, group_id: Optional[str] = None):
        self.onset_day = onset_day
        self.group_id = group_id
        self.severity = "medium"

    def on_create(self, factory: PlayerFactory, p: Player, rng): ...
    def session_params(self, sim, p: Player, day: int, params: SessionParams, rng): ...
    def after_session(self, sim, p: Player, day: int, s: Session, txns: list, params, rng) -> Out:
        return [], []
    def daily_extra(self, sim, p: Player, day: int, params: SessionParams, rng) -> Out:
        return [], []

    @staticmethod
    def ensure_active(p: Player, rng, lo=0.5, hi=1.2):
        """Session-based typologies need a player who actually turns up."""
        p.traits.activity_rate = max(p.traits.activity_rate, float(rng.uniform(lo, hi)))

    def label(self, p: Player, start: date) -> Label:
        return Label(p.profile.player_id, self.lens, self.typology, self.severity,
                     start + timedelta(days=self.onset_day), self.group_id, self.is_hard_negative)


# ---------- Gambling harm --------------------------------------------------------
class HarmEscalation(Behaviour):
    """Frequency/intensity/variability/trajectory markers plus chasing, cancelled
    withdrawals, limit increases, game diversification and eventual self-exclusion."""
    lens = "HARM"
    typology = "harm_escalation"

    def __init__(self, onset_day, rng, severity="moderate", n_days=180):
        super().__init__(onset_day, rng)
        self.severity = severity
        hi = severity == "high"
        self.r = rng.uniform(0.08, 0.15) if hi else rng.uniform(0.03, 0.07)       # weekly growth
        self.night_target = rng.uniform(0.25, 0.40) if hi else rng.uniform(0.12, 0.25)
        self.chase_p = rng.uniform(0.35, 0.60) if hi else rng.uniform(0.15, 0.30)
        self.cancel_p = 0.30 if hi else 0.12
        self.limit_p = 0.03 if hi else 0.012
        self.cv = 1.6 if hi else 1.1
        self.bust = rng.uniform(0.6, 1.2) if hi else rng.uniform(0.3, 0.6)
        self.limit = 500.0
        excl_p = 0.30 if hi else 0.08
        d = onset_day + int(rng.integers(40, 120))
        self.excl_day = d if (rng.random() < excl_p and d < n_days) else None

    def on_create(self, factory, p, rng):
        # already heavier than their peers before the visible escalation starts
        lo, hi = (1.3, 2.0) if self.severity == "high" else (1.1, 1.5)
        p.traits.typical_stake *= rng.uniform(lo, hi)
        p.traits.activity_rate *= rng.uniform(lo, hi)

    def session_params(self, sim, p, day, params, rng):
        w = (day - self.onset_day) / 7.0
        ramp = min(1.0, w / 6.0)
        # escalation is capped in absolute terms (median buy-in ~ $3,000) so already-large
        # bettors don't turn into whales whose variance swamps everything else
        cap = max(1.5, min(6.0, 3000.0 / max(p.traits.typical_stake, 1.0)))
        params.stake_mult *= min(math.exp(self.r * w), cap)
        params.rate_mult *= min(math.exp(self.r * 0.6 * w), 3.0)
        params.turnover_mult *= 1.0 + self.bust * ramp   # keeps re-betting until the buy-in is gone
        params.night_share += (self.night_target - params.night_share) * ramp
        params.stake_cv = max(params.stake_cv, self.cv)
        params.game_explore_prob = 0.25 * ramp
        params.withdraw_prob *= 0.6
        params.cancel_withdraw_prob = self.cancel_p

    def after_session(self, sim, p, day, s, txns, params, rng):
        sessions, out = [], []
        if s.net_result < -0.5 * s.buy_in and rng.random() < self.chase_p:
            prev = s
            for _ in range(int(rng.integers(1, 4))):    # re-buy within the hour: chasing
                st = prev.end_ts + timedelta(minutes=float(rng.uniform(5, 60)))
                s2, t2 = sim.make_session(p, day, params, rng, channel=Channel(prev.channel),
                                          venue_id=prev.venue_id, game_type=prev.game_type,
                                          start_ts=st, buy_in=prev.buy_in * rng.uniform(0.5, 1.5))
                sessions.append(s2)
                out += t2
                prev = s2
                if s2.net_result >= 0:
                    break
        return sessions, out

    def daily_extra(self, sim, p, day, params, rng):
        out = []
        if p.profile.channel != Channel.VENUE and rng.random() < self.limit_p:
            self.limit *= rng.uniform(1.3, 2.0)
            out.append(sim.txn(p, sim.at(day, rng.integers(8, 24)), Txn.LIMIT_CHANGE, round(self.limit, -1),
                               venue_id="ONLINE", device_id=p.traits.devices[0] if p.traits.devices else None))
        if self.excl_day is not None and day == self.excl_day:
            out.append(sim.txn(p, sim.at(day, 22), Txn.SELF_EXCLUSION, 0.0))
            p.excluded_from_day = day + 1
        return [], out


# ---------- AML typologies (FATF 2009 casino typologies) -------------------------
class MinimalPlay(Behaviour):
    """Buy in large, barely play, cash out (often by cheque)."""
    lens, typology = "AML", "minimal_play"

    def __init__(self, onset_day, rng):
        super().__init__(onset_day, rng)
        self.amount = lognormal_median(5000, 0.45, rng)
        self.freq = rng.uniform(2, 4)
        self.mix = rng.uniform(0.5, 0.9)      # share of days in laundering mode (rest = cover play)
        self.severity = "high" if self.amount > 6000 else "medium"

    def on_create(self, factory, p, rng):
        self.ensure_active(p, rng)

    def session_params(self, sim, p, day, params, rng):
        if rng.random() < self.mix:
            online = p.profile.channel == Channel.ONLINE
            params.stake_median_override = self.amount / (2 if online else 1)
            params.stake_cv = 0.35
            params.turnover_sampler = lambda r: 0.02 + r.beta(1, 6) * 0.5
            params.rate_mult *= self.freq
            params.cheque_prob = 0.5
            params.withdraw_prob = 0.95
            params.cancel_withdraw_prob = 0.0


class Structuring(Behaviour):
    """Several cash buy-ins of $8,000-9,990 inside 24h across venues/shifts:
    each under the $10,000 threshold, together above it."""
    lens, typology = "AML", "structuring"

    def __init__(self, onset_day, rng):
        super().__init__(onset_day, rng)
        self.p_day = rng.uniform(0.06, 0.15)
        self.severity = "high" if self.p_day > 0.1 else "medium"

    def daily_extra(self, sim, p, day, params, rng):
        if rng.random() >= self.p_day:
            return [], []
        sessions, txns = [], []
        venues = sim.venues_for(p.profile.home_region, "slots")
        st = sim.at(day, int(rng.integers(10, 18)))
        p2 = replace(params, cash_denom_in="20s", cheque_prob=0.4)
        for _ in range(int(rng.integers(2, 5))):
            v = venues[int(rng.integers(len(venues)))]
            s, t = sim.make_session(p, day, p2, rng, channel=Channel.VENUE, venue_id=v, game_type="slots",
                                    start_ts=st, buy_in=rng.uniform(8000, 9990),
                                    turnover_ratio=0.05 + rng.beta(1, 4) * 0.8)
            sessions.append(s)
            txns += t
            st = s.end_ts + timedelta(hours=float(rng.uniform(1.0, 4.0)))
        return sessions, txns


class Refining(Behaviour):
    """Small bills in, large bills or cheque out, with little play."""
    lens, typology = "AML", "refining"

    def __init__(self, onset_day, rng):
        super().__init__(onset_day, rng)
        self.amount = lognormal_median(3000, 0.4, rng)
        self.mix = rng.uniform(0.5, 0.9)
        self.freq = rng.uniform(1.5, 3)

    def on_create(self, factory, p, rng):
        self.ensure_active(p, rng)

    def session_params(self, sim, p, day, params, rng):
        if rng.random() < self.mix:
            params.force_channel = Channel.VENUE
            params.stake_median_override = self.amount
            params.stake_cv = 0.4
            params.cash_denom_in, params.cash_denom_out = "20s", "100s"
            params.turnover_sampler = lambda r: r.beta(2, 5) * 0.8
            params.cheque_prob = 0.3
            params.rate_mult *= self.freq


class ProfileMismatch(Behaviour):
    """Spend far beyond what the reported occupation/income supports (whole window)."""
    lens, typology = "AML", "profile_mismatch"

    def on_create(self, factory, p, rng):
        cands = [o for o in OCC_BY_AGE[p.profile.age_band] if OCCUPATIONS[o][0] <= 40_000] or ["unemployed"]
        occ = cands[int(rng.integers(len(cands)))]
        p.profile.occupation = occ
        p.traits.income = min(sample_income(occ, rng), 32_000)
        p.profile.income_band = income_band(p.traits.income)
        online = p.profile.channel == Channel.ONLINE
        p.traits.typical_stake = lognormal_median(600 if online else 1200, 0.4, rng)
        p.traits.activity_rate = max(p.traits.activity_rate, float(rng.uniform(1.5, 3.0)))
        self.severity = "high" if p.traits.typical_stake > 2500 else "medium"


class LocationAnomaly(Behaviour):
    """Play at distant venues / from foreign IPs, plus occasional impossible travel."""
    lens, typology = "AML", "location_anomaly"

    def __init__(self, onset_day, rng):
        super().__init__(onset_day, rng)
        self.factor = rng.uniform(1.5, 3)
        self.travel_p = rng.uniform(0.02, 0.06)

    def on_create(self, factory, p, rng):
        self.ensure_active(p, rng)

    def session_params(self, sim, p, day, params, rng):
        params.far_venue_prob = 0.6
        params.foreign_ip_prob = 0.5
        params.stake_mult *= self.factor

    def daily_extra(self, sim, p, day, params, rng):
        if p.profile.channel == Channel.ONLINE or rng.random() >= self.travel_p:
            return [], []
        home = p.profile.home_region
        far = sim.far_region(home, rng, min_km=500)
        s1, t1 = sim.make_session(p, day, params, rng, channel=Channel.VENUE, game_type="slots",
                                  venue_id=sim.venues_for(home, "slots")[0])
        st = s1.start_ts + timedelta(minutes=float(rng.uniform(40, 100)))   # 500+ km in < 2h
        s2, t2 = sim.make_session(p, day, params, rng, channel=Channel.VENUE, game_type="slots",
                                  venue_id=sim.venues_for(far, "slots")[0], start_ts=st)
        return [s1, s2], t1 + t2


class ThirdPartyPayout(Behaviour):
    """Winnings paid by cheque / withdrawal to someone other than the patron."""
    lens, typology = "AML", "third_party_payout"

    def __init__(self, onset_day, rng):
        super().__init__(onset_day, rng)
        self.payee = rand_name(rng)
        self.inst = (rand_id("INS", rng), "EWALLET")
        self.factor = rng.uniform(2, 4)
        self.freq = rng.uniform(1.5, 2.5)

    def on_create(self, factory, p, rng):
        self.ensure_active(p, rng)

    def session_params(self, sim, p, day, params, rng):
        params.stake_mult *= self.factor
        params.rate_mult *= self.freq
        params.cheque_prob = 0.85
        params.payee = self.payee
        params.withdraw_instrument = self.inst
        params.withdraw_prob = 0.9


class PrizeClaimant(Behaviour):
    """Frequent large lottery prize claims (buying winning tickets with dirty cash)."""
    lens, typology = "AML", "prize_claimant"

    def __init__(self, onset_day, rng):
        super().__init__(onset_day, rng)
        self.per_year = rng.uniform(6, 20)
        self.severity = "high" if self.per_year > 12 else "medium"

    def daily_extra(self, sim, p, day, params, rng):
        out = []
        for _ in range(int(rng.poisson(self.per_year / 365))):
            amt = max(600.0, round(lognormal_median(2500, 0.8, rng)))
            big = amt >= 1000
            region = list(REGIONS)[int(rng.integers(len(REGIONS)))]
            out.append(sim.txn(p, sim.at(day, int(rng.integers(10, 17))), Txn.PRIZE_CLAIM, amt,
                               method="CHEQUE" if big else "CASH",
                               venue_id="PRIZE_CENTRE" if big else f"R_{region}"))
        return [], out


class SuddenEscalation(Behaviour):
    """A near-dormant account suddenly moves 8-30x its usual volume."""
    lens, typology = "AML", "sudden_escalation"

    def __init__(self, onset_day, rng):
        super().__init__(onset_day, rng)
        self.jump = rng.uniform(8, 30)
        self.freq = rng.uniform(3, 8)
        self.severity = "high" if self.jump > 15 else "medium"

    def on_create(self, factory, p, rng):
        p.traits.activity_rate = min(p.traits.activity_rate, 0.25)

    def session_params(self, sim, p, day, params, rng):
        params.stake_mult *= self.jump
        params.rate_mult *= self.freq
        params.cheque_prob = 0.4


# ---------- Group typologies -------------------------------------------------------
class MuleMember(Behaviour):
    """Online accounts sharing a device, each funnelling withdrawals to one hub instrument."""
    lens, typology = "AML", "mule_ring"

    def __init__(self, onset_day, rng, group_id, hub_instrument, shared_device):
        super().__init__(onset_day, rng, group_id)
        self.hub, self.dev = hub_instrument, shared_device
        self.amount = lognormal_median(1500, 0.5, rng)
        self.freq = rng.uniform(2, 4)

    def on_create(self, factory, p, rng):
        p.traits.devices.append(self.dev)
        self.ensure_active(p, rng)

    def session_params(self, sim, p, day, params, rng):
        params.force_channel = Channel.ONLINE
        params.stake_median_override = self.amount
        params.stake_cv = 0.4
        params.turnover_sampler = lambda r: 0.02 + r.beta(1, 5) * 0.5
        params.withdraw_prob = 0.95
        params.withdraw_instrument = self.hub
        params.cancel_withdraw_prob = 0.0
        params.device_override = self.dev if rng.random() < 0.8 else None
        params.rate_mult *= self.freq


class ChipDumping(Behaviour):
    """Two players at the same table: the dumper deliberately loses to the receiver."""
    lens, typology = "AML", "chip_dumping"

    def __init__(self, onset_day, rng, group_id, role, schedule):
        super().__init__(onset_day, rng, group_id)
        self.role = role
        self.schedule = schedule      # day -> (venue_id, start_ts, amount, loss_fraction)

    def daily_extra(self, sim, p, day, params, rng):
        if day not in self.schedule:
            return [], []
        venue, st, x, lf = self.schedule[day]
        if self.role == "dumper":
            buy_in, net, method = x, -x * lf, None
        else:
            buy_in, net = 0.15 * x, x * lf * 0.97
            method = "CHEQUE" if rng.random() < 0.5 else "CASH"
            st = st + timedelta(minutes=float(rng.uniform(0, 10)))
        s, t = sim.make_session(p, day, params, rng, channel=Channel.VENUE, venue_id=venue, game_type="table",
                                start_ts=st, buy_in=buy_in, turnover_ratio=1.5, net_result=net,
                                cashout_method=method)
        return [s], t


# ---------- Hard negatives (legitimate but unusual) ----------------------------------
class LegitVIP(Behaviour):
    lens, typology, is_hard_negative = "NONE", "legit_vip", 1

    def on_create(self, factory, p, rng):
        p.profile.occupation = "executive"
        p.traits.income = max(160_000.0, lognormal_median(350_000, 0.4, rng))
        p.profile.income_band = income_band(p.traits.income)
        factory.set_archetype(p, "vip", rng)


class JackpotWinner(Behaviour):
    lens, typology, is_hard_negative = "NONE", "jackpot_winner", 1

    def daily_extra(self, sim, p, day, params, rng):
        if day != self.onset_day:
            return [], []
        s, t = sim.make_session(p, day, params, rng, channel=Channel.VENUE, game_type="slots",
                                net_result=round(rng.uniform(10_000, 40_000)), cashout_method="CHEQUE")
        return [s], t


class IncomeChange(Behaviour):
    """Promotion/new job: income and spending both rise; profile shows the new income."""
    lens, typology, is_hard_negative = "NONE", "income_change", 1

    def on_create(self, factory, p, rng):
        old = p.traits.income
        p.profile.occupation = "executive" if rng.random() < 0.3 else "professional"
        p.traits.income = max(90_000.0, old * rng.uniform(2.0, 3.5))
        p.profile.income_band = income_band(p.traits.income)
        self.mult = rng.uniform(1.5, 2.5)

    def session_params(self, sim, p, day, params, rng):
        params.stake_mult *= self.mult


class Traveller(Behaviour):
    """One week of normal-sized play in a distant region (vacation)."""
    lens, typology, is_hard_negative = "NONE", "traveller", 1

    def on_create(self, factory, p, rng):
        self.region = factory.sim.far_region(p.profile.home_region, rng)
        self.duration = int(rng.integers(5, 10))

    def session_params(self, sim, p, day, params, rng):
        if day < self.onset_day + self.duration:
            params.forced_region = self.region
            params.rate_mult *= 3


AML_TYPOLOGIES = {
    "minimal_play": (MinimalPlay, VENUE_ARCH + ["online_casino"]),
    "structuring": (Structuring, VENUE_ARCH),
    "refining": (Refining, VENUE_ARCH),
    "profile_mismatch": (ProfileMismatch, GAMBLING_ARCH),
    "location_anomaly": (LocationAnomaly, GAMBLING_ARCH),
    "third_party_payout": (ThirdPartyPayout, GAMBLING_ARCH),
    "prize_claimant": (PrizeClaimant, list(ARCHETYPES)),
    "sudden_escalation": (SuddenEscalation, GAMBLING_ARCH),
}
HARD_NEGATIVES = {
    "legit_vip": (LegitVIP, VENUE_ARCH),
    "jackpot_winner": (JackpotWinner, ["recreational_slots"]),
    "income_change": (IncomeChange, GAMBLING_ARCH),
    "traveller": (Traveller, VENUE_ARCH),
}
