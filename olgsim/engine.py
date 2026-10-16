"""Simulator: geography, time, session and transaction generation, lottery,
and the day-by-day timeline for each player."""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np

from .models import Channel, Player, Session, SessionParams, Transaction, Txn
from .reference import (DAY_HOURS, DOW_WEIGHTS, FOREIGN_COUNTRIES, GAMES, NIGHT_HOURS, NIGHT_W,
                        ONLINE_GAMES, ONLINE_HOUR_W, REGIONS, VENUE_GAMES, VENUE_HOUR_W, build_venues,
                        haversine_km, lognormal_median, round_cash, round_online, weighted_choice)


class Simulator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.start = date.fromisoformat(cfg["start_date"])
        self.start_dt = datetime(self.start.year, self.start.month, self.start.day)
        self.n_days = cfg["n_days"]
        self.venues = build_venues()
        self.venue_type = {v.venue_id: v.venue_type for v in self.venues}
        self._sid = 0
        self._tid = 0
        self._venue_cache: dict = {}
        self.region_xy = {k: (v[2], v[3]) for k, v in REGIONS.items()}
        self.dist = {(a, b): haversine_km(self.region_xy[a], self.region_xy[b]) for a in REGIONS for b in REGIONS}

    # ---- geography -------------------------------------------------------------
    def venues_for(self, region: str, game: str) -> list[str]:
        """Venues in `region` offering `game`, falling back to the nearest region that does."""
        key = (region, game)
        if key not in self._venue_cache:
            types = GAMES[game]["venue_types"]
            for r in sorted(REGIONS, key=lambda r: self.dist[(region, r)]):
                vs = [v.venue_id for v in self.venues if v.region == r and v.venue_type in types]
                if vs:
                    self._venue_cache[key] = vs
                    break
        return self._venue_cache[key]

    def far_region(self, home: str, rng, min_km: Optional[float] = None) -> str:
        min_km = min_km or self.cfg["far_distance_km"]
        far = [r for r in REGIONS if self.dist[(home, r)] >= min_km]
        if not far:
            far = [max(REGIONS, key=lambda r: self.dist[(home, r)])]
        return far[int(rng.integers(len(far)))]

    # ---- time & ids ------------------------------------------------------------
    def at(self, day: int, hour: int, minute: int = 0) -> datetime:
        return self.start_dt + timedelta(days=int(day), hours=int(hour), minutes=int(minute))

    def sample_start(self, day: int, night_share: float, online: bool, rng) -> datetime:
        if rng.random() < night_share:
            hour = int(rng.choice(NIGHT_HOURS, p=NIGHT_W))
        else:
            hour = int(rng.choice(DAY_HOURS, p=ONLINE_HOUR_W if online else VENUE_HOUR_W))
        return self.at(day, hour, int(rng.integers(60)))

    def txn(self, p: Player, ts: datetime, kind: Txn, amount: float, **kw) -> Transaction:
        self._tid += 1
        return Transaction(self._tid, p.profile.player_id, ts, kind, float(amount), **kw)

    # ---- per-day parameters ----------------------------------------------------
    def base_params(self, p: Player, day: int) -> SessionParams:
        t, c = p.traits, self.cfg
        return SessionParams(stake_cv=t.stake_cv, night_share=t.night_share,
                             far_venue_prob=c["far_venue_prob"], withdraw_prob=c["online_withdraw_prob"],
                             cancel_withdraw_prob=c["base_cancel_withdraw_prob"])

    # ---- one session -------------------------------------------------------------
    def make_session(self, p: Player, day: int, params: SessionParams, rng, *, channel=None, venue_id=None,
                     game_type=None, start_ts=None, buy_in=None, turnover_ratio=None, net_result=None,
                     cashout_method=None):
        t, home = p.traits, p.profile.home_region
        if channel is None:
            channel = params.force_channel
        if channel is None:
            if p.profile.channel == Channel.BOTH:
                channel = Channel.ONLINE if rng.random() < t.online_share else Channel.VENUE
            else:
                channel = Channel.ONLINE if p.profile.channel == Channel.ONLINE else Channel.VENUE
        online = channel == Channel.ONLINE

        if game_type is None:
            games = (t.online_games if online else t.venue_games) or {("online_slots" if online else "slots"): 1.0}
            if rng.random() < params.game_explore_prob:
                pool = ONLINE_GAMES if online else VENUE_GAMES
                game_type = pool[int(rng.integers(len(pool)))]
            else:
                game_type = weighted_choice(games, rng)

        region = home
        if params.forced_region:
            region = params.forced_region
        elif rng.random() < params.far_venue_prob:
            region = self.far_region(home, rng)
        if online:
            venue_id = "ONLINE"
        elif venue_id is None:
            types = GAMES[game_type]["venue_types"]
            cands = [v for v in t.preferred_venues if self.venue_type[v] in types] if region == home else []
            cands = cands or self.venues_for(region, game_type)
            venue_id = cands[int(rng.integers(len(cands)))]

        if start_ts is None:
            start_ts = self.sample_start(day, params.night_share, online, rng)

        if buy_in is None:
            med = params.stake_median_override or (t.typical_stake * params.stake_mult * t.drift[day])
            sig = math.sqrt(math.log1p(params.stake_cv ** 2))
            buy_in = med * rng.lognormal(0.0, sig)
        buy_in = round_online(buy_in) if online else round_cash(buy_in)

        if turnover_ratio is None:
            if params.turnover_sampler is not None:
                turnover_ratio = float(params.turnover_sampler(rng))
            else:
                turnover_ratio = t.turnover_mean * params.turnover_mult * rng.lognormal(0, 0.35)
        g = GAMES[game_type]
        turnover = buy_in * turnover_ratio
        if net_result is None:
            net = -g["edge"] * turnover + rng.normal(0, g["sd"] * turnover)
            if game_type in ("slots", "online_slots") and rng.random() < self.cfg["jackpot_prob"]:
                net += min(50_000.0, buy_in * (5 + rng.pareto(1.5) * 20))
            net = max(net, -buy_in)
        else:
            net = max(float(net_result), -buy_in)
        cash_out = round(buy_in + net, 2) if online else float(round(buy_in + net))
        net = cash_out - buy_in

        minutes = g["minutes"] * rng.lognormal(0, 0.45) * (0.4 if turnover_ratio < 0.5 else 1.0)
        end_ts = start_ts + timedelta(minutes=float(np.clip(minutes, 5, 600)))

        self._sid += 1
        sid, pid = self._sid, p.profile.player_id
        s = Session(sid, pid, venue_id, channel.value, start_ts, end_ts, game_type, buy_in, turnover, net, cash_out)
        txns = []
        if online:
            inst = t.instruments[int(rng.integers(len(t.instruments)))] if t.instruments else (t.debit_card, "CARD")
            dev = params.device_override or (t.devices[int(rng.integers(len(t.devices)))] if t.devices else None)
            ipc = FOREIGN_COUNTRIES[int(rng.integers(len(FOREIGN_COUNTRIES)))] if rng.random() < params.foreign_ip_prob else "CA"
            ipr = region if ipc == "CA" else None
            common = dict(session_id=sid, device_id=dev, venue_id="ONLINE", ip_region=ipr, ip_country=ipc)
            txns.append(self.txn(p, start_ts - timedelta(minutes=1), Txn.DEPOSIT, buy_in,
                                 method=inst[1], instrument_id=inst[0], **common))
            if cash_out > 0 and rng.random() < params.withdraw_prob:
                winst = params.withdraw_instrument or (t.instruments[0] if t.instruments else inst)
                wts = end_ts + timedelta(minutes=float(rng.uniform(1, 120)))
                txns.append(self.txn(p, wts, Txn.WITHDRAWAL, cash_out, method=winst[1],
                                     instrument_id=winst[0], **common))
                if rng.random() < params.cancel_withdraw_prob:
                    txns.append(self.txn(p, wts + timedelta(minutes=float(rng.uniform(5, 240))),
                                         Txn.WITHDRAWAL_CANCELLED, cash_out, method=winst[1],
                                         instrument_id=winst[0], **common))
        else:
            if params.cash_denom_in or rng.random() < 0.8:
                denom = params.cash_denom_in or ("20s" if buy_in < 2000 or rng.random() < 0.5 else "100s")
                txns.append(self.txn(p, start_ts, Txn.BUY_IN, buy_in, session_id=sid, method="CASH",
                                     denomination=denom, venue_id=venue_id))
            else:
                txns.append(self.txn(p, start_ts, Txn.BUY_IN, buy_in, session_id=sid, method="DEBIT",
                                     instrument_id=t.debit_card, venue_id=venue_id))
            if cash_out > 0:
                if cashout_method:
                    method = cashout_method
                else:
                    if params.cheque_prob is not None and cash_out >= 1000:
                        cp = params.cheque_prob
                    else:
                        cp = 0.7 if cash_out >= 10_000 else 0.05 if cash_out >= 3000 else 0.0
                    method = "CHEQUE" if rng.random() < cp else "CASH"
                if method == "CHEQUE":
                    txns.append(self.txn(p, end_ts, Txn.CASH_OUT, cash_out, session_id=sid, method="CHEQUE",
                                         venue_id=venue_id, payee_name=params.payee))
                else:
                    denom = params.cash_denom_out or ("100s" if cash_out >= 1000 else "20s")
                    txns.append(self.txn(p, end_ts, Txn.CASH_OUT, cash_out, session_id=sid, method="CASH",
                                         denomination=denom, venue_id=venue_id))
        return s, txns

    # ---- lottery ---------------------------------------------------------------------
    def lottery(self, p: Player, day: int, dow_w: float, rng) -> list[Transaction]:
        out = []
        n = rng.poisson(p.traits.lottery_rate / 7 * dow_w)
        retail = f"R_{p.profile.home_region}"
        for _ in range(int(n)):
            ts = self.at(day, int(rng.integers(8, 22)), int(rng.integers(60)))
            amt = max(2.0, float(round(lognormal_median(10, 0.7, rng))))
            out.append(self.txn(p, ts, Txn.LOTTERY_PURCHASE, amt, method="CASH", venue_id=retail))
            u = rng.random()
            prize = 0.0
            if u < 0.0001:
                prize = min(1_000_000.0, round(lognormal_median(20_000, 1.0, rng)))
            elif u < 0.006:
                prize = float(round(lognormal_median(150, 0.8, rng)))
            if prize >= 50:
                big = prize >= 1000
                out.append(self.txn(p, ts + timedelta(hours=1), Txn.PRIZE_CLAIM, prize,
                                    method="CHEQUE" if big else "CASH",
                                    venue_id="PRIZE_CENTRE" if big else retail))
        return out

    # ---- full timeline for one player -------------------------------------------------
    def run_player(self, p: Player, rng):
        sessions, txns = [], []
        t = p.traits
        online_capable = p.profile.channel != Channel.VENUE
        for day in range(p.first_day, self.n_days):
            if p.excluded_from_day is not None and day >= p.excluded_from_day:
                break
            dow_w = DOW_WEIGHTS[(self.start + timedelta(days=day)).weekday()]
            params = self.base_params(p, day)
            active = [b for b in p.behaviours if day >= b.onset_day]
            for b in active:
                b.session_params(self, p, day, params, rng)
            params.stake_mult = min(params.stake_mult, self.cfg["max_stake_mult"])
            params.rate_mult = min(params.rate_mult, self.cfg["max_rate_mult"])

            if t.activity_rate > 0:
                lam = t.activity_rate / 7 * dow_w * params.rate_mult * math.sqrt(t.drift[day])
                for _ in range(int(rng.poisson(lam))):
                    s, tx = self.make_session(p, day, params, rng)
                    sessions.append(s)
                    txns += tx
                    for b in active:
                        s2, t2 = b.after_session(self, p, day, s, tx, params, rng)
                        sessions += s2
                        txns += t2

            txns += self.lottery(p, day, dow_w, rng)
            if online_capable and rng.random() < self.cfg["base_limit_change_prob"]:
                txns.append(self.txn(p, self.at(day, int(rng.integers(8, 24))), Txn.LIMIT_CHANGE,
                                     float(rng.choice([100, 250, 500, 1000, 2500])), venue_id="ONLINE"))
            for b in active:
                s2, t2 = b.daily_extra(self, p, day, params, rng)
                sessions += s2
                txns += t2
        return sessions, txns
