"""Enums and dataclasses shared by the simulator and downstream code
(feature pipeline, dashboard). No dependencies on other olgsim modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Callable, Optional

import numpy as np


class Channel(str, Enum):
    VENUE = "VENUE"
    ONLINE = "ONLINE"
    BOTH = "BOTH"


class Txn(str, Enum):
    BUY_IN = "BUY_IN"
    CASH_OUT = "CASH_OUT"
    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"
    WITHDRAWAL_CANCELLED = "WITHDRAWAL_CANCELLED"
    LOTTERY_PURCHASE = "LOTTERY_PURCHASE"
    PRIZE_CLAIM = "PRIZE_CLAIM"
    LIMIT_CHANGE = "LIMIT_CHANGE"
    SELF_EXCLUSION = "SELF_EXCLUSION"


@dataclass(slots=True)
class Venue:
    venue_id: str
    name: str
    region: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    venue_type: str  # casino | slots | online | lottery_retail | prize_centre

    def row(self):
        return (self.venue_id, self.name, self.region, self.lat, self.lon, self.venue_type)


@dataclass(slots=True)
class Profile:
    """Observable patron profile -> written to `patrons`."""
    player_id: str
    age_band: str
    occupation: str
    income_band: str
    home_region: str
    channel: Channel
    signup_date: date
    loyalty_tier: str = "Bronze"

    def row(self):
        return (self.player_id, self.age_band, self.occupation, self.income_band, self.home_region,
                self.channel.value, self.loyalty_tier, self.signup_date.isoformat())


@dataclass(slots=True)
class Traits:
    """Hidden behavioural parameters. Only reach the DB via `truth_players`."""
    archetype: str
    income: float
    activity_rate: float = 0.0          # sessions per week
    typical_stake: float = 0.0          # median buy-in / deposit
    stake_cv: float = 0.6
    turnover_mean: float = 1.0          # amount bet / buy-in
    night_share: float = 0.04
    online_share: float = 0.0           # for BOTH-channel players
    lottery_rate: float = 0.0           # purchases per week
    venue_games: dict = field(default_factory=dict)
    online_games: dict = field(default_factory=dict)
    preferred_venues: list = field(default_factory=list)
    devices: list = field(default_factory=list)
    instruments: list = field(default_factory=list)   # [(instrument_id, method)] for online
    debit_card: str = ""
    drift: Optional[np.ndarray] = None  # per-day multiplier


@dataclass(slots=True)
class Player:
    profile: Profile
    traits: Traits
    behaviours: list = field(default_factory=list)
    first_day: int = 0                  # > 0 for accounts opened during the window
    excluded_from_day: Optional[int] = None


@dataclass(slots=True)
class SessionParams:
    """Per-day knobs. Built from traits, then modified by active behaviours."""
    stake_cv: float
    night_share: float
    rate_mult: float = 1.0
    stake_mult: float = 1.0
    stake_median_override: Optional[float] = None
    turnover_mult: float = 1.0
    turnover_sampler: Optional[Callable] = None
    far_venue_prob: float = 0.01
    forced_region: Optional[str] = None
    foreign_ip_prob: float = 0.0
    cheque_prob: Optional[float] = None
    payee: Optional[str] = None
    cash_denom_in: Optional[str] = None
    cash_denom_out: Optional[str] = None
    withdraw_prob: float = 0.35
    withdraw_instrument: Optional[tuple] = None
    cancel_withdraw_prob: float = 0.02
    device_override: Optional[str] = None
    game_explore_prob: float = 0.0
    force_channel: Optional[Channel] = None


def _ts(dt: datetime) -> str:
    return dt.isoformat(sep=" ", timespec="seconds")


@dataclass(slots=True)
class Session:
    session_id: int
    player_id: str
    venue_id: str
    channel: str
    start_ts: datetime
    end_ts: datetime
    game_type: str
    buy_in: float
    turnover: float
    net_result: float  # player's perspective: negative = player lost
    cash_out: float

    def row(self):
        return (self.session_id, self.player_id, self.venue_id, self.channel, _ts(self.start_ts),
                _ts(self.end_ts), self.game_type, round(self.buy_in, 2), round(self.turnover, 2),
                round(self.net_result, 2), round(self.cash_out, 2))


@dataclass(slots=True)
class Transaction:
    txn_id: int
    player_id: str
    ts: datetime
    txn_type: Txn
    amount: float
    session_id: Optional[int] = None
    method: Optional[str] = None
    denomination: Optional[str] = None
    instrument_id: Optional[str] = None
    device_id: Optional[str] = None
    venue_id: Optional[str] = None
    ip_region: Optional[str] = None
    ip_country: Optional[str] = None
    payee_name: Optional[str] = None

    def row(self):
        return (self.txn_id, self.player_id, self.session_id, _ts(self.ts), self.txn_type.value,
                round(float(self.amount), 2), self.method, self.denomination, self.instrument_id,
                self.device_id, self.venue_id, self.ip_region, self.ip_country, self.payee_name)


@dataclass(slots=True)
class Label:
    player_id: str
    lens: str              # AML | HARM | NONE
    typology: str
    severity: str
    onset_date: date
    group_id: Optional[str] = None
    is_hard_negative: int = 0

    def row(self):
        return (self.player_id, self.lens, self.typology, self.severity, self.onset_date.isoformat(),
                self.group_id, self.is_hard_negative)
