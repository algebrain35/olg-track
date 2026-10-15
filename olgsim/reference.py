"""Fixed domain data (demographics, regions, games, archetypes, time-of-day
profiles) and small sampling helpers built on it."""
from __future__ import annotations

import math

import numpy as np

from .models import Channel, Venue

AGE_BANDS = {"19-24": 0.10, "25-34": 0.18, "35-44": 0.17, "45-54": 0.16, "55-64": 0.18, "65+": 0.21}

# occupation -> (median income, lognormal sigma)
OCCUPATIONS = {
    "student": (18_000, 0.5),
    "retail_service": (38_000, 0.35),
    "trades": (65_000, 0.35),
    "office_admin": (55_000, 0.30),
    "professional": (95_000, 0.40),
    "executive": (180_000, 0.50),
    "retired": (45_000, 0.50),
    "unemployed": (15_000, 0.60),
}

OCC_BY_AGE = {
    "19-24": {"student": .45, "retail_service": .30, "trades": .10, "office_admin": .08, "professional": .03, "unemployed": .04},
    "25-34": {"student": .05, "retail_service": .22, "trades": .18, "office_admin": .20, "professional": .25, "executive": .03, "unemployed": .07},
    "35-44": {"retail_service": .18, "trades": .18, "office_admin": .20, "professional": .30, "executive": .08, "unemployed": .06},
    "45-54": {"retail_service": .17, "trades": .17, "office_admin": .20, "professional": .28, "executive": .12, "unemployed": .06},
    "55-64": {"retail_service": .15, "trades": .14, "office_admin": .18, "professional": .22, "executive": .10, "retired": .15, "unemployed": .06},
    "65+": {"retail_service": .05, "professional": .04, "executive": .02, "retired": .87, "unemployed": .02},
}

INCOME_BANDS = [
    (25_000, "<25k"), (50_000, "25-50k"), (75_000, "50-75k"),
    (100_000, "75-100k"), (150_000, "100-150k"), (math.inf, "150k+"),
]

# code -> (name, population weight, lat, lon, n_casinos, n_slots_venues)
REGIONS = {
    "GTA": ("Greater Toronto Area", 0.45, 43.65, -79.38, 2, 2),
    "OTT": ("Ottawa", 0.11, 45.42, -75.70, 1, 1),
    "HAM": ("Hamilton-Niagara", 0.12, 43.09, -79.08, 2, 1),
    "SWO": ("London-Windsor", 0.12, 42.98, -81.25, 1, 2),
    "KWC": ("Kitchener-Waterloo", 0.07, 43.45, -80.49, 1, 1),
    "EON": ("Kingston-Eastern Ontario", 0.05, 44.23, -76.49, 1, 1),
    "SUD": ("Sudbury-Northeast", 0.04, 46.49, -80.99, 1, 0),
    "SSM": ("Sault Ste. Marie-Algoma", 0.015, 46.52, -84.33, 1, 0),
    "TBY": ("Thunder Bay-Northwest", 0.015, 48.38, -89.25, 1, 0),
}

# game -> house edge, outcome sd per $ of turnover (~ per-bet sd / sqrt(bets per session)),
#         median session minutes, where it can be played. Edges are assumptions.
GAMES = {
    "slots":        {"edge": 0.090, "sd": 0.15, "minutes": 75,  "venue_types": ("casino", "slots")},
    "table":        {"edge": 0.030, "sd": 0.13, "minutes": 90,  "venue_types": ("casino",)},
    "baccarat":     {"edge": 0.012, "sd": 0.13, "minutes": 120, "venue_types": ("casino",)},
    "online_slots": {"edge": 0.050, "sd": 0.12, "minutes": 35,  "venue_types": ("online",)},
    "online_table": {"edge": 0.025, "sd": 0.13, "minutes": 30,  "venue_types": ("online",)},
    "sports":       {"edge": 0.060, "sd": 0.35, "minutes": 20,  "venue_types": ("online",)},
}
VENUE_GAMES = ["slots", "table", "baccarat"]
ONLINE_GAMES = ["online_slots", "online_table", "sports"]

ARCHETYPES = {
    "lottery_only": dict(weight=0.38, channel=Channel.VENUE, venue_games={}, online_games={},
                         sessions_wk=0.0, stake=0, turnover=0.0, lottery_wk=1.2),
    "recreational_slots": dict(weight=0.24, channel=Channel.VENUE, venue_games={"slots": 1.0},
                               online_games={"online_slots": 1.0}, sessions_wk=0.5, stake=60,
                               turnover=4.0, lottery_wk=0.3),
    "table_regular": dict(weight=0.08, channel=Channel.VENUE, venue_games={"table": 0.85, "slots": 0.15},
                          online_games={"online_table": 1.0}, sessions_wk=0.6, stake=150,
                          turnover=2.0, lottery_wk=0.2),
    "online_casino": dict(weight=0.15, channel=Channel.ONLINE, venue_games={"slots": 1.0},
                          online_games={"online_slots": 0.8, "online_table": 0.2}, sessions_wk=1.5,
                          stake=40, turnover=5.0, lottery_wk=0.2),
    "online_sports": dict(weight=0.13, channel=Channel.ONLINE, venue_games={"slots": 1.0},
                          online_games={"sports": 1.0}, sessions_wk=2.0, stake=25,
                          turnover=1.3, lottery_wk=0.3),
    "vip": dict(weight=0.02, channel=Channel.VENUE, venue_games={"baccarat": 0.8, "table": 0.2},
                online_games={"online_table": 1.0}, sessions_wk=1.0, stake=500,
                turnover=2.5, lottery_wk=0.1),
}
GAMBLING_ARCH = [a for a in ARCHETYPES if a != "lottery_only"]
VENUE_ARCH = ["recreational_slots", "table_regular", "vip"]
ONLINE_ARCH = ["online_casino", "online_sports"]

DOW_WEIGHTS = np.array([0.8, 0.8, 0.9, 1.0, 1.3, 1.4, 1.1])
DOW_WEIGHTS = DOW_WEIGHTS / DOW_WEIGHTS.mean()          # Monday..Sunday, mean 1
DAY_HOURS = np.arange(6, 24)
VENUE_HOUR_W = np.array([1, 1, 2, 3, 4, 5, 6, 6, 6, 6, 7, 8, 10, 11, 11, 10, 8, 6], float)
ONLINE_HOUR_W = np.array([1, 2, 3, 3, 4, 4, 4, 4, 4, 5, 6, 7, 9, 11, 12, 12, 11, 9], float)
NIGHT_HOURS = np.arange(0, 6)
NIGHT_W = np.array([4, 3, 2, 1, 1, 1], float)
VENUE_HOUR_W /= VENUE_HOUR_W.sum()
ONLINE_HOUR_W /= ONLINE_HOUR_W.sum()
NIGHT_W /= NIGHT_W.sum()

ONLINE_METHODS = {"CARD": 0.50, "INTERAC": 0.35, "EWALLET": 0.15}
FOREIGN_COUNTRIES = ["US", "CN", "AE", "MT", "CY", "RU", "VN", "PH"]
FIRST_NAMES = ["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery", "Quinn"]
LAST_NAMES = ["Chen", "Singh", "Martin", "Tremblay", "Nguyen", "Brown", "Wilson", "Roy", "Li", "Patel"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def weighted_choice(weights: dict, rng):
    keys = list(weights)
    p = np.array([weights[k] for k in keys], float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def income_band(income: float) -> str:
    for upper, label in INCOME_BANDS:
        if income < upper:
            return label
    return INCOME_BANDS[-1][1]


def sample_income(occupation: str, rng) -> float:
    median, sigma = OCCUPATIONS[occupation]
    return float(median * rng.lognormal(0.0, sigma))


def lognormal_median(median: float, sigma: float, rng) -> float:
    return float(median * rng.lognormal(0.0, sigma))


def haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def rand_id(prefix: str, rng) -> str:
    return f"{prefix}_{int(rng.integers(16 ** 8)):08x}"


def rand_name(rng) -> str:
    return f"{FIRST_NAMES[int(rng.integers(len(FIRST_NAMES)))]} {LAST_NAMES[int(rng.integers(len(LAST_NAMES)))]}"


def round_cash(x: float) -> float:   # venue cash in $20 bills
    return float(max(20.0, round(x / 20.0) * 20.0))


def round_online(x: float) -> float:
    return float(max(10.0, round(x / 5.0) * 5.0))


def build_venues() -> list[Venue]:
    venues = []
    for code, (name, _w, lat, lon, n_cas, n_slots) in REGIONS.items():
        for k in range(n_cas):
            venues.append(Venue(f"V_{code}_C{k + 1}", f"{name} Casino {chr(65 + k)}", code, lat, lon, "casino"))
        for k in range(n_slots):
            venues.append(Venue(f"V_{code}_S{k + 1}", f"{name} Slots {chr(65 + k)}", code, lat, lon, "slots"))
        venues.append(Venue(f"R_{code}", f"{name} Lottery Retail", code, lat, lon, "lottery_retail"))
    gta = REGIONS["GTA"]
    venues.append(Venue("PRIZE_CENTRE", "Prize Centre", "GTA", gta[2], gta[3], "prize_centre"))
    venues.append(Venue("ONLINE", "Online Platform", None, None, None, "online"))
    return venues
