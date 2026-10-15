@dataclass(slots=True)
class Profile:                 # observable → written to `patrons`
    player_id: str
    age_band: str
    occupation: str
    income_band: str           # binned from the sampled income
    home_region: str
    channel: Channel           # VENUE / ONLINE / BOTH
    signup_date: date

@dataclass(slots=True)
class Traits:                  # hidden → drives behaviour, never written except via labels
    archetype: str
    income: float              # exact value; only the band is observable
    activity_rate: float       # λ, sessions/week ~ Gamma
    typical_stake: float       # lognormal, scaled to income
    stake_cv: float
    turnover_mean: float       # bets ÷ buy-in
    night_share: float
    preferred_venues: list[str]
    devices: list[str]         # ids shared across players in mule rings
    instruments: list[str]     # card/e-wallet ids, also used for network analysis

@dataclass(slots=True)
class Player:
    profile: Profile
    traits: Traits
    behaviours: list["Behaviour"] = field(default_factory=list)
