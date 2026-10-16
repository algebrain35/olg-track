"""Every tunable simulation parameter lives here."""

CONFIG = {
    "n_players": 5000,
    "n_days": 180,
    "start_date": "2026-01-05",
    "seed": 42,
    # population
    "new_account_share": 0.10,          # accounts opened during the simulation window
    "both_channel_prob": 0.15,          # chance a player uses both venues and online
    # normal behaviour
    "activity_gamma_shape": 0.5,        # <1 => many occasional players, few very frequent
    "between_player_stake_sigma": 0.9,  # lognormal sigma of typical stake across players
    "within_player_stake_cv": 0.6,      # session-to-session variability
    "income_ref": 55_000,
    "income_stake_elasticity": 0.5,     # stake scales with (income / ref) ** elasticity
    "drift_sd_per_30d": 0.06,           # slow random walk on stake/frequency
    "far_venue_prob": 0.01,
    "far_distance_km": 300,
    "online_withdraw_prob": 0.35,
    "base_cancel_withdraw_prob": 0.02,
    "base_limit_change_prob": 0.002,    # per day, online players
    "jackpot_prob": 0.001,              # per slots session
    "max_stake_mult": 15.0,             # cap on stacked behaviour multipliers
    "max_rate_mult": 8.0,
    # harm overlay prevalence (share of all players; drawn from gamblers only)
    "harm": {"high": 0.012, "moderate": 0.022},
    # AML typology prevalence (share of all players)
    "aml": {
        "minimal_play": 0.0016,
        "structuring": 0.0012,
        "refining": 0.0010,
        "profile_mismatch": 0.0014,
        "location_anomaly": 0.0010,
        "third_party_payout": 0.0010,
        "prize_claimant": 0.0008,
        "sudden_escalation": 0.0010,
    },
    "aml_second_typology_prob": 0.3,
    "mule_rings_per_1000": 1.0,
    "mule_ring_size": (3, 7),
    "chip_dumping_pairs_per_1000": 0.4,
    # hard negatives: unusual but legitimate
    "hard_negatives": {
        "legit_vip": 0.004,
        "jackpot_winner": 0.001,
        "income_change": 0.004,
        "traveller": 0.006,
    },
    "reporting_threshold": 10_000,      # FINTRAC large cash / 24-hour rule
}
