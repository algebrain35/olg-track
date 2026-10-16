"""Assigns behaviour overlays to players and sets up multi-player group
scenarios (mule rings, chip-dumping pairs)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .behaviours import AML_TYPOLOGIES, HARD_NEGATIVES, ChipDumping, HarmEscalation, MuleMember
from .models import Player
from .reference import GAMBLING_ARCH, ONLINE_ARCH, VENUE_ARCH, lognormal_median, rand_id

if TYPE_CHECKING:
    from .engine import Simulator


def assign_behaviours(players: list[Player], rng, cfg: dict, sim: "Simulator"):
    n, n_days = len(players), sim.n_days
    taken: set[int] = set()

    def pool(archetypes):
        return [i for i, p in enumerate(players)
                if i not in taken and p.first_day == 0 and p.traits.archetype in archetypes]

    def pick(archetypes, k, weight_by_intensity=False):
        cands = pool(archetypes)
        k = min(k, len(cands))
        w = None
        if weight_by_intensity and k:
            # harm is more likely among already-heavy players (Deng et al.: top spenders
            # are over-represented among self-excluders)
            w = np.array([(players[i].traits.activity_rate * players[i].traits.typical_stake) ** 0.25
                          for i in cands]) + 1e-9
            w = w / w.sum()
        chosen = [cands[i] for i in rng.choice(len(cands), k, replace=False, p=w)] if k else []
        taken.update(chosen)
        return chosen

    def count(prev):
        return max(1, int(round(prev * n)))

    def mid_onset(lo=0.35, hi=0.80):
        return int(rng.integers(int(lo * n_days), int(hi * n_days)))

    # AML: one primary typology each, sometimes a second compatible one
    for name, (cls, archs) in AML_TYPOLOGIES.items():
        for i in pick(archs, count(cfg["aml"][name])):
            onset = 0 if name == "profile_mismatch" else mid_onset()
            players[i].behaviours.append(cls(onset, rng))
            if rng.random() < cfg["aml_second_typology_prob"]:
                # profile_mismatch / sudden_escalation redefine the baseline, so never stack them
                others = [k for k, (_c, a) in AML_TYPOLOGIES.items()
                          if k != name and players[i].traits.archetype in a
                          and k not in ("profile_mismatch", "sudden_escalation")]
                if not others:
                    continue
                second = others[int(rng.integers(len(others)))]
                o2 = min(n_days - 7, onset + int(rng.integers(0, 14)))
                players[i].behaviours.append(AML_TYPOLOGIES[second][0](o2, rng))

    # Mule rings
    for r in range(max(1, int(round(cfg["mule_rings_per_1000"] * n / 1000)))):
        size = int(rng.integers(cfg["mule_ring_size"][0], cfg["mule_ring_size"][1] + 1))
        members = pick(ONLINE_ARCH, size)
        hub = (rand_id("INS", rng), "EWALLET")
        dev = rand_id("DEV", rng)
        onset = mid_onset()
        for i in members:
            players[i].behaviours.append(
                MuleMember(onset + int(rng.integers(0, 7)), rng, f"RING{r:03d}", hub, dev))

    # Chip-dumping pairs (same home region, same casino table)
    for k in range(max(1, int(round(cfg["chip_dumping_pairs_per_1000"] * n / 1000)))):
        cands = pool(VENUE_ARCH)
        if len(cands) < 2:
            break
        a = cands[int(rng.integers(len(cands)))]
        same = [i for i in cands if i != a and players[i].profile.home_region == players[a].profile.home_region]
        if not same:
            continue
        b = same[int(rng.integers(len(same)))]
        taken.update([a, b])
        onset = mid_onset(0.3, 0.6)
        venue = sim.venues_for(players[a].profile.home_region, "table")[0]
        schedule = {}
        for d in sorted(rng.choice(np.arange(onset, n_days), int(rng.integers(4, 11)), replace=False)):
            st = sim.at(int(d), int(rng.integers(19, 23)), int(rng.integers(60)))
            schedule[int(d)] = (venue, st, lognormal_median(4000, 0.5, rng), rng.uniform(0.85, 1.0))
        gid = f"PAIR{k:03d}"
        players[a].behaviours.append(ChipDumping(onset, rng, gid, "dumper", schedule))
        players[b].behaviours.append(ChipDumping(onset, rng, gid, "receiver", schedule))

    # Gambling harm
    for sev, prev in cfg["harm"].items():
        for i in pick(GAMBLING_ARCH, count(prev), weight_by_intensity=True):
            players[i].behaviours.append(HarmEscalation(mid_onset(0.30, 0.75), rng, sev, n_days))

    # Hard negatives
    for name, (cls, archs) in HARD_NEGATIVES.items():
        for i in pick(archs, count(cfg["hard_negatives"][name])):
            onset = 0 if name == "legit_vip" else mid_onset(0.3, 0.9)
            players[i].behaviours.append(cls(onset, rng))
