# coord_pairs_mixed

**Base:** `remedy_rush_coop_multi_agent` (see `../../base.md`)
**CFG:** falls back to `scenarios/remedy_rush_coop/base.cfg` — no per-variant
overrides. Add a `scenario.cfg` here to override (full replacement, not a patch).

## Idea

Identical to `coord_pairs` except each pair is heterogeneous: vial A (west) is an
`ArmorBonus` and vial B (east) is a `HealthBonus`. The two item types are visually
distinct on the floor, so agents can learn to associate the two different sprites as
belonging to the same pair. All reward logic and timing is unchanged.

## Differences from coord_pairs

1. **Mixed item types per pair.** `SpawnPair` now spawns `"ArmorBonus"` at
   `mid_x - PAIR_HALF_GAP` and `"HealthBonus"` at `mid_x + PAIR_HALF_GAP`.
   In `coord_pairs` both halves were `ArmorBonus`. The two types are functionally
   identical in gameplay terms (same pickup radius, same ACS trigger path) — the
   only difference is the sprite on the floor.

Everything else — `DESPAWN_DELAY`, reward shaping (+1 first / +2 second),
`MAX_PAIRS`, `PAIR_HALF_GAP`, spawn/despawn logic — is byte-equivalent to
`coord_pairs`.

## Expected training behavior

The visual asymmetry between the two halves gives agents an additional cue they
could use to specialise: one agent learns to seek `ArmorBonus` sprites and the
other learns to seek `HealthBonus` sprites, naturally splitting the map without
explicit role assignment. Whether this signal is exploitable depends on whether the
observation space resolves the colour difference at the target resolution (72×96).

## Known caveats

- At 72×96 the sprite colour difference between `ArmorBonus` (green) and
  `HealthBonus` (blue/purple) may be subtle after bilinear downsampling. If the
  specialisation hypothesis is the research question, consider increasing
  `obs_h`/`obs_w` so the colour cue is preserved.
