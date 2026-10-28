# coord_pairs

**Base:** `remedy_rush_coop_multi_agent` (see `../../base.md`)
**CFG:** falls back to `scenarios/remedy_rush_coop/base.cfg` — no per-variant
overrides. Add a `scenario.cfg` here to override (full replacement, not a patch).

## Idea

Pure-pair remedy rush. Remove the dense distractor stream that exists in the
base so that *every* pickup on the floor is one half of a linked armor pair.
Both halves must be grabbed inside `DESPAWN_DELAY = 5` tics or the partner
vanishes. Forces strict spatial + temporal coordination between agents: no
easy solo reward left on the floor to fall back on.

## Differences from base

1. **Paired pickup item** changed: `HealthBonus → ArmorBonus`. Purely
   cosmetic in gameplay terms — same per-pickup reward, just a different
   sprite class on the floor. Kept to match the original coord variant.
2. **Distractor population removed entirely.** Deleted state
   (`n_distractions`, `spawn_delay`, `object_tid`), the `SpawnItem` helper,
   the 40-item distractor burst in `OPEN`, and the `while(1)` armor
   replenishment loop. Net effect: the only objects on the map are paired
   `ArmorBonus`es; no background clutter and no unbounded accumulation.
3. **`script 4` (solo armor pickup → `cost += 1`) is deleted.** This variant
   never writes to the `cost` global, so any reward shaping that consumes
   `cost` (e.g. distractor-pickup penalty) will be identically zero. Plan
   shaping accordingly — the variant is only distinguishable from a
   `reward`-only signal.
4. **Flat pickup reward.** Every vial gives `+1.0`, first or second half
   alike. Per-pair total: `1` solo, `2` coord. Because both halves pay the
   same, pickups cannot be classified as solo vs. coord from the reward
   stream, so the `r >= 1.5` coord classification used by the pickup metrics
   (`pickups/coop_rate`, trajectory-plot markers) does not apply to this
   variant; the shaped variants (`coord_pairs_mixed`, `_sequential`,
   `_big_armor`, ...) pay `+2.0` for the second half.
5. **Cosmetic only, no other logic change.** `AbsInt` inlined, shorter
   section comments. Pair spawn / despawn scripts are byte-equivalent to
   base.

## Expected training behavior

- Reward curve should climb more slowly than base: no solo-pickup floor, so
  early random policies get near-zero reward until one agent accidentally
  triggers a partner pickup inside the 5-tic window.
- Per-pair reward is `1` if only one half is grabbed, `2` if coordinated.
  Max theoretical throughput per episode ≈ 31 pairs × 2 = 62 (initial 10
  pairs + ~21 replenishments at `PAIR_SPAWN_RATE = 100` tics over the
  2100-tic episode), assuming every pair is fully coordinated.
- Good diagnostic: track fraction of pairs fully collected vs. fraction
  with only one half grabbed. The latter is pure waste and a proxy for
  coordination quality.

## Known caveats

- 5-tic despawn window (≈ 143 ms at 35 tic/s) across a 160-unit horizontal
  gap effectively requires `SPEED` to be pressed; without it, the partner
  vanishes before any second agent can close the distance.
- Because distractors are gone, the initial distribution of objects on the
  floor is sparser — exploration is strictly harder than base.
