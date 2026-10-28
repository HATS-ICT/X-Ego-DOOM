# ultracoord

**Base:** `remedy_rush_coop_multi_agent` (see `../../base.md`)
**CFG:** falls back to `scenarios/remedy_rush_coop/base.cfg` — no per-variant overrides.

## Idea

Extends `coord_pairs` with a strong completion bonus: collecting the *second* vial
in a pair (within `DESPAWN_DELAY`) scores **+5** instead of +1, while a solo first
pick still scores +1. This amplifies the incentive to coordinate — agents that
consistently complete pairs earn ~3× more than agents that only ever grab the first
half.

Pair gap widened to **200 units** (`PAIR_HALF_GAP = 100`) to make simultaneous
arrival harder, demanding tighter spatial planning.

## Differences from coord_pairs

1. **`PAIR_HALF_GAP` 80 → 100** — vials are 200 units apart instead of 160.
2. **Completion bonus in script 10** — if the partner vial was already picked,
   `reward += 5.0`; otherwise `reward += 1.0`.
3. Everything else identical: no distractors, `ArmorBonus` items, same despawn
   window, same pair budget.

## Reward structure

| Event | Reward |
|---|---|
| First pick in a pair | +1 |
| Second pick (completes pair) | +5 |
| Solo pick (partner despawns) | +1 only |

Max theoretical per fully-coordinated pair: **+6**. A pair where only one half
is grabbed yields **+1** — a strong signal to avoid half-coordination.

## Expected training behavior

- Early random policies get near-zero reward (same as coord_pairs).
- Once agents learn to complete pairs the reward signal jumps sharply, providing
  a steep gradient toward tighter coordination than coord_pairs.
- Track completion rate and mean reward-per-pair (should approach 6 for a
  well-coordinated policy).
