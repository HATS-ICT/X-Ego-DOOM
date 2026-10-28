# coord_pairs_big_armor

**Base:** `remedy_rush_coop_multi_agent` (see `../../base.md`)
**CFG:** falls back to `scenarios/remedy_rush_coop/base.cfg`.

## Idea

Identical to `coord_pairs` except each paired pickup is an enlarged
`ArmorBonus`: pickup hitbox `Radius`/`Height` and sprite `Scale` are bumped
in ACS via `SetActorProperty` immediately after spawn. The actor class
remains the built-in `ArmorBonus`, so all downstream consumers that filter
by `o.name == "ArmorBonus"` (e.g. the pettingzoo wrapper's `info["objects"]`
list, used by the training-video topdown panel) continue to work unmodified.

Reward shaping, despawn timing, and pair geometry are unchanged.

## Differences from coord_pairs

1. **Per-vial `SetActorProperty` calls in `SpawnPair`.** After `Spawn(...)`,
   each vial has `APROP_Radius`, `APROP_Height`, `APROP_ScaleX`,
   `APROP_ScaleY` overridden:
     - `Radius`: 20 → 48 (default → 2.4×)
     - `Height`: 16 → 48
     - `Scale`:  1.0 → 4.0 (visual-only)

Everything else — scripts, reward shaping, pair counts, gap, despawn delay —
is byte-equivalent to `coord_pairs`.

## Tuning

Edit the four `SetActorProperty(tid_*, APROP_*, ...)` lines in `scripts.acs`:

- **`APROP_Radius` / `APROP_Height`** — pickup-trigger cylinder. Larger
  values make the pair easier to grab (agent can pick up from farther away).
  Note: at `PAIR_HALF_GAP = 80`, going past `Radius ~64` will start to make
  the two halves' hitboxes touch and effectively become one giant pickup.
- **`APROP_ScaleX` / `APROP_ScaleY`** — purely cosmetic sprite size; does
  not affect the pickup hitbox or the automap rendering.

After editing, rebuild:

```bash
uv run python scripts/build_scenario.py remedy_rush_coop coord_pairs_big_armor --force
```

## Expected training behavior

Larger pickup radius makes coordination *easier* than `coord_pairs` — agents
no longer need to align as precisely with the vial centre, and the wider
sprite is more visible at the 72×96 input resolution. Useful as a
difficulty-easing comparator: if `coord_pairs` plateaus, `coord_pairs_big_armor`
should converge faster.
