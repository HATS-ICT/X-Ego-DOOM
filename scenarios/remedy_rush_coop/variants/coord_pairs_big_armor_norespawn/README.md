# coord_pairs_big_armor_norespawn

**Base:** `remedy_rush_coop_multi_agent` (see `../../base.md`)
**CFG:** falls back to `scenarios/remedy_rush_coop/base.cfg`.

## Idea

Like `coord_pairs_big_armor`, but items **never respawn**. At map open, **`INITIAL_PAIRS` (10)** enlarged
`ArmorBonus` pairs are placed. Once collected or despawned, nothing replaces them.

## Per-agent “exclusion” (ACS-only, reward shaping)

**Note (ViZDoom):** A DECORATE `CustomInventory` whose `Pickup` state calls
`ACS_ExecuteWithResult` was tried so the engine could **reject** pickup (walk-through). In practice the
**stock ViZDoom** binary **segfaults** as soon as a pickup script runs that way and an agent overlaps a
vial during `game.init()` (e.g. first-frame pickup). For stability this variant uses **enlarged
`ArmorBonus`** and **thing specials** only — **no** `decorate.txt` lump.

**Behavior:** the **same player** who took the first vial in a pair **cannot receive the +2 coordination
reward** on the partner vial: that second grab adds **+0** reward (the item is still consumed by the
engine like a normal pickup). A **different** player still gets **+2** on the second vial. This steers
learning toward two agents sharing pairs without requiring a custom actor class that crashes the
runtime.

## Differences from coord_pairs_big_armor

1. **No respawn loop.** Script 20 is absent.
2. **Despawn window.** Script 11 removes unpicked pieces after **`DESPAWN_DELAY = 20`** tics.
3. **`INITIAL_PAIRS = 10`.** Matches reliable placement under `PAIR_EXCLUSION = 400` on this map.
4. **No `decorate.txt`.** (Would inject DECORATE; not used — see above.)

## Expected training behavior

Same as the big-armor variant, with a **finite** item pool. Agents that **split** work across vials get
higher per-episode return than one agent **soloing both halves** of a pair (which only yields **+1** for
that pair instead of **+3**).

## Rebuild

```bash
uv run python scripts/build_scenario.py remedy_rush_coop coord_pairs_big_armor_norespawn --force
```
