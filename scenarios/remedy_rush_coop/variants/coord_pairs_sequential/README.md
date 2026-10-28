# coord_pairs_sequential

**Base:** `remedy_rush_coop_multi_agent` (see `../../base.md`)
**CFG:** falls back to `scenarios/remedy_rush_coop/base.cfg`.

## Idea

Sequential variant of `coord_pairs`: only **one** paired pickup exists in the
map at any time. The next pair only spawns once the current one is resolved
— either both vials get collected, or one vial is collected and its partner
is despawned after a short timeout.

This removes the standing buffet of pickups in `coord_pairs` and forces the
agents to focus on (and coordinate around) a single coop opportunity at a
time, which in turn produces a much sparser, easier-to-attribute reward
signal for credit assignment / curriculum experiments.

## Differences from coord_pairs

1. **`MAX_PAIRS = 1`, `INITIAL_PAIRS = 1`.** At most one pair is alive in
   the map at any moment. The map opens with a single pair already spawned.
2. **No periodic `SpawnPair` loop.** Script 20 (the `delay(PAIR_SPAWN_RATE)`
   refill loop from `coord_pairs`) is removed entirely. New pairs are
   spawned reactively from the pickup handlers instead.
3. **Reactive spawning in script 10 / 11.**
     - When the *second* vial of the current pair is picked, script 10
       awards the full coop reward (`+2`), clears the pair, and immediately
       calls `SpawnPair()` to seed the next one.
     - When only one vial has been picked, script 10 awards the partial
       reward (`+1`) and arms script 11 with a `DESPAWN_DELAY`-tick timer.
       When the timer fires, the unpicked partner is removed and the next
       pair is spawned.
4. **`DESPAWN_DELAY = 15` ticks.** After one half of a pair is collected,
   the other half stays alive for 15 tics (~0.43s at the default 35Hz) to
   give the partner agent a chance to grab it for the bonus reward, then
   despawns.

Reward shaping (`+1` solo, `+2` coop) and pair geometry (`PAIR_HALF_GAP`,
`PAIR_EXCLUSION`, sample bounds) are unchanged.

## Tuning

In `scripts.acs`:

- **`DESPAWN_DELAY`** — how long (in tics) the partner vial lingers after
  the first half is picked. Larger values are more forgiving for the
  trailing agent; smaller values punish poor synchronisation harder.
- **`PAIR_HALF_GAP`** — distance from pair midpoint to each vial.
- **`PAIR_EXCLUSION`** — minimum spacing between sequentially-spawned pair
  midpoints (with `MAX_PAIRS = 1` this only excludes the *previous* pair's
  location, which keeps the agents from camping a single hot-spot).

After editing, rebuild:

```bash
uv run python scripts/build_scenario.py remedy_rush_coop coord_pairs_sequential --force
```

## Expected training behavior

Sparser, more sequential reward stream than `coord_pairs`. Each "trial" is
clearly delimited (first vial picked → outcome within 15 tics), which
should make per-pair credit assignment cleaner at the cost of lower overall
throughput of reward per episode. Useful when studying coordination
dynamics or when `coord_pairs`' dense reward washes out the contribution
of the cooperative bonus.
