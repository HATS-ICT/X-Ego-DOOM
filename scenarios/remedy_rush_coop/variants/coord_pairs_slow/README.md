# coord_pairs_slow

**Base:** `remedy_rush_coop_multi_agent` (see `../../base.md`)
**CFG:** falls back to `scenarios/remedy_rush_coop/base.cfg` — no per-variant
overrides. Add a `scenario.cfg` here to override (full replacement, not a patch).

## Idea

Identical to `coord_pairs` except: (1) the partner vial lingers 3× longer after
the first pickup, and (2) a single agent that grabs both halves earns nothing for
the second. Together these widen the coordination window while blocking the
trivial exploit of one agent sweeping both vials of a pair alone.

## Differences from coord_pairs

1. **`DESPAWN_DELAY` raised from 5 → 15 tics.** After the first vial of a pair
   is picked, the partner stays on the floor for 15 tics (≈ 429 ms at 35 tic/s)
   instead of 5 (≈ 143 ms). Agents no longer need `SPEED` pressed to beat the
   window; walking speed is now sufficient to cross the 160-unit gap.
2. **Same-agent double-grab is unrewarded.** Script 10 records which player
   (`PlayerNumber()`) picked each vial via new `pair_a_picker` / `pair_b_picker`
   arrays. If the second vial is taken by the same player who took the first,
   `reward` is not incremented (the comment marks the suppressed path). The first
   vial still yields `+1` regardless — only the completion bonus is withheld.

Per-pair reward outcomes:

| Outcome                         | reward delta |
|---------------------------------|:------------:|
| One vial taken, partner expires |     +1       |
| Both vials by different agents  |     +3       |
| Both vials by the same agent    |     +1       |

The last row is new relative to `coord_pairs`: a solo sweeper can no longer earn
+3 by running across both halves quickly.

## Expected training behavior

- Early training should see more complete pairs than `coord_pairs` simply because
  the 15-tic window is achievable by walking. Expect reward to climb faster in the
  first phase.
- The same-agent suppression prevents a degenerate policy where one fast agent
  vacuums all pairs while the other idles. Once agents discover that solo sweeping
  caps out at +1, selection pressure should push toward genuine split pickups.
- Good diagnostics:
  - Fraction of pairs completed (both halves picked) — should be higher than
    `coord_pairs` baseline due to the wider window.
  - Fraction of completed pairs taken by same vs. different agent — "same-agent
    completions" should trend toward zero as training converges.
  - Per-step reward distribution: `r == 2.0` step ⇒ coord completion; `r == 1.0`
    ⇒ first pickup (or solo second); `r == 0` ⇒ no pickup that step.

## Known caveats

- `PlayerNumber()` is evaluated at the moment the pickup script fires. In an
  async multi-agent setup (two ZDoom instances per agent), both sides share the
  same global arrays but run in separate game contexts — `PlayerNumber()` is
  always 0 in single-player-per-instance configs. The same-agent suppression only
  fires as intended in a single ZDoom instance with multiple human/bot players in
  the same game. Verify that your async mode routes all agents into the same
  instance if you want the suppression to be effective.
- `pair_a_picker` / `pair_b_picker` are initialised to `-1` in `SpawnPair` and
  are only written on actual pickup. The guard `partner_picked` ensures the picker
  comparison is only reached when the partner field is valid (non-−1).
