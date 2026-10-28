# remedy_rush_coop_multi_agent — base scenario

Source of truth: `scenarios/remedy_rush_coop/base.acs` and `base.cfg`
(pristine copies of the upstream ViZDoom scenario). Keep this document in
sync when you edit the base; variant-specific behavior lives in the header
comments of each file under `variants/`.

## Elevator pitch

Multi-agent "remedy rush". Every agent spawns with 1 HP, no inventory, at a
random position inside the agent-playable region of a single room. The floor
is littered with two kinds of pickups:

- **Paired health vials** (`HealthBonus`) — spawn in linked A/B pairs with a
  fixed horizontal gap. Picking one up rewards +1 and starts a short timer;
  if the partner is not grabbed within `DESPAWN_DELAY` tics it vanishes.
- **Solo armor bonuses** (`ArmorBonus`) — constantly replenishing background
  distractors. Picking one up has no reward but does increment a separate
  `cost` signal (i.e. it's useful as a shaping penalty if you want to train
  agents that ignore distractors).

The intended pressure: agents must coordinate to exploit the reward stream
(pair grabs) while resisting the easier, denser distractor stream.

## Action / observation surface (from `base.cfg`)

| Key                        | Value                                   |
|----------------------------|-----------------------------------------|
| `available_buttons`        | `TURN_LEFT`, `TURN_RIGHT`, `MOVE_FORWARD`, `SPEED` |
| `available_game_variables` | `HEALTH`, `POSITION_X`, `POSITION_Y`, `ANGLE` |
| `episode_timeout`          | 2100 tics (≈ 60 s at 35 tic/s)          |
| `screen_format`            | `CRCGCB` (planar RGB)                   |
| `mode`                     | `PLAYER`                                |
| rendering                  | HUD on; crosshair / weapon / decals / particles off; window hidden |

Note: no `STRAFE_LEFT` / `STRAFE_RIGHT`, no attack — agents move forward and
turn only. `SPEED` is the "run" toggle.

## Global ACS signals (indices 0–2)

These are the channels training code should read from each frame:

| Global      | Index | Units           | Semantics                                    |
|-------------|-------|-----------------|----------------------------------------------|
| `reward`    | 0     | float, monotonic | +1 per *single vial pickup* (both halves of a pair add +1 each, so max +2 per fully-collected pair) |
| `max_num_players` | 1 | int            | Set to 4 in `OPEN`; informational only       |
| `cost`      | 2     | int, monotonic  | +1 per distractor armor pickup              |

Both `reward` and `cost` are *accumulators* — the per-step delta is the
training signal. The PettingZoo wrapper already handles this differencing.

## Map geometry (ACS constants)

- Agent-playable rect: `x ∈ [-320, 352]`, `y ∈ [128, 1024]`
- Object-spawn rect:   `x ∈ [-608, 1056]`, `y ∈ [-224, 1312]`
  (much larger than the agent rect — many pickups live outside the initial
  spawn zone, forcing exploration)
- Pair geometry: two vials spawned at `(mid_x ± 80, mid_y)` — 160-unit
  horizontal gap, same y. Agents must split laterally to grab both halves.

## Spawn dynamics

### Initial state (`script 1 OPEN`)
- Reset `reward = cost = 0`, `max_num_players = 4`.
- Spawn `INITIAL_PAIRS = 10` vial pairs.
- Spawn `n_distractions = 40` armor bonuses.
- Kick off the pair-replenishment loop (`script 20`).
- Fall into the armor-replenishment loop: one new armor every
  `spawn_delay = 60` tics forever.

### Pair-replenishment loop (`script 20`)
Every `PAIR_SPAWN_RATE = 100` tics: if fewer than `MAX_PAIRS = 15` pairs are
active, try to spawn one more.

### Placement constraints (`SpawnPair`)
- Up to 15 slots tracked in parallel map-scope arrays.
- Midpoint must be at least `PAIR_EXCLUSION = 400` (Manhattan) from every
  active pair's midpoint — stops pairs from piling on top of each other.
- 100 attempts per spawn; if no valid slot found, that tick's spawn is
  skipped.
- `pair_gen` is a per-slot generation counter used to stop stale despawn
  timers from firing on a slot that's already been reused.

## Pickup mechanics

### Paired vial (`script 10`)
Triggered by `SetThingSpecial(..., ACS_Execute, 10, ...)` when either vial
of a pair is touched. Args: `(pair_idx, which_vial ∈ {0, 1}, gen)`.

1. Guard: slot still active and generation matches — otherwise stale.
2. `reward += 1`.
3. Mark that half of the pair as picked.
4. Fire `ACS_Execute(11, ...)` to start the partner-despawn timer.

### Despawn timer (`script 11`)
Runs once per pickup, `DESPAWN_DELAY = 5` tics after the first pickup.

- If nothing else has resolved the slot in the meantime, remove whichever
  partner is still on the floor (`Thing_Remove` on its TID).
- Free the slot either way (`pair_active[...] = 0`, `active_pair_count--`).

Net consequence: once *any* half of a pair is collected, the team has at
most 5 tics (≈ 143 ms at 35 tic/s) to grab the partner for a second +1.
After that the partner vanishes — wasted half a pair.

### Distractor armor (`script 4`)
`cost += 1`. No reward, no despawn logic — armor is stateless background
noise.

## Agent respawn (`script 2 ENTER`)
Fires each time a player enters the level (initial spawn and any respawn).

- Set player health to 1.
- Clear inventory.
- Randomise position inside the agent rect (rejection sampling via
  `SetActorPosition`, which returns false if the target is blocked).
- Randomise angle (256-step quantisation).

Because `SetActorProperty(0, APROP_Health, 1)` runs on respawn too, agents
are kept fragile across an episode — any damage taken is effectively fatal
— but the episode itself ends on `episode_timeout`, not on death.

## What this means for training

- Reward is *dense but noisy*: solo pickups already give +1, so a policy
  that ignores pairs and just vacuums the floor still gets reward. The
  pair-coordination pressure comes from the fact that the partner vanishes
  fast — leaving reward on the floor if you don't coordinate.
- `cost` is the knob for negative shaping: charge agents a penalty on
  distractor grabs and you select for agents that only take paired vials.
  Without any `cost`-coupling, distractors act as a capacity trap.
- The 4-button action set keeps the policy head small. `SPEED` matters —
  without it you won't beat the 5-tic despawn window for pairs whose
  partner is on the far side of the 160-unit gap.
- `POSITION_X`, `POSITION_Y`, `ANGLE` are always available in state, so
  privileged-info critics or auxiliary losses can cleanly condition on
  pose.

## Known quirks

- `max_num_players` is hard-coded to 4 in `OPEN` regardless of how many
  agents you actually launch. The PettingZoo wrapper is what enforces the
  real agent count — don't rely on this global.
- The armor-replenishment `while(1)` in `OPEN` means armor grows
  unboundedly in practice (`SpawnItem` has no cap, only pairs do). Long
  episodes accumulate floor clutter, which shifts the distractor baseline
  upward over time.
- `PAIR_HALF_GAP = 80.0` in `#define` but the inline comment on line 21
  says "each vial is 25 units from midpoint (50 unit gap total)" — the
  comment is stale; the actual gap is 160 units.
