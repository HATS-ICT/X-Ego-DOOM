<div align="center">

# Egocentric Team AI

**Learning a shared tactical picture from the operator's view: cross-egocentric
contrastive alignment for decentralized multi-agent reinforcement learning in
ViZDoom.**

Multi-agent RL · Contrastive representation learning · Shared situational awareness · ViZDoom

<img src="docs/media/hero.avif" width="1000" alt="Two agents' egocentric views stacked on the left and a top-down map of the room on the right, playing in sync as the agents split up to collect both halves of linked pickup pairs.">

*Two decentralized agents, each acting from its own first-person view. The
top-down map is for the reader only.*

[Paper (AAAI SSS 2026)](https://ojs.aaai.org/index.php/AAAI-SS/article/view/42516) ·
[X-Ego prior work](https://arxiv.org/abs/2510.19150) ·
[X-Ego-CS dataset](https://huggingface.co/datasets/wangyz1999/X-EGO-CS)

</div>

## Research overview

Good human teams rarely coordinate by talking constantly. A fire team clearing
a building moves as one because each member carries a mental model of what the
others are doing, inferred from their own limited first-person view. Command
centers make that shared model explicit as a **Common Operating Picture
(COP)**; for the operator on the ground it stays implicit, a picture each team
member constructs for themselves.

AI research tends to sit at one of two extremes of this problem:

- **The broadcast perspective.** Video-understanding and sports-analytics
  models reason over third-person, "god's eye" footage. They see formations,
  but not the partial observability an embedded operator actually faces.
- **The isolated egocentric perspective.** Embodied agents and standard
  multi-agent RL act from first-person inputs, but under centralized training
  with decentralized execution (CTDE) a centralized critic with access to
  global state can carry the team-level reasoning during training. Nothing
  forces each decentralized policy to encode an understanding of the team in
  its own representation.

**Egocentric Team AI** is the research direction of closing this gap: agents
that build an *implicit, distributed* COP directly from their own egocentric
observations, so that decentralized policies can coordinate without explicit
communication or privileged state.

## From passive video to active control

This repository is the experimental testbed for that direction. It builds on
**Cross-Ego Contrastive Learning (CECL)**, introduced in our prior work
[X-Ego](https://arxiv.org/abs/2510.19150). CECL aligns the representations of
synchronized first-person views from teammates: two players observing the same
moment of the same round are a positive pair, so the encoder is pushed toward
features shared at the team level (formation, timing, tactical phase) rather
than viewpoint-specific detail. Trained on
[X-Ego-CS](https://huggingface.co/datasets/wangyz1999/X-EGO-CS), 124 hours of
synchronized egocentric video from professional *Counter-Strike 2* matches,
CECL improved prediction of teammate, enemy, and global round state from a
single player's video (up to +28% relative on teammate state).

Passive prediction, however, is only a precursor to decision-making. Here we
take CECL into **cooperative multi-agent reinforcement learning**, as an
auxiliary representation objective inside a standard CTDE algorithm (MAPPO).
A projection head on each agent's vision encoder produces the embeddings that
are aligned across teammates. The two losses are optimized jointly,
`L = L_MAPPO + λ · L_CECL`. Execution stays fully decentralized: each agent
acts from its own pixels, and alignment only shapes what its encoder learns to
represent.

In the implementation, contrastive pairs are defined over time within an
episode:

- **positives**: embeddings from the same episode within `w_pos` steps,
  across all agents. A teammate's view of the same moment is a positive;
- **negatives**: embeddings from other episodes, or more than `w_neg` steps
  apart;
- **gap zone**: pairs in between are excluded, so nearby-but-not-simultaneous
  frames are neither pulled together nor pushed apart.

The default objective is the pairwise sigmoid loss used by CECL
([Zhai et al. 2023](https://arxiv.org/abs/2303.15343)); a multi-positive
InfoNCE variant is also available.

## The testbed: coordinated pairs in ViZDoom

We need a task where reward *requires* coordination that agents can only
achieve by understanding each other from their own views. The scenario is a
cooperative variant of ViZDoom's Remedy Rush. Agents spawn in a room scattered
with **linked pickup pairs**: the two halves of a pair sit 160 units apart, and
once one half is collected the other vanishes after 5 tics (≈ 143 ms). Each
pickup scores +1, so a completed pair is worth twice a lone half, and only a
team that splits up and arrives almost simultaneously completes pairs.

Agents see 72×96 RGB first-person frames and choose among 16 actions
(turn left/right, move forward, run). Scenario variants isolate different
aspects of the problem:

| Variant | What changes |
| --- | --- |
| `coord_pairs` | The main task: linked pairs, +1 per pickup, no solo distractor items |
| `coord_pairs_sequential` | One pair alive at a time, respawned once resolved (sparse, easy credit assignment) |
| `coord_pairs_big_armor` | Enlarged pickup hitboxes |
| `coord_pairs_mixed` | The two halves of each pair are visually distinct items |
| `coord_pairs_slow` | Partner lingers 3× longer, but one agent grabbing both halves earns nothing extra |
| `ultracoord` | +5 completion bonus and a wider 200-unit gap |

Apart from `coord_pairs`, the variants pay more for the second half of a pair
than the first (+2, or +5 in `ultracoord`), so each pickup's reward reveals
whether it completed a pair. On those variants the fraction of coordinated
pickups is logged as `pickups/coop_rate`, a direct coordination metric
alongside episode reward. Each variant's design and expected training behavior
is documented in
[`scenarios/remedy_rush_coop/variants/`](scenarios/remedy_rush_coop/variants/).

The paper frames three questions this testbed is built to study:

- **Representation internalization:** does cross-ego alignment make
  decentralized policies encode team state without relying on a centralized
  critic?
- **Coordination efficiency:** does representation-level alignment improve
  coordination under partial observability?
- **Generalization:** do aligned representations transfer across task
  variations and environment configurations?

## Repository and reproduction

```text
run.py                  train one experiment config (baseline or cecl candidate)
eval.py                 replay checkpoints; POV + top-down videos, A/B comparison
doom_marl/
├── algorithms/         Contrastive MAPPO and the cross-ego contrastive loss
├── models/             CNN, small ResNet, and DINOv3 vision encoders
├── task.py             ViZDoom PettingZoo → BenchMARL task adapter
└── experiment.py       training loop, episode/step tagging for contrastive pairs
experiments/            one YAML per experiment, each with a baseline and a cecl candidate
scenarios/              ACS sources for the scenario and its variants (built on demand)
wheels/                 the ViZDoom build used in the experiments
```

The code builds on [BenchMARL](https://github.com/facebookresearch/BenchMARL)
and [ViZDoom](https://github.com/Farama-Foundation/ViZDoom). The bundled
ViZDoom wheel (built from [our fork](https://github.com/HATS-ICT/ViZDoom/tree/v1.0-marl)
with multi-agent and object-information extensions) is
built for **Linux x86-64 and Python 3.11**; training expects an NVIDIA GPU.
Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Every experiment config defines two candidates with identical environment and
training settings, `baseline` (MAPPO) and `cecl` (MAPPO + CECL):

```bash
uv run python run.py experiments/remedy_rush_coop_coord_pairs.yaml
uv run python run.py experiments/remedy_rush_coop_coord_pairs.yaml --candidate cecl
```

Runs log to [Weights & Biases](https://wandb.ai) and save checkpoints under
`outputs/checkpoints/`. A checkpoint can be replayed through the environment
stack it was trained on, alone or against another:

```bash
uv run python eval.py --experiment <run-name-substring>
uv run python eval.py --compare <baseline_checkpoint.pt> <cecl_checkpoint.pt>
```

<details>
<summary><strong>Configuration, other algorithms, and differences from the paper</strong></summary>

Every option is documented in
[`experiments/config_commented.yaml`](experiments/config_commented.yaml).
Top-level keys can be overridden with `--set key=value`, for example
`--set num_envs=8 model.rnn_type=none`. A candidate's `algo` selects which
`<algo>_kwargs` block it reads; `cecl` reads `mappo_kwargs` for its PPO core
and `contrastive_kwargs` for the alignment loss.

Other BenchMARL baselines (`ippo`, `qmix`, `vdn`, `iql`) run by overriding the
candidate's algorithm. The `coord_pairs_sequential` and `coord_pairs_big_armor*`
configs carry hyperparameters for all of them:

```bash
uv run python run.py experiments/remedy_rush_coop_coord_pairs_sequential.yaml \
  --set candidates.baseline.algo=ippo
```

`scripts/test_wad.py <config.yaml>` checks that a scenario builds and steps
correctly before training. The DINOv3 encoder config uses a gated Hugging Face
model; request access on its model page and run `huggingface-cli login` first.

The code has moved on since the paper was written:

- positives use a temporal window with a gap zone rather than exact timestep
  matches, and the sigmoid loss uses a fixed temperature without a learned bias;
- the contrastive projection head branches off the policy encoder alongside
  the action head (the actor does not read the projected embedding), and the
  MAPPO critic has its own encoder;
- experiments use two agents;
- the paper also names MASAC as a CTDE testbed; this release implements CECL
  on MAPPO only;
- the linear-probe diagnostics proposed in the paper are not included.

</details>

## Publication

Hans, S., Wang, Y., & Ustun, V. (2026). *Egocentric Team AI: Enabling Tactical
Reasoning from the Operator's View*. Proceedings of the AAAI Symposium Series,
8(1), 45–49. [doi:10.1609/aaaiss.v8i1.42516](https://doi.org/10.1609/aaaiss.v8i1.42516).

```bibtex
@inproceedings{hans2026egocentric,
  title     = {Egocentric Team {AI}: Enabling Tactical Reasoning from the Operator's View},
  author    = {Hans, Soham and Wang, Yunzhe and Ustun, Volkan},
  booktitle = {Proceedings of the AAAI Symposium Series},
  volume    = {8},
  number    = {1},
  pages     = {45--49},
  year      = {2026},
  doi       = {10.1609/aaaiss.v8i1.42516}
}
```

CECL and the X-Ego-CS dataset were introduced in: Wang, Y., Hans, S., &
Ustun, V. (2025). *X-Ego: Acquiring Team-Level Tactical Situational Awareness
via Cross-Egocentric Contrastive Video Representation Learning*.
[arXiv:2510.19150](https://arxiv.org/abs/2510.19150).

The project or effort depicted was or is sponsored by the U.S. Army Combat
Capabilities Development Command – Soldier Center under contract number
W912CG-24-D-0001. The content of the information does not necessarily reflect
the position or the policy of the Government, and no official endorsement
should be inferred.

## License

The code in this repository is released under the [MIT License](LICENSE).
The ViZDoom wheel in `wheels/` is a build of our fork
[HATS-ICT/ViZDoom](https://github.com/HATS-ICT/ViZDoom/tree/v1.0-marl)
(release `v1.0-marl`, commit `8fc4d363a7`) and is not covered by this license:
ViZDoom's own code is MIT, while its ZDoom-based engine carries ZDoom's
licenses; see the fork for source and terms.
