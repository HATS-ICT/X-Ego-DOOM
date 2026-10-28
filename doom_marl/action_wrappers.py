"""Action-space wrappers for ViZDoom PettingZoo envs."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class ForcedForwardTurnOnlyWrapper(ParallelEnv):
    """Restrict the action space to {turn_left, turn_right} with MOVE_FORWARD
    permanently held and SPEED never held.

    The underlying ViZDoom env (with the four binary buttons
    ``TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, SPEED`` and ``simple_discrete=True``)
    exposes ``Discrete(16)`` over the joint button mask. This wrapper presents
    the policy with ``Discrete(2)`` (0 = left, 1 = right) and translates each
    choice into the joint-mask index that has MOVE_FORWARD on, SPEED off, and
    exactly one of the turn buttons on.

    Bit order in the simple_discrete mask follows the ``available_buttons``
    declaration in ``base.cfg``:
        bit 0 = TURN_LEFT
        bit 1 = TURN_RIGHT
        bit 2 = MOVE_FORWARD
        bit 3 = SPEED
    so left -> 0b0101 = 5 and right -> 0b0110 = 6.
    """

    LEFT_INDEX = 0b0101
    RIGHT_INDEX = 0b0110

    def __init__(self, env: ParallelEnv):
        self.env = env
        self.possible_agents = list(env.possible_agents)
        self.agents = list(env.agents)
        self._action_space = spaces.Discrete(2)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["env"], name)

    def action_space(self, agent: str):
        return self._action_space

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    @property
    def num_agents(self) -> int:
        return getattr(self.env, "num_agents", len(self.possible_agents))

    def _remap(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for agent, a in actions.items():
            idx = int(np.asarray(a).item())
            if idx == 0:
                out[agent] = self.LEFT_INDEX
            elif idx == 1:
                out[agent] = self.RIGHT_INDEX
            else:
                raise ValueError(
                    f"ForcedForwardTurnOnlyWrapper expects action in {{0,1}}, got {idx}"
                )
        return out

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.env.agents)
        return obs, infos

    def step(self, actions: Dict[str, Any]):
        result = self.env.step(self._remap(actions))
        self.agents = list(self.env.agents)
        return result

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class LeftRightForwardWrapper(ParallelEnv):
    """Restrict the action space to {left, right, forward} with SPEED off.

    Like :class:`LeftRightForwardStandWrapper` but drops the stand-still
    option, so the policy head emits ``Discrete(3)``:
        0 = turn-left      -> 0b0001 = 1
        1 = turn-right     -> 0b0010 = 2
        2 = move-forward   -> 0b0100 = 4
    Useful when you want the agent to always be doing *something* (no
    no-op exploration loophole).
    """

    ACTION_TO_INDEX = {0: 0b0001, 1: 0b0010, 2: 0b0100}

    def __init__(self, env: ParallelEnv):
        self.env = env
        self.possible_agents = list(env.possible_agents)
        self.agents = list(env.agents)
        self._action_space = spaces.Discrete(3)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["env"], name)

    def action_space(self, agent: str):
        return self._action_space

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    @property
    def num_agents(self) -> int:
        return getattr(self.env, "num_agents", len(self.possible_agents))

    def _remap(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for agent, a in actions.items():
            idx = int(np.asarray(a).item())
            if idx not in self.ACTION_TO_INDEX:
                raise ValueError(
                    f"LeftRightForwardWrapper expects action in {{0,1,2}}, got {idx}"
                )
            out[agent] = self.ACTION_TO_INDEX[idx]
        return out

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.env.agents)
        return obs, infos

    def step(self, actions: Dict[str, Any]):
        result = self.env.step(self._remap(actions))
        self.agents = list(self.env.agents)
        return result

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class LeftRightForwardStandWrapper(ParallelEnv):
    """Restrict the action space to {left, right, forward, stand} with SPEED off.

    Underlying ``Discrete(16)`` over the joint button mask of
    ``TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, SPEED`` is mapped from a
    ``Discrete(4)`` policy action:
        0 = turn-left      -> 0b0001 = 1
        1 = turn-right     -> 0b0010 = 2
        2 = move-forward   -> 0b0100 = 4
        3 = stand-still    -> 0b0000 = 0
    """

    ACTION_TO_INDEX = {0: 0b0001, 1: 0b0010, 2: 0b0100, 3: 0b0000}

    def __init__(self, env: ParallelEnv):
        self.env = env
        self.possible_agents = list(env.possible_agents)
        self.agents = list(env.agents)
        self._action_space = spaces.Discrete(4)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["env"], name)

    def action_space(self, agent: str):
        return self._action_space

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    @property
    def num_agents(self) -> int:
        return getattr(self.env, "num_agents", len(self.possible_agents))

    def _remap(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for agent, a in actions.items():
            idx = int(np.asarray(a).item())
            if idx not in self.ACTION_TO_INDEX:
                raise ValueError(
                    f"LeftRightForwardStandWrapper expects action in {{0,1,2,3}}, got {idx}"
                )
            out[agent] = self.ACTION_TO_INDEX[idx]
        return out

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.env.agents)
        return obs, infos

    def step(self, actions: Dict[str, Any]):
        result = self.env.step(self._remap(actions))
        self.agents = list(self.env.agents)
        return result

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
