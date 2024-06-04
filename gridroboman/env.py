from dataclasses import dataclass
from typing import *
import gymnasium as gym
import os
import random

import numpy as np

GRID_SIZE = 7
MAX_TIME = 50


@dataclass
class ObjData:
    x: int
    y: int
    obj_above: Optional[int]  # The object above this one
    obj_below: Optional[int]  # The object below this one


class BaseGridrobomanEnv(gym.Env):

    def __init__(self, render_mode: str):
        self.action_space = gym.spaces.Discrete(7)
        self.render_mode = render_mode
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.objs = [ObjData(0, 0, None, None) for _ in range(3)]
        self.lifted_obj_idx: Optional[int] = None
        self.timer = 0

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action)

        # Move
        if action == 1:
            self.agent_pos = (self.agent_pos[0], max(0, self.agent_pos[1] - 1))
        if action == 2:
            self.agent_pos = (
                self.agent_pos[0],
                min(GRID_SIZE - 1, self.agent_pos[1] + 1),
            )
        if action == 3:
            self.agent_pos = (max(0, self.agent_pos[0] - 1), self.agent_pos[1])
        if action == 4:
            self.agent_pos = (
                min(GRID_SIZE - 1, self.agent_pos[0] + 1),
                self.agent_pos[1],
            )
        
        # Pick up
        if action == 5:
            if self._top_obj_idx(self.agent_pos) is not None:
                self._pick_up_obj(self.agent_pos)
        # Drop
        if action == 6:
            if self.lifted_obj_idx is not None:
                self._place_obj(self.agent_pos)

        if self.lifted_obj_idx is not None:
            (self.objs[self.lifted_obj_idx].x, self.objs[self.lifted_obj_idx].y) = self.agent_pos

        self.timer += 1
        trunc = self.timer == MAX_TIME

        return self._gen_obs(), 0.0, False, trunc, self._gen_info()

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        random.seed(seed)
        self.agent_pos = (
            random.randrange(0, GRID_SIZE),
            random.randrange(0, GRID_SIZE),
        )
        used_pos = {self.agent_pos}
        for i in range(3):
            pos = (random.randrange(0, GRID_SIZE), random.randrange(0, GRID_SIZE))
            while pos in used_pos:
                pos = (random.randrange(0, GRID_SIZE), random.randrange(0, GRID_SIZE))
            used_pos.add(pos)
            self.objs[i].x = pos[0]
            self.objs[i].y = pos[1]
            self.objs[i].obj_above = None
            self.objs[i].obj_below = None
        return self._gen_obs(), self._gen_info()

    def _gen_obs(self) -> np.ndarray:
        obs = np.zeros([11], dtype=float)
        (obs[0], obs[1]) = self.agent_pos
        for i in range(3):
            obj = self.objs[i]
            obs[2 + i * 3] = obj.x
            obs[2 + i * 3 + 1] = obj.y
            status = 0
            if obj.obj_above is not None:
                status = -1
            if self.lifted_obj_idx == i or obj.obj_below is not None:
                status = 1
            obs[2 + i * 3 + 2] = status
        return obs
    
    def _gen_info(self) -> dict[str, np.ndarray]:
        mask = np.zeros([7], dtype=np.int8)
        mask[1] = self.agent_pos[1] == 0
        mask[2] = self.agent_pos[1] == GRID_SIZE - 1
        mask[3] = self.agent_pos[0] == 0
        mask[4] = self.agent_pos[0] == GRID_SIZE - 1
        mask[5] = self._top_obj_idx(self.agent_pos) is None
        mask[6] = self.lifted_obj_idx is None
        return {
            "action_mask": mask
        }

    def _top_obj_idx(self, pos: Tuple[int, int]) -> Optional[int]:
        """
        Returns the index of the top object at this position.
        """
        for i, obj in enumerate(self.objs):
            if (
                self.lifted_obj_idx != i
                and obj.obj_above is None
                and pos == (obj.x, obj.y)
            ):
                return i
        return None

    def _place_obj(self, pos: Tuple[int, int]):
        """
        Places the lifted object at this position.
        """
        assert self.lifted_obj_idx is not None
        top_obj_idx = self._top_obj_idx(pos)
        if top_obj_idx is not None:
            self.objs[top_obj_idx].obj_above = self.lifted_obj_idx
            self.objs[self.lifted_obj_idx].obj_below = top_obj_idx
        self.lifted_obj_idx = None

    def _pick_up_obj(self, pos: Tuple[int, int]):
        """
        Picks up the top object at this position and sets it as the lifted object.
        """
        top_obj_idx = self._top_obj_idx(pos)
        assert top_obj_idx is not None
        new_top_obj_idx = self.objs[top_obj_idx].obj_below
        if new_top_obj_idx is not None:
            self.objs[top_obj_idx].obj_below = None
            self.objs[new_top_obj_idx].obj_above = None
        self.lifted_obj_idx = top_obj_idx

    def render(self):
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                symbol = " "
                if self.agent_pos == (x, y):
                    symbol = "*"
                for i, color in enumerate(["R", "G", "B"]):
                    if i == self._top_obj_idx((x, y)):
                        symbol = color
                print(f"[{symbol}]", end="")
            print("")


if __name__ == "__main__":
    import time

    env = BaseGridrobomanEnv(render_mode="human")
    obs_, info = env.reset()
    action_space = env.action_space
    for _ in range(100):
        action = action_space.sample(1 - info["action_mask"])
        obs_, rew_, done_, trunc_, info = env.step(action)
        os.system("clear")
        env.render()
        time.sleep(0.1)
