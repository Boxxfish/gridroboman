from dataclasses import dataclass
from typing import *
import gymnasium as gym
import random

import numpy as np
import pygame

GRID_SIZE = 7
MAX_TIME = 50
CENTER = [(x, y) for x in range(2, 5) for y in range(2, 5)]
CORNERS = [
    (sx + x, sy + y)
    for x in range(0, 2)
    for y in range(0, 2)
    for sx in [0, 5]
    for sy in [0, 5]
]
WINDOW_SIZE = 600


@dataclass
class ObjData:
    x: int
    y: int
    obj_above: Optional[int]  # The object above this one
    obj_below: Optional[int]  # The object below this one


class BaseGridrobomanEnv(gym.Env):
    """
    Base class for Gridroboman tasks.
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 10,
    }

    def __init__(
        self,
        x_obj: Union[int, str] = -1,
        y_obj: Union[int, str] = -1,
        render_mode: Optional[str] = None,
    ):
        self.metadata["render_modes"] = ["human"]
        self.metadata["render_fps"] = 10
        color_map = {
            "red": 0,
            "green": 1,
            "blue": 2,
        }
        if isinstance(x_obj, str):
            self.x_obj = color_map[x_obj]
        else:
            self.x_obj = x_obj
        if isinstance(y_obj, str):
            self.y_obj = color_map[y_obj]
        else:
            self.y_obj = y_obj
        assert self.x_obj in [0, 1, 2], "X object is not in [0, 1, 2]"
        assert self.x_obj != self.y_obj, "X object and Y object must be different"

        self.observation_space = gym.spaces.Box(-1, 6, [11])
        self.action_space = gym.spaces.Discrete(7)
        self.render_mode = render_mode
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.objs = [ObjData(0, 0, None, None) for _ in range(3)]
        self.lifted_obj_idx: Optional[int] = None
        self.timer = 0

        self.screen = None
        self.clock = None
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            self.clock = pygame.time.Clock()

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
            if (
                self._top_obj_idx(self.agent_pos) is not None
                and self.lifted_obj_idx is None
            ):
                self._pick_up_obj(self.agent_pos)
        # Drop
        if action == 6:
            if self.lifted_obj_idx is not None:
                self._place_obj(self.agent_pos)

        if self.lifted_obj_idx is not None:
            (self.objs[self.lifted_obj_idx].x, self.objs[self.lifted_obj_idx].y) = (
                self.agent_pos
            )

        done = self._goal_fn()
        reward = 0.0
        if done:
            reward = 1.0

        self.timer += 1
        trunc = self.timer == MAX_TIME

        return self._gen_obs(), reward, done, trunc, self._gen_info()

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        random.seed(seed)
        self.lifted_obj_idx = None
        self.agent_pos = (
            random.randrange(0, GRID_SIZE),
            random.randrange(0, GRID_SIZE),
        )
        used_pos = {self.agent_pos}
        self.timer = 0
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
        obs = np.zeros([11], dtype=np.float32)
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
        mask[5] = (
            self._top_obj_idx(self.agent_pos) is None or self.lifted_obj_idx is not None
        )
        mask[6] = self.lifted_obj_idx is None
        return {"action_mask": mask}

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

    def _adj_to_obj(self, obj_idx: int, pos: Tuple[int, int]) -> bool:
        """
        Returns True if the position is immediately adjacent to the object.
        """
        obj = self.objs[obj_idx]
        dist = abs(obj.x - pos[0]) + abs(obj.y - pos[1])
        return dist == 1

    def _goal_fn(self) -> bool:
        """
        Returns True when the goal has been reached.
        """
        raise NotImplementedError("This method must be overrided.")

    def render(self):
        if self.render_mode == "human":
            bg_color = (122, 122, 122)
            self.screen.fill(bg_color)
            cell_size = WINDOW_SIZE / GRID_SIZE
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    cell_color = bg_color
                    if self.agent_pos == (x, y):
                        cell_color = (0, 0, 0)
                    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
                        if (
                            i == self._top_obj_idx((x, y))
                            and self.objs[i].obj_above is None
                        ):
                            cell_color = color
                    pygame.draw.rect(
                        self.screen,
                        cell_color,
                        (
                            x * cell_size,
                            y * cell_size,
                            (x + 1) * cell_size,
                            (y + 1) * cell_size,
                        ),
                    )
            pygame.display.flip()
            self.clock.tick(10)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None


class LiftXEnv(BaseGridrobomanEnv):
    """
    The agent must lift the correct object.
    """

    def _goal_fn(self) -> bool:
        return self.lifted_obj_idx == self.x_obj


class TouchXEnv(BaseGridrobomanEnv):
    """
    The agent must be directly adjecent to the correct object.
    """

    def _goal_fn(self) -> bool:
        return self.lifted_obj_idx is None and self._adj_to_obj(
            self.x_obj, self.agent_pos
        )


class MoveXToCenterEnv(BaseGridrobomanEnv):
    """
    The agent must place the correct object in the center (3x3 area in the center of the grid).
    """

    def _goal_fn(self) -> bool:
        obj = self.objs[self.x_obj]
        return self.lifted_obj_idx is None and (obj.x, obj.y) in CENTER


class MoveXToCornerEnv(BaseGridrobomanEnv):
    """
    The agent must place the correct object in any of the corners (any of the 4 2x2 in the corners).
    """

    def _goal_fn(self) -> bool:
        obj = self.objs[self.x_obj]
        return self.lifted_obj_idx is None and (obj.x, obj.y) in CORNERS


class TouchXWithYEnv(BaseGridrobomanEnv):
    """
    The agent must be directly adjacent to object X while holding object Y.
    """

    def _goal_fn(self) -> bool:
        assert self.y_obj in [0, 1, 2], "Y object is not in [0, 1, 2]"
        return self.lifted_obj_idx is self.y_obj and self._adj_to_obj(
            self.x_obj, self.agent_pos
        )


class MoveXCloseToYEnv(BaseGridrobomanEnv):
    """
    The agent must place objects X and Y next to each other, such that the distance between both objects in the X and Y
    directions do not exceed 1.
    """

    def _goal_fn(self) -> bool:
        assert self.y_obj in [0, 1, 2], "Y object is not in [0, 1, 2]"
        x_obj = self.objs[self.x_obj]
        y_obj = self.objs[self.y_obj]
        x_dist = abs(x_obj.x - y_obj.x)
        y_dist = abs(x_obj.y - y_obj.y)
        return self.lifted_obj_idx is None and x_dist <= 1 and y_dist <= 1


class MoveXFarFromYEnv(BaseGridrobomanEnv):
    """
    The agent must place objects X and Y away from each other, such that the Manhattan distance between the objects are
    greater than 9.
    """

    def _goal_fn(self) -> bool:
        assert self.y_obj in [0, 1, 2], "Y object is not in [0, 1, 2]"
        x_obj = self.objs[self.x_obj]
        y_obj = self.objs[self.y_obj]
        x_dist = abs(x_obj.x - y_obj.x)
        y_dist = abs(x_obj.y - y_obj.y)
        return self.lifted_obj_idx is None and x_dist + y_dist > 9


class StackXOnYEnv(BaseGridrobomanEnv):
    """
    The agent must place object X on top of object Y.
    """

    def _goal_fn(self) -> bool:
        assert self.y_obj in [0, 1, 2], "Y object is not in [0, 1, 2]"
        return self.objs[self.x_obj].obj_below == self.y_obj
