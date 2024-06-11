import pytest
import gymnasium as gym

from gridroboman.envs import *


@pytest.mark.parametrize(
    "x_obj, y_obj",
    [
        ("red", "green"),
        ("red", "blue"),
        ("green", "red"),
        ("green", "blue"),
        ("blue", "red"),
        ("blue", "green"),
    ],
)
@pytest.mark.parametrize(
    "env_class",
    [
        LiftXEnv,
        TouchXEnv,
        MoveXToCenterEnv,
        MoveXToCornerEnv,
        TouchXWithYEnv,
        MoveXCloseToYEnv,
        MoveXFarFromYEnv,
        StackXOnYEnv,
    ],
)
class TestAllEnvs:
    @pytest.fixture
    def env(self, env_class, x_obj: str, y_obj: str):
        env = env_class(x_obj=x_obj, y_obj=y_obj)
        env.reset()
        return env

    @pytest.mark.parametrize(
        "action, dx, dy", [(1, 0, -1), (2, 0, 1), (3, -1, 0), (4, 1, 0)]
    )
    def test_move(self, action: int, env: BaseGridrobomanEnv, dx: int, dy: int):
        env.agent_pos = (3, 3)
        env.step(action)
        assert env.agent_pos == (3 + dx, 3 + dy)

    @pytest.mark.parametrize(
        "action, dim, wall",
        [
            (1, 1, 0),
            (2, 1, 6),
            (3, 0, 0),
            (4, 0, 6),
        ],
    )
    def test_into_wall(self, action: int, dim: int, wall: int, env: BaseGridrobomanEnv):
        while env.agent_pos[dim] != wall:
            env.step(action)
        prev_pos = env.agent_pos
        env.step(action)
        assert env.agent_pos == prev_pos
