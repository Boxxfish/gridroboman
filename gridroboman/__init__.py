from gymnasium import register


register(
    id="Gridroboman-LiftX-v0",
    entry_point="gridroboman.envs:LiftXEnv",
)

register(
    id="Gridroboman-TouchX-v0",
    entry_point="gridroboman.envs:TouchXEnv",
)

register(
    id="Gridroboman-MoveXToCenter-v0",
    entry_point="gridroboman.envs:MoveXToCenterEnv",
)

register(
    id="Gridroboman-MoveXToCorner-v0",
    entry_point="gridroboman.envs:MoveXToCornerEnv",
)

register(
    id="Gridroboman-TouchXWithY-v0",
    entry_point="gridroboman.envs:TouchXWithY",
)

register(
    id="Gridroboman-MoveXCloseToY-v0",
    entry_point="gridroboman.envs:MoveXCloseToYEnv",
)

register(
    id="Gridroboman-MoveXFarFromY-v0",
    entry_point="gridroboman.envs:MoveXFarFromYEnv",
)

register(
    id="Gridroboman-StackXOnY-v0",
    entry_point="gridroboman.envs:MoveXFarFromYEnv",
)