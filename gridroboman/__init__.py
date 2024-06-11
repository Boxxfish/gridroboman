from gymnasium import register

for x in ["Red", "Green", "Blue"]:
    register(
        id=f"Gridroboman-Lift{x}-v0",
        entry_point="gridroboman.envs:LiftXEnv",
        kwargs={"x_obj": x.lower()},
    )

    register(
        id=f"Gridroboman-Touch{x}-v0",
        entry_point="gridroboman.envs:TouchXEnv",
        kwargs={"x_obj": x.lower()},
    )

    register(
        id=f"Gridroboman-Move{x}ToCenter-v0",
        entry_point="gridroboman.envs:MoveXToCenterEnv",
        kwargs={"x_obj": x.lower()},
    )

    register(
        id=f"Gridroboman-Move{x}ToCorner-v0",
        entry_point="gridroboman.envs:MoveXToCornerEnv",
        kwargs={"x_obj": x.lower()},
    )

    for y in ["Red", "Green", "Blue"]:
        register(
            id=f"Gridroboman-Touch{x}With{y}-v0",
            entry_point="gridroboman.envs:TouchXWithYEnv",
            kwargs={"x_obj": x.lower(), "y_obj": y.lower()},
        )

        register(
            id=f"Gridroboman-Move{x}CloseTo{y}-v0",
            entry_point="gridroboman.envs:MoveXCloseToYEnv",
            kwargs={"x_obj": x.lower(), "y_obj": y.lower()},
        )

        register(
            id=f"Gridroboman-Move{x}FarFrom{y}-v0",
            entry_point="gridroboman.envs:MoveXFarFromYEnv",
            kwargs={"x_obj": x.lower(), "y_obj": y.lower()},
        )

        register(
            id=f"Gridroboman-Stack{x}On{y}-v0",
            entry_point="gridroboman.envs:StackXOnYEnv",
            kwargs={"x_obj": x.lower(), "y_obj": y.lower()},
        )
