"""
Allows a human player to "play" an environment.

Controls
--------
Movement:   Arrow keys
Lift:       L
Put:        P
Quit:       Esc
"""

from argparse import ArgumentParser

import gymnasium as gym
import pygame


def main():
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="Gridroboman-LiftRed-v0")
    args = parser.parse_args()

    env = gym.make(args.name, render_mode="human")
    env.reset()
    env.render()
    while True:
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        action = 0
        if keys[pygame.K_UP]:
            action = 1
        elif keys[pygame.K_DOWN]:
            action = 2
        elif keys[pygame.K_LEFT]:
            action = 3
        elif keys[pygame.K_RIGHT]:
            action = 4
        elif keys[pygame.K_l]:
            action = 5
        elif keys[pygame.K_p]:
            action = 6
        elif keys[pygame.K_ESCAPE]:
            break
        else:
            env.render()
            continue

        obs_, rew_, done, trunc, info_ = env.step(action)

        if done or trunc:
            env.reset()

        env.render()
    env.close()


if __name__ == "__main__":
    main()
