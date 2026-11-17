from gymnasium.envs.registration import register

from marcas.env import MarkedMaze

register(id="MarkedMaze-v0", entry_point="marcas.env:MarkedMaze")

__all__ = ["MarkedMaze"]
