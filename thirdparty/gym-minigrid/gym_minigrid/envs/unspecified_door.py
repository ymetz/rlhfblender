from gym_minigrid.minigrid import (
    Door,
    Floor,
    Goal,
    Grid,
    MiniGridEnv,
    Switch,
    UnspecifiedDoor2,
)
from gym_minigrid.register import register


class UnspecifiedDoor(MiniGridEnv):
    """
    Environment with a door and switch, sparse reward
    """

    def __init__(self, size=8, **kwargs):
        super().__init__(grid_size=size, max_steps=size * 3, **kwargs)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(3, width - 2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(2, width - 2)
        self.put_obj(UnspecifiedDoor2("purple"), splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.place_obj(
            obj=Switch("purple", True), top=(splitIdx - 1, doorIdx - 1), size=(1, 1)
        )
        self.place_obj(obj=Switch("yellow", True), top=(1, 1), size=(1, 1))

        self.mission = "use the switch to open the door and then get to the goal"


class UnspecifiedDoor10x10(UnspecifiedDoor):
    def __init__(self):
        super().__init__(size=10)


register(
    id="MiniGrid-UnspecifiedDoor-10x10-v0",
    entry_point="gym_minigrid.envs:UnspecifiedDoor",
)
