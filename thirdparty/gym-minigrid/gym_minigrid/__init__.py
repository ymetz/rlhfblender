# Import the envs module so that envs register themselves
# Import wrappers so it's accessible when installing with pip
import gym_minigrid.envs
import gym_minigrid.wrappers
from gym.envs.registration import register

# register(
#    id='MiniGrid-SwitchTest-10x10-v0',
#    entry_point='gym_minigrid.envs:SwitchEnvTest'
# )
