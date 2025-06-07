import gymnasium as gym

from .ran_slice import RanSlice

gym.register(
    id='RanSlice-v1',
    entry_point='gym_ran_slice:RanSlice'
)