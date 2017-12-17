# Copied from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
import os

import gym
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import WarpFrame, ScaledFloatFrame, \
                                            ClipRewardEnv, FrameStack, \
                                            EpisodicLifeEnv,  NoopResetEnv, \
                                            MaxAndSkipEnv
import numpy as np

try:
    import pybullet_envs
except ImportError:
    pass

def make_env(env_id, seed, rank, eval=False):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = WarpFrame(env)
        # Janky Fix to Resize Environments to be 50x50
        env.width = 50
        env.height = 50
        env = ScaledFloatFrame(env)
        if not eval:
            env = ClipRewardEnv(env)
            env = EpisodicLifeEnv(env)
        env = FrameStack(env, 3)
        env = TransposeOb(env)
        return env

    return _thunk

class TransposeOb(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeOb, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [3, 50, 50]
        )

    def _observation(self, observation):
        return np.array(observation).transpose(2, 0, 1)
