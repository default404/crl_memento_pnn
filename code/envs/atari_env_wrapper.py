###################################################
# 
# Environment wrapper functionality for the Gym
# Atari games.
#
###################################################
import os, pickle
import numpy as np
import gym, ray

from common.util import setup_logger

_logger = setup_logger(name=__name__,verbose=True)

class MementoEnv(gym.Wrapper):
    """This wrapper initializes the env to a given 
    emulator state for the ALE at each reset.
    The restore_buffer should be a list-like with entries:
        {'return': ..., 'state': ...}
    For reference see On Catastrophic Interference in Atari 2600 Games
    (https://arxiv.org/abs/2002.12499)
    """
    def __init__(self, env, restore_buffer_path=None):
        super(MementoEnv, self).__init__(env)
        if restore_buffer_path is not None and os.path.isfile(restore_buffer_path):
            with open(restore_buffer_path, 'rb') as f:
                self.restore_buffer = pickle.load(f)
        else:
            self.restore_buffer = None
        self._add_reward_offset = False 
        self.reward_offset = 0
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if self.restore_buffer is not None:
            #using uniform sample strategy
            s = np.random.randint(len(self.restore_buffer))
            to_restore = self.restore_buffer[s]
            self.reward_offset = to_restore.get('return')
            self.restore_state(to_restore.get('state'))
            self._add_reward_offset = True
        #NOTE: that on reset still the raw initial obs will be returned 
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self._add_reward_offset:     #only add the base reward once per episode
            reward += self.reward_offset
            self._add_reward_offset = False
        return obs, reward, done, info


def wrap_deepmind(env, dim=84, framestack=True, episodic_life=True):
    """Configure environment for DeepMind-style Atari as from Rllib
    but also with the option to disable the EpisodicLife wrapper
    for full env evaluation.

    Args:
        dim (int): Dimension to resize observations to (dim x dim).
        framestack (bool): Whether to framestack observations.
        episodic_life (bool): Whether to add episodic life hehavior.
    """
    from ray.rllib.env.atari_wrappers import (MonitorEnv, NoopResetEnv,
                                              WarpFrame)
    env = MonitorEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    if "NoFrameskip" in env.spec.id:
        from ray.rllib.env.atari_wrappers import MaxAndSkipEnv
        env = MaxAndSkipEnv(env, skip=4)
    if episodic_life:
        from ray.rllib.env.atari_wrappers import EpisodicLifeEnv
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        from ray.rllib.env.atari_wrappers import FireResetEnv
        env = FireResetEnv(env)
    env = WarpFrame(env, dim)
    # env = ScaledFloatFrame(env)  # TODO: use for dqn?
    # env = ClipRewardEnv(env)  # reward clipping is handled by policy eval
    if framestack:
        from ray.rllib.env.atari_wrappers import FrameStack
        env = FrameStack(env, 4)
    return env


def wrap_atari(env_creator, env_config, logger=_logger):
    ''' Creates the gym.atari environment but also adds some wrappers 
    around it depending on the parameters in env_config.
    This module uses the RLlib wrappers available.
    The `env_style` can be 'deepmind' or 'rllib'
    '''
    env_style = env_config['env_style']

    if env_style == "deepmind":
        def build_atari(config):
            env = wrap_deepmind(env_creator(config), 
                                config.get('image_dim',84), 
                                config.get('dm_enable_frameSkip',True),
                                config.get('dm_enable_episodicLife',True))
            return env

    elif env_style == "rllib":
        def build_atari(config):
            env = env_creator(config)

            assert 'NoFrameskip' in env.spec.id     #FIXME: maybe not necessary
            #add the noop init wrapper to randomize the start point
            if config.get('enable_noopInit', False):
                from ray.rllib.env.atari_wrappers import NoopResetEnv
                env = NoopResetEnv(env, noop_max=config.get('noop_start_max',0))
            # add the frame skipper wrapper to repeat actions at least `skip_frames` times
            if config.get('enable_frameSkip', False):
                from ray.rllib.env.atari_wrappers import MaxAndSkipEnv
                env = MaxAndSkipEnv(env, skip=config.get('skip_frames',4))
            
            return env

    else:
        if env_style is not None:
            logger.warning('Unknown env_style [{}]! The raw Atari env for the given ID will be used'.format(env_style))
        build_atari = env_creator

    return build_atari
