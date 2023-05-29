###################################################
#
# Environment wrapper functionality for the Gym
# Pong game.
#
###################################################

import gym
import numpy as np

from common.util import setup_logger
from envs.gym_wrappers import VerticalFlip, HorizontalFlip, ZoomAndRescale

_logger = setup_logger(name=__name__,verbose=True)


PONG_BASE_ENV = 'PongNoFrameskip-v4'
PONG_ENV_MAP = {
    'Pong_flipVertical': [VerticalFlip],
    'Pong_flipHorizontal': [HorizontalFlip],
    'Pong_flipBoth': [VerticalFlip, HorizontalFlip],
    'Pong_zoom': [ZoomAndRescale]
}

def make_pong_soup(env_id, config, logger=_logger):
    '''Returns an env factory for variations of Pong.
    Depending on which Pong version shall be used, the factory
    adds observation wrappers aound the base env.
    '''
    #get list of all registered gym envs
    if env_id not in PONG_ENV_MAP.keys():
        err_str = "Given env ID [{}] is not a pong soup environment!\nKnown: {}".format(env_id, PONG_ENV_MAP.keys())
        logger.error(err_str)
        raise ValueError(err_str)

    wrappers = PONG_ENV_MAP[env_id]

    def build_pong(env_config):
        env = gym.make(PONG_BASE_ENV)
        for wrapper in wrappers:
            env = wrapper(env, **env_config)

        return env

    return build_pong