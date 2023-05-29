###################################################
#
# The standart implementation of PPO in Ray.
# Only minor default values are changes by default.
#
###################################################

import os
import copy

import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy


#--------------------------------------------------

PPOAgent_CONFIG = copy.deepcopy(DEFAULT_CONFIG)
PPOAgent_CONFIG.update(framework='torch')

'''
@rllib/agents/ppo/ppo.py
ray.Trainer cls signature:
  def __init__(self, config=None, env=None, logger_creator=None):
'''
PPOAgent = PPOTrainer.with_updates(
    name="vanilla_PPO",
    default_config=PPOAgent_CONFIG,     #updated config values (FIXME: necessary?)
    # default_policy=PPOTorchPolicy,      #change default policy from TF to PyTorch
    # get_policy_class=None               #if None, the default policy is always used
    )