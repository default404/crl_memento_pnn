###################################################
#
# The general PPO Agent for use of custom policies.
# The Agent uses a custom ray.TorchPolicy class
# from the policy module.
#
###################################################
import logging

# from ray.rllib.agents.trainer import with_base_config
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo import validate_config as ppo_validate_config
# from ray.rllib.agents.trainer import Trainer
# from ray.rllib.agents.trainer_template import build_trainer

# from ray.rllib.policy.torch_policy import TorchPolicy
# from ray.rllib.policy.torch_policy_template import build_torch_policy

from common.util import dict_deep_update
from continual_atari.agents.continual_trainer_template import build_continual_trainer


logger = logging.getLogger(__name__)

DEFAULT_CONTINUAL_CONFIG = dict_deep_update(DEFAULT_CONFIG, 
    {
        'algorithm': None,  #default config will raise for now until the first working algo is running 
        'env_config': {
            'task_list': [],
            'per_env_config': {},
            'add_monitor': True,
            'init_on_creation': True,
            'reset_on_switch': True,
            'allow_revisit': False
        },
        'framework': 'torch'     #algos are only implemented in PyTorch
    }
)
#NOTE: (rllib/evaluation/worker_set.py#L188)-(RolloutWorker creation)
# to the env creator, the config["env_config"] subdict is passed as env_config
# to the policy, the whole config is passed as policy_config 
# to the model, the config["model"] subdict is passed as model_config

def get_policy_class(config):
    """Selects the policy to attach to the Agent (Trainer).
    The config must have the following entry:
        "algorithm": [pnn|ewc|progress_compress]
    """
    from continual_atari.agents.policies import (PPOPolicy_PNN, PPOPolicy_EWC, 
        PPOPolicy_ProgComp)
    if not config.get("framework") != 'torch':
        raise ValueError("The continual learning policies are currently only "
            "implemented in PyTorch! Please set `framework` in the config to "
            "'torch' or remove it to use its default value.")
    if config.get("algorithm") == "pnn":
        return PPOPolicy_PNN
    elif config.get("algorithm") == "ewc":
        return PPOPolicy_EWC
    elif config.get("algorithm") == "progress_compress":
        return PPOPolicy_ProgComp
    else:
        raise ValueError("Unknown continual learning algorithm selected "
            "[{}]".format(config.get("algorithm")))


def collect_metrics_fn(trainer, selected_workers=None):
    """Collects metrics from the remote workers of this agent.
        This is the same data as returned by a call to train().
    """
    orig_metrics = trainer.optimizer.collect_metrics(
        trainer.config["collect_metrics_timeout"],
        min_history=trainer.config["metrics_smoothing_episodes"],
        selected_workers=selected_workers)

    #TODO: collect the continual env metrics
    return orig_metrics


def validate_config(config):
    """Extends the original PPO config validation"""
    ppo_validate_config(config)

    if not len(config["env_config"].get("task_list", [])) > 0:
        raise ValueError("The task list in the env_config must have "
            "at least one task for continual learning!")
    
    if len(config["env_config"].get('per_env_config',{}).keys()) != \
        len(config["env_config"].get("task_list", [])):
        raise ValueError("The configuration per environment in the env_config "
            "must have the same number of tasks as the task list!")

    #FIXME: the `allow_revisit` flag is implemented for the continual env but adds
    # huge complexity to the model! Thus for now this will be disabled here!
    if config["env_config"].get('allow_revisit'):
        logging.warning("'allow_revisit' was set to allow the agent to retrain "
        "on tasks already visited. This is not implemented in the continual "
        "policy for now and will be disabled!")
        config["env_config"].update(allow_revisit = False)

#--------------------------------------------------


"""
@rllib/agents/trainer_template.py
Available parameters to override from definition:
def build_trainer(name,
                  default_policy,
                  default_config=None,
                  validate_config=None,
                  get_initial_state=None,
                  get_policy_class=None,
                  before_init=None,
                  make_workers=None,
                  make_policy_optimizer=None,
                  after_init=None,
                  before_train_step=None,
                  after_optimizer_step=None,
                  after_train_result=None,
                  collect_metrics_fn=None,
                  before_evaluate_fn=None,
                  mixins=None,
                  execution_plan=None)
"""

'''
@rllib/agents/ppo/ppo.py
ray.Trainer cls signature:
  def __init__(self, config=None, env=None, logger_creator=None):
'''

# ContinualPPOTrainer = build_continual_trainer("Continual-PPO", 
#                                             PPOTrainer.with_updates(
#                                                 name="PPO",
#                                                 default_config=DEFAULT_CONFIG,
#                                                 validate_config=validate_config,
#                                                 get_policy_class=get_policy_class,
#                                                 collect_metrics_fn=collect_metrics_fn),
#                                             default_config=DEFAULT_CONTINUAL_CONFIG)