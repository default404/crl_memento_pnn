

from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy

from continual_atari.agents.policies.continual_torch_policy_template import build_continual_torch_policy
from continual_atari.agents.ppo_agent import DEFAULT_CONTINUAL_CONFIG
from continual_atari.agents.policies.pnn import pnn_model


"""Args that can be passed to the build_torch_policy:
    name (str): name of the policy (e.g., "PPOTorchPolicy")
    loss_fn (func): function that returns a loss tensor as arguments
        (policy, model, dist_class, train_batch)
    get_default_config (func): optional function that returns the default
        config to merge with any overrides
    stats_fn (func): optional function that returns a dict of
        values given the policy and batch input tensors
    postprocess_fn (func): optional experience postprocessing function
        that takes the same args as Policy.postprocess_trajectory()
    extra_action_out_fn (func): optional function that returns
        a dict of extra values to include in experiences
    extra_grad_process_fn (func): optional function that is called after
        gradients are computed and returns processing info
    optimizer_fn (func): optional function that returns a torch optimizer
        given the policy and config
    before_init (func): optional function to run at the beginning of
        policy init that takes the same arguments as the policy constructor
    after_init (func): optional function to run at the end of policy init
        that takes the same arguments as the policy constructor
    make_model_and_action_dist (func): optional func that takes the same
        arguments as policy init and returns a tuple of model instance and
        torch action distribution class. If not specified, the default
        model and action dist from the catalog will be used
    mixins (list): list of any class mixins for the returned policy class.
        These mixins will be applied in order and will have higher
        precedence than the TorchPolicy class
"""


def make_pnn_model(policy, obs_space, action_space, config):
    """ Gets the whole config to work with"""

    model = pnn_model.PNN(obs_space=obs_space, 
               action_space=action_space, 
               num_outputs=None, 
               model_config=config['model'], 
               name='PNN')
    return model

#TODO: one can override make_model or make_model_and_action_dist if a custom distribution
# is needed for the custom policy. Only one must be provided!
__PPOPolicy_PNN = PPOTorchPolicy.with_updates(
    name='_PPOPolicy_PNN',
    get_default_config=lambda: DEFAULT_CONTINUAL_CONFIG,
    make_model=make_pnn_model
    )


"""Args that can be passed to the `build_continual_torch_policy`:
    name (str): name of the policy (e.g., "PPOTorchPolicy")
    base_policy_cls (class): a base policy class to derive most functionality
        from. If given, most of the other settings are ignored.
    get_default_config (Optional[callable]): Optional callable that returns
        the default config to merge with any overrides.
    next_task_loss_fn (Optional[callable]): Optional callable that is executed
        on the loss object when the task is switched. Gets the policy instance
        and the config as parameter.
    next_task_model_fn (Optional[callable]): Optional callable that is executed
        on the model object when the task is switched. Gets the policy instance
        and the config as parameter.
"""

# Not needed for PNN
# def next_task_loss_fn(policy, config):
#     policy.model.new_task()


def next_task_model_fn(policy, obs_space, action_space, config):
    policy.model.freeze_columns()
    policy.model.next_task(obs_space, action_space)

"""Policy class c'tor signature:
    PPOPolicy_PNN(obs_space, action_space, config)
"""
PPOPolicy_PNN = build_continual_torch_policy('PPOPolicy_PNN',
                                             base_policy_cls=__PPOPolicy_PNN,
                                             next_task_model_fn=next_task_model_fn)
