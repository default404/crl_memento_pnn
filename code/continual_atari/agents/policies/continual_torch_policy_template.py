
import os, errno
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()

@DeveloperAPI
def build_continual_torch_policy(name,
                                 *,
                                 base_policy_cls,
                                 get_default_config=None,
                                 next_task_loss_fn=None,
                                 next_task_model_fn=None):
    """Helper function for creating a torch policy class for continual learning
    at runtime.
    Arguments:
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
    Returns:
        type: TorchPolicy child class constructed from the specified args.
    Note:
        If neither `next_task_loss_fn` nor `next_task_model_fn` is defined, no action
        is performed at all on task switch.
    """

    original_kwargs = locals().copy()
    if not issubclass(base_policy_cls, TorchPolicy):
        raise TypeError('The given policy base class is no child of rllib\'s TorchPolicy!')

    class policy_cls(base_policy_cls):
        def __init__(self, obs_space, action_space, config):
            #this sets:
            # self.config : this config dict
            # self.model : the model that learns the policy
            # sel.dist_class : class of the action distribution
            super(policy_cls,self).__init__(obs_space, action_space, config)

            #TODO: this is dangerous since these are 'unknown' top level config keys!
            #policy will fail in trainer that doesnt `_allow_unknown_configs` 
            self.algo_type = config['algorithm']    
            self.algo_config = config['algo_config']
            
        
        #TODO: This whole precedure of switching / exchanging fields of the trainer and 
        # ensuring that the internal fields are updated when the env is switched COULD be 
        # easier by exchanging the whole Workers with new Workers with needed base env creator
        # for this task and using the current trainer policy
        def next_task(self, obs_space, action_space):
            """Tell the policy to train on the next task.
            This function takes the neccessary actions to switch / add a task
            to the underlying loss and model.
            """
            loss_ret = None
            if next_task_loss_fn is not None:
                try:
                    loss_ret = next_task_loss_fn(self, self.config)
                except Exception as e:
                    raise Exception("Error occurred while switching tasks: "
                        "Error in the `next_task_loss_fn` function:\n{}".format(e))

            model_ret = None
            if next_task_model_fn is not None:
                try:
                    model_ret = next_task_model_fn(self, obs_space, action_space, self.config)
                except Exception as e:
                    raise Exception("Error occurred while switching tasks: "
                        "Error in the `next_task_model_fn` function:\n{}".format(e))
            
            #TODO: update all fields that are dependent on the env
            self.observation_space = obs_space
            self.action_space = action_space
            
            return (loss_ret, model_ret)


        @override(TorchPolicy)
        def export_model(self, export_dir):
            """
            Example:
                >>> trainer = MyTrainer()
                >>> for _ in range(10):
                >>>     trainer.train()
                >>> trainer.export_policy_model("/tmp/")
            """
            try:
                os.makedirs(export_dir)
            except OSError as e:
                # ignore error if export dir already exists
                if e.errno != errno.EEXIST:
                    raise
            self.model.export_model(export_dir)


        @override(TorchPolicy)
        def export_checkpoint(self, export_dir, filename):
            """
            Example:
                >>> trainer = MyTrainer()
                >>> for _ in range(10):
                >>>     trainer.train()
                >>> trainer.export_policy_checkpoint("/tmp","weights.h5")
                
            This creates a torch save point with fields:
                "opt_n_state_dict": optimizer state for `n` optimizers
                "model_state_dict": state dict of the policy model
            """
            try:
                os.makedirs(export_dir)
            except OSError as e:
                # ignore error if export dir already exists
                if e.errno != errno.EEXIST:
                    raise
            export_file = os.path.join(export_dir, filename)
            save_dict = {}
            for i, opt in self._optimizers:
                save_dict["opt_{}_state_dict".format(i)] = opt.state_dict()
            save_dict["model_state_dict"] = self.model.state_dict()
            
            torch.save(save_dict, export_file)


        def load_checkpoint(self, import_file, strict=True):
            """Load an entire checkpoint file saved with `export_checkpoint`. 
            This restores not only the model weights, but also the optimizer.
            Example:
                >>> trainer = MyTrainer()
                >>> trainer.workers.local_worker().for_policy(
                >>>     lambda p: p.load_checkpoint("/tmp/policy_ckpt.h5"))
            This needs a torch save point with fields:
                "opt_n_state_dict": optimizer state for n optimizers
                "model_state_dict": state dict of the policy model
            """
            self.model.import_from_h5(import_file)
            checkpoint = torch.load(import_file)
            for i, opt in self._optimizers:
                opt.load_state_dict(checkpoint["opt_{}_state_dict".format(i)], strict=strict)


    def with_updates(**overrides):
        return build_continual_torch_policy(**dict(original_kwargs, **overrides))

    policy_cls.with_updates = staticmethod(with_updates)
    policy_cls.__name__ = name
    policy_cls.__qualname__ = name
    return policy_cls
