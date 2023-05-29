import logging

from ray.rllib.agents.trainer import Trainer, COMMON_CONFIG
from ray.rllib.utils.annotations import override, DeveloperAPI

from envs.continual_env import ContinualEnv

logger = logging.getLogger(__name__)


@DeveloperAPI
def build_continual_trainer(name,
                            base_trainer_cls,
                            default_config=None):
    """Helper function for defining a custom continual trainer.
    Arguments:
        name (str): name of the trainer (e.g. "PPO")
        base_trainer_cls (cls): the base Trainer class to extend for
            continual training. This class should have the ray.Trainer
            as a base class!
        default_config (dict): The default config dict of the algorithm,
            otherwise uses the Trainer default config.
            Besides the config items needed and used in the Trainer
            base class, also the following fields are required:
                'env_config':'task_list': list
                'env_config':'per_env_config': dict
    
    Returns:
        a Trainer instance that uses the specified args.
    """

    original_kwargs = locals().copy()

    if not issubclass(base_trainer_cls, Trainer):
        raise TypeError('The given trainer base class is no child of rllib\'s Trainer!')

    class trainer_cls(base_trainer_cls):
        _name = name
        _default_config = default_config or base_trainer_cls._default_config

        def __init__(self, config=None, env=None, logger_creator=None):
            # new config keys can only be added if this is True
            trainer_cls._allow_unknown_configs = True
            
            #this sets:
            # self.config : this config dict
            # self.env_creator : given env creator (eg gym.make), maybe wrapped with NormalizeAction
            # self.logdir : the log dir for this trainer
            # self.iteration : number of times train() was called
            super(trainer_cls,self).__init__(config, env, logger_creator)
            

        @override(base_trainer_cls)
        def _init(self, config, env_creator):

            #this will built the workers, create the envs, etc
            #this sets:
            # self.workers : WorkerSet()
            # self.workers.local_worker() : RolloutWorker
            # self.workers.remote_workers() : [RolloutWorker]
            # self.workers.*_worker().async_env : gymEnv -> VectorEnv -> _VectorEnvToBaseEnv(BaseEnv)
            # self.optimizer : (Sync)SampleOptimizer
            super(trainer_cls,self)._init(config, env_creator)

            self.task_list = []     #the central task store
            self._curr_task_id = None
            self._curr_task_name = None

            task_list = config["env_config"].get("task_list", [])
            per_env_config = config["env_config"].get('per_env_config',{})
            assert task_list and per_env_config, "Task list and config must at least have one entry!"
            for task in task_list:
                config = per_env_config.get(task, {})
                self.add_task(task, config)      #adds the tasks as task entries to the tasklist

            #whether to reset the next env automatically on task switch
            self._reset_on_switch = config["env_config"].get('reset_on_switch')
            
            #whether to allow to revisit tasks in the task list that were already done
            self._allow_revisit = config["env_config"].get('allow_revisit')


            # Validate that the agent received an continual learning env!
            # Because the super _init is called before, the rollout workers are already created
            # and one can access the worker fields
            if not isinstance(self.workers.local_worker().env, ContinualEnv):
                raise ValueError(
                    "The given environment is not a continual learning environment! "
                    "Please make sure to inherit from 'ContinualEnv' if using a custom env.")

            #IMPORTANT: set the pre-processor disabler for all the workers because the continual
            # env has its own preproc setup (must be done for each task in the env creator!)
            #Problem: the RolloutWorker's always create Preprocessors (@rllib.models.preprocessors.py) 
            # (even for deepmind a NoPreprocessor obj) which are FIXED at worker creation! Since the 
            # preproc behaviour has to change for each new env, a dynamic adaption of the preprocs
            # (needed for a specific task env) can not be done here (a new Sampler would be needed each 
            # time the task is switched). Instead always use the NoPreprocessor and the correct 
            # conversion of the env space must be handled by the wrappers for the continual env 
            # (e.g. deepmind wrapper does it according to Ray).
            # Note: the RolloutWorker should throw an error if the obs-space is not suitable
            # Further readings: 
            #  RolloutWorker (rllib.models.preprocessors.py): the class for all the rollout work
            #       env wrapping: l303; preproc creation: l825; samplers (use the preprocs): l449
            #  (A)SyncSampler (rllib.evaluation.sampler.py): actual env execution, obs processing
            #       env sampling (using a VectorEnv conversion of the gym.Env): l233, obs proc: l377
            def dis_preproc(worker):
                worker.preprocessing_enabled = False
            self.workers.foreach_worker(dis_preproc)

            #set the first task immediately if specified in the settings
            if config["env_config"].get("init_on_creation") and \
                self.workers.local_worker().env._curr_task_id is None:
                self.next_task()

        #TODO: This whole precedure of switching / exchanging fields of the trainer and 
        # ensuring that the internal fields are updated when the env is switched COULD be 
        # easier by exchanging the whole Workers with new Workers with needed base env creator
        # for this task and using the current trainer policy
        def next_task(self):
            """Set the agent to train on the next task by incrementing
            the task id counter.
            This function is designed as a propagator to underlying functions
            of the policy and environment with the same name.
            
            Return:
                _ : (bool) whether the next task could be set. 
                    False indicates the end of the task list.

            Note:
                Once the task list is exhaused and this function returns False,
                the Trainer keeps track of the last processed task ID. A new task 
                can be added still and then this function can be called to
                pick up the new tasks.
            """
            curr_task_id = self._curr_task_id   #the old task id
            if curr_task_id is None:
                curr_task_id = 0
            elif curr_task_id < (len(self.task_list) - 1):
                #there is a next task in the tasklist
                curr_task_id += 1
            else:
                #if we are at done with the last task in the task list
                #TODO: do some metric wrap-up here
                self._curr_task_name = None

                return False

            self._curr_task_id = curr_task_id
            self._curr_task_name = self.task_list[curr_task_id]['task_name']

            # Set the envs to the next task
            #Note: foreach_env is called on the BaseEnv.get_unwrapped() which yields the
            # env created by the original env (creator) given 
            try:
                # this returns the initial obs if 'reset_on_switch' is set (default)
                #TODO: check if the env is reset properly after a task switch
                self.workers.foreach_worker(
                    lambda ev: ev.foreach_env(
                        lambda env: env.next_task()))
            except AttributeError as ae:
                logger.error("The environment has no next_task() function! "
                    "Make sure to use a suitable environment for continual learning.")
                raise AttributeError from ae
            
            # Set the policy to the next task
            #Note: with this abstract solution, each underlying policy must implement the 
            # next_task() functor
            obs_space = self.workers.local_worker().env.observation_space
            action_space = self.workers.local_worker().env.action_space
            try:
                self.workers.foreach_policy(
                    lambda p, pid: p.next_task(obs_space, action_space))
            except AttributeError as ae:
                logger.error("The policy has no next_task() function! "
                    "Make sure to use a suitable policy for continual learning.")
                raise AttributeError from ae

            return True
        

        def set_task(self, task_id):
            """Sets a specific task from the task list. This switches the Trainer
            to the task with the specific task ID given. 
            Return:
                env and policy info: (tuple) the return values of the environment
                    (ret[0]) and policy (ret[1]) functions when the task is switched.
            Notes:
                - Does *not* reset the env by default.
                - `allow_revisit` allows to revisit already trained envs. This 
                    will override all metrics for this and ALL envs after the 
                    revisited one in the tasklist! Be sure to collect them 
                    before if you need them! 
                    If task_id == _curr_task_id, the current 
                    task will be stopped, its config cloned and added as a new
                    task in the tasklist.
            """
            if task_id not in range(len(self.task_list)):
                raise ValueError("Given task ID is not in the range of available tasks!")
            
            #end the current task if any
            if self._curr_task_id is not None:
                self.task_list[self._curr_task_id]['task_stats'].update(finished=True)
                # self.collect_task_stats()    #TODO:implement this function to collect the additional infos from each task
            
            # if the next task id is the current one
            if task_id == self._curr_task_id:
                if self._allow_revisit:
                    #clone the task at task_id and set it to the next task
                    task_id += 1
                    self.add_task(
                        self.get_task_property(self._curr_task_id, "_task"), 
                        self.get_task_property(self._curr_task_id, "env_config"),
                        task_id=task_id)
                else:
                    raise ValueError("The next task ID cannot be the current task ID! "
                        "Set 'allow_revisit' if this is intended.")
            
            # if the task to set was already finished
            if self.task_list[task_id]['task_stats'].get("finished", False):
                if self._allow_revisit:
                    #reset the stats for this task
                    self._reset_task(task_id)
                else:
                    raise ValueError("Trying to set an already visited task! "
                        "Set 'allow_revisit' if this is intended.")

            self._curr_task_id = task_id
            self._curr_task_name = self.task_list[curr_task_id]['task_name']

            # Set the envs to the next task
            #Note: foreach_env is called on the BaseEnv.get_unwrapped() which yields the
            # env created by the original env (creator) given (NOT the gym env.unwrapped)
            try:
                env_ret = self.workers.foreach_worker(
                    lambda ev: ev.foreach_env(
                        lambda env: env.set_task(task_id,
                            reset_on_switch=self._reset_on_switch, 
                            allow_revisit=self._allow_revisit)))
            except AttributeError as ae:
                logger.error("The environment has no set_task() function! "
                    "Make sure to use a suitable environment for continual learning.")
                raise AttributeError from ae

            # Set the policy to the next task
            #Note: with this abstract solution, each underlying policy must implement the 
            # next_task() functor
            raise NotImplementedError("set_task() function for the policy is not implemented!")
            try:
                pol_ret = self.workers.foreach_policy(
                    lambda p, pid: p.set_task())
            except AttributeError as ae:
                logger.error("The policy has no set_task() function! "
                    "Make sure to use a suitable policy for continual learning.")
                raise AttributeError from ae

            return (env_ret, pol_ret)


        def add_task(self, task_name, env_config, task_id=None):
            """Adds a new task to the tasklist, extending the
            initial task list.
                env_id_or_creator: can either be a vaild gym env name 
                    or a callable that returns an env object.
                env_config: the config needed to set up this env.
                    The env creator is called with this config.
                task_id: the position in the tasklist where the
                    new task shall be inserted. If not set, the task
                    is put at the end of the tasklist.
            Note: 
                If task_pos < _curr_task_id, the added task can 
                only be visited by manually calling set_task(task_pos)
            """
            if task_id is None or task_id > len(self.task_list):
                task_id = len(self.task_list)
            
            task_entry = self._make_task_list_entry(task_name, env_config)
            self.task_list.insert(task_id, task_entry)


        @override(Trainer)
        def _train(self):
            #each train() step runs:
            #  1. collect [train_batch_size] samples from remote workers (is not always excactly that number)
            #  2. for num_sgd_iter times do:
            #    4. random shuffle all samples (if no RNN state)
            #    3. for [train_batch_size/sgd_minibatch_size] num of mini-batches do: 
            #       5. run SGD on [sgd_minibatch_size] samples
            # => performs [num_sgd_iter * train_batch_size / sgd_minibatch_size] number of SGD steps
            #    on ~[num_sgd_iter * train_batch_size] num of samples (=steps in environment)
            res = super()._train()

            #call next_task after sufficient enough train steps
            #TODO: use metric or hard timestep value?
            
        

        # @override(Trainer)
        # def _before_evaluate(self):
        #     if before_evaluate_fn:
        #         before_evaluate_fn(self)

        #TODO: this is for saving and loading of the model, maybe it can be kept 'as-is'
        # def __getstate__(self):
        #     state = Trainer.__getstate__(self)
        #     state["trainer_state"] = self.state.copy()
        #     if self.train_exec_impl:
        #         state["train_exec_impl"] = (
        #             self.train_exec_impl.shared_metrics.get().save())
        #     return state

        # def __setstate__(self, state):
        #     Trainer.__setstate__(self, state)
        #     self.state = state["trainer_state"].copy()
        #     if self.train_exec_impl:
        #         self.train_exec_impl.shared_metrics.get().restore(
        #             state["train_exec_impl"])


        @staticmethod
        def _make_task_list_entry(task_name, env_config):
            """Create a new task list entry"""
            task_entry = {
                    "task_name": task_name,
                    "env_config": env_config,
                    "task_stats": {}
                }
            return task_entry


    def with_updates(**overrides):
        """Build a copy of this trainer with the specified overrides.
        Arguments:
            overrides (dict): use this to override any of the arguments
                originally passed to build_trainer() for this policy.
        """
        return build_continual_trainer(**dict(original_kwargs, **overrides))

    trainer_cls.with_updates = staticmethod(with_updates)
    trainer_cls.__name__ = name
    trainer_cls.__qualname__ = name
    return trainer_cls