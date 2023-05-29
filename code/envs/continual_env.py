
from copy import deepcopy
import gym
from gym.envs.atari import AtariEnv

from ray.rllib.env.atari_wrappers import get_wrapper_by_cls, is_atari, MonitorEnv
from envs.ray_env_util import get_env_creator


def delegateUnknown(toMember, methods):
    """The delegator decorator that allows to forward any unknown 
    method calls to a class to one of its member objects.
    """ 
    def dec(klass):
        def create_delegator(method):
            def delegator(self, *args, **kwargs):
                obj = getattr(self, toMember)
                m = getattr(obj, method)
                return m(*args, **kwargs)
            return delegator
        for m in methods:    #deligate all methods under dir() to toMember that are unknown to klass
            if m not in dir(klass):
                setattr(klass, m, create_delegator(m))
        return klass
    return dec

#NOTE: 
#   - maybe wrap this in a creator function to make the delegation class (here AtariEnv)
#       capable of parametrisation
#   - also with this solution, the continual env will only work with envs from the same type
# @delegateUnknown('_curr_env', dir(AtariEnv))
class ContinualEnv(gym.Wrapper):
    """This is the continual learning environment for Gym (Atari).
    This class inherits the gym.Env class, but since it can hold
    multiple environments for continual learning it deligates all
    unknown method calls to the `gym.envs.AtariEnv` environment, 
    which is an extension of the gym.Env.
    """
    
    #Initial list of attributes that are never forwarded to the sub-env
    #NOTE: this list is updated once at initialization with any class member 
    # defined in this class
    _DELEGATE_NEVER = ['_add_monitor', '_curr_env' '_curr_task_id', '_curr_task_name', 
        '_DELEGATION_LIST', 'task_list', 'env', 'step', 'reset']
    
    default_per_env_metrics={
        "finished": False,              #if the task has been visited; set when set_task() is called
        "total_steps": 0,               #the total steps taken in this environment (indicates task adaption rate)
        "total_episodes": 0,            #number of episodes the agent spent in this environment (env.reset ends an episode)
        "steps_per_episode": [],        #list of length total_episodes; steps taken in each of the episodes
        "mean_episode_length": 0,       #the mean length of the episodes
        "total_reward": float('-Inf'),  #the total reward the agent achieved during the training on this env 
        "rewards_per_episode": [],      #list of length total_episodes; rewards per episode
        "mean_reward": float('-Inf'),   #the mean reward the agent achieved during the training on this env 
    }

    def __init__(self, env_config):
        """Creates a continual learning env from a list of valid gym 
        env ID's. For continual learning there are two additional fields 
        in the config:
            task_list: list of valid gym env ID's OR callable env creators.
                (env creators take a config dict and return the env object)
            per_env_config: dict of config params passed to each of the 
                envs in the task list.
            other optional fields have default values if not provided.
        """
        # Add all class members not already in _DELEGATE_NEVER to the list
        self._DELEGATE_NEVER += sorted(
            set([_d for _d in self.__class__.__dict__.keys()]) - set(self._DELEGATE_NEVER)
            )
        #Initial list of attributes that are always delegated to the sub-env
        # (even if the member is private or already defined in this class)
        #NOTE: this list is continuously updated with all members that were found 
        # to be auto-delegated from the previous env to avoid member corruption
        #FIXME: depreciated, this is now handled by the super c'tor 
        self._DELEGATION_LIST = ['unwrapped', 'metadata', 'observation_space', 
                'action_space', 'reward_range', 'spec']
        
        #internals for the current task
        self._curr_task_id = None
        self._curr_task_name = None
        self._curr_env = None     #the env object of the current task
        
        self.task_list = []     #the central task store

        task_list = env_config.get("task_list", [])
        per_env_config = env_config.get('per_env_config',{})
        assert task_list and per_env_config, "Task list and per-env-config must at least have one entry!"
        for task in task_list:
            config = per_env_config.get(task, {})
            self.add_task(task, config)      #adds the tasks as task entries to the tasklist

        #whether to add the Monitor wrapper; needed for env metrics!
        self._add_monitor = env_config.get('add_monitor')
        #whether to reset the next env automatically on task switch
        self._reset_on_switch = env_config.get('reset_on_switch')
        #whether to allow to revisit tasks in the task list that were already done
        self._allow_revisit = env_config.get('allow_revisit')

        # #init the first task
        #NOTE: remove this here for now bc this is handeled by the agent already
        # if env_config.get('init_on_creation',True):
        #     self.next_task()


    @property
    def env(self):
        return self._curr_env

    @env.setter
    def env(self, env):
        self._curr_env = env


    def step(self, action):
        if self._curr_task_id is None:
            raise ValueError("The continual env has no current task! Maybe you forgot to call next_task()?")
        obs, rew, done, info = self.env.step(action)
        info.update(current_task= self._curr_task_id)
        info.update(current_task_name= self._curr_task_name)
        return (obs, rew, done, info)

    def reset(self, **kwargs):
        if self._curr_task_id is None:
            raise ValueError("The continual env has no current task! Maybe you forgot to call next_task()?")
        obs = self.env.reset(**kwargs)
        return obs

    # def render(self, mode='human'):
    #     return self.env.render(mode)

    # def close(self):
    #     return self.env.close()
    
    # def seed(self, seed=None):
    #     return self.env.seed(seed)


    def get_current_task_info(self):
        """Returns a 3-tuple of the current task info:
            (task id, task name, task environment)
        """
        return (self._curr_task_id, self._curr_task_name, self.env)


    def next_task(self):
        """Sets the env to the next task env in the task_list. By 
        incrementing the task id counter. This needs also to be 
        called once to initialize the ContinualEnv with the first 
        env. Here the next env is *not* reset after initialization.
        Return:
            _ : (bool) whether the next task could be set. 
                False indicates the end of the task list.

        Note:
            Once the task list is exhaused and the next_env flag is False,
            the env keeps track of the last processed task ID. A new task 
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
            self._curr_task_name = None
            self.close()

            return False

        #init the next env
        obs_or_none = self.set_task(curr_task_id, 
                                    reset_on_switch=self._reset_on_switch, 
                                    allow_revisit=self._allow_revisit)

        return obs_or_none


    def set_task(self, task_id, reset_on_switch=False, allow_revisit=False):
        """Sets a specific env from the task list. This creates the actual 
        environment for this task using the env creator.
        Return:
            prev_task_stats: (dict) the metric dict of the previous task
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
            self._set_task_done()
            self.collect_env_stats()
            self.close()
        # if the next task id is the current one
        if task_id == self._curr_task_id:
            if allow_revisit:
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
        if self.get_stats_property(task_id, "finished"):
            if allow_revisit:
                #reset the stats for this task
                self._reset_task(task_id)
            else:
                raise ValueError("Trying to set an already visited task! "
                    "Set 'allow_revisit' if this is intended.")

        self._curr_task_id = task_id
        self._curr_task_name = self.get_task_name(task_id)
        env = self.get_task_property(
            self._curr_task_id, "env_creator")(
                self.get_task_property(self._curr_task_id, "env_config")
                )

        if not is_atari(env):
            raise ValueError("Currently only env creators are supported that create a gym AtariEnv!")

        if self._add_monitor and not get_wrapper_by_cls(env, MonitorEnv):
            # Wrap Monitor around base env to record stats per trained env
            #checking if an env creator already includes the MonitorEnv 
            #(e.g. the deepmind wrapper) can only be done after env creation
            env = MonitorEnv(env)
        
        super().__init__(env)
        # self.env = env
        # self._dyn_delegate_env_members()

        if reset_on_switch:
            return self.reset()



    def add_task(self, env_id_or_creator, env_config, task_id=None):
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
        
        task_entry = self._make_task_list_entry(env_id_or_creator, env_config)
        self.task_list.insert(task_id, task_entry)

    
    def get_task_property(self, task_id, property_):
        """Get a specific property of a given task is.
        Returns None if the task doesn't exists.
        """
        try:
            return self.task_list[task_id].get(property_, None)
        except IndexError as ie:
            raise IndexError('Given task ID is not in range of the current task list!\n{}'.format(ie))
        except TypeError as te:
            raise TypeError('Looks like the env is not initialized! Did you forget to call next_task()?\n{}'.format(te))
    
    def _set_task_property(self, task_id, property_, value):
        try:
            self.task_list[task_id].update({property_: value})
        except IndexError as ie:
            raise IndexError('Given task ID is not in range of the current task list!\n{}'.format(ie))
        except TypeError as te:
            raise TypeError('Looks like the env is not initialized! Did you forget to call next_task()?\n{}'.format(te))


    def get_stats_property(self, task_id, property_):
        """Gets a specific property of the statistics for the given task ID.
         Returns None if the task doesn't exists.
         """
        return self.get_task_stats(task_id).get(property_, None)
    
    def _set_stats_property(self, task_id, property_, value):
        try:
            self.task_list[task_id]['task_stats'].update({property_: value})
        except IndexError as ie:
            raise IndexError('Given task ID is not in range of the current task list!\n{}'.format(ie))
        except TypeError as te:
            raise TypeError('Looks like the env is not initialized! Did you forget to call next_task()?\n{}'.format(te))
    

    def get_task_name(self, task_id=None):
        """Returns the task name of the given task ID.
        If no task ID given, returns the current task name.
        """
        if task_id is None: task_id=self._curr_task_id
        return self.get_task_property(task_id, 'task_name')

    def get_task_names(self):
        """Returns the ordered list of the task names"""
        return [self.get_task_name(i) for i in range(len(self.task_list))]

    
    def get_task_stats(self, task_id=None):
        """Retuns the stats for a specific task ID.
        If no task ID given, returns the current task stats.
        """
        if task_id is None: task_id=self._curr_task_id
        return self.get_task_property(task_id, 'task_stats') or self.default_per_env_metrics

    def get_stats(self):
        """Returns all the statistics for all tasks"""
        return [self.get_task_stats(i) for i in range(len(self.task_list))]


    def collect_env_stats(self):
        """Collects the current env stats and writes them to the task_stats entry.
        This is called when the env is switched. 
        If metrics are required in between env sampling this can be called manually.
        Returns the stats dict of the current task.
        """
        #we need access to the private attributes of the monitor in order to
        # get the stats of the current episode, since episode stats are only 
        # written to public members during reset. Depending on which other 
        # wrappers are set, a normal reset call not always triggers the 
        # actual env reset (and thus the Monitor to start a new episode) bc 
        # some wrappers like EpisodicLife only let a reset happen when the
        # step function reports it.
        monitor = get_wrapper_by_cls(self.env, MonitorEnv)
        if not (getattr(self, "env") and monitor):
            return None

        curr_ep_steps = monitor._num_steps or 0
        curr_ep_rew = monitor._current_reward or 0
        if curr_ep_steps != 0 or curr_ep_rew != 0:
            all_ep_lengths = self.get_episode_lengths() + [curr_ep_steps]
            all_ep_rewards = self.get_episode_rewards() + [curr_ep_rew]
        else:
            all_ep_lengths = self.get_episode_lengths()
            all_ep_rewards = self.get_episode_rewards()

        self._set_stats_property(
            self._curr_task_id, "total_steps", self.get_total_steps())
        self._set_stats_property(
            self._curr_task_id, "total_episodes", len(all_ep_lengths))
        self._set_stats_property(
            self._curr_task_id, "steps_per_episode", all_ep_lengths)
        mean_ep_len = sum(all_ep_lengths)/(len(all_ep_lengths) or 1)
        self._set_stats_property(
            self._curr_task_id, "mean_episode_length", mean_ep_len)
        self._set_stats_property(
            self._curr_task_id, "total_reward", sum(all_ep_rewards))
        self._set_stats_property(
            self._curr_task_id, "rewards_per_episode", all_ep_rewards)
        mean_ep_rew = sum(all_ep_rewards)/(len(all_ep_rewards) or 1)
        self._set_stats_property(
            self._curr_task_id, "mean_reward", mean_ep_rew)
        
        return self.get_task_stats()


    def _dyn_delegate_env_members(self, full_delegation=False):
        """This is the crutial delegation updater that makes this
        class look like the underlying env of the current task!
        This delegates either public or ALL members of the most
        outer underlying env object to this top-level.
        The update strategy is (in this order):
        * collect all attributes in `_DELEGATION_LIST` (=default 
            delegation list + members from the previous env).
        * exclude protected members starting with '__'.
        * exclude attributes included in `_DELEGATE_NEVER`
            (=default list + class members @ obj creation).

        Removes delegated attributes from previous task, if:
        * member was delegated before (i.e. is in`_DELEGATION_LIST`).
        * attribute is registered in `self` but not in the env.

        If `full_delegation` is True, then ALL members will be
        delegated instead of only the public object attributes. 
        """
        to_delegate = self._DELEGATION_LIST[:] # get the previous/default delegations
        #Get the list of new members in self.env that need to be registered
        curr_env_members = self.env.__dict__.keys() if not full_delegation else dir(self.env)
        for _d in  curr_env_members:
            if _d.startswith('__'): #dont add protected members
                continue
            if _d in self._DELEGATE_NEVER:  #_d is protected from override
                continue
            if _d in self._DELEGATION_LIST: #dont add dublicates
                continue
            to_delegate.append(_d)

        #Check the previously delegated members for deprecated ones
        to_remove = []
        for _d in self._DELEGATION_LIST:
            if hasattr(self, _d) and not hasattr(self._curr_env, _d):
                #member is in self but NOT in self.env
                to_remove.append(_d)

        for _d in to_remove:
            try:
                delattr(self, _d)
            except AttributeError:
                try:        #if the member is a class attribute
                    delattr(ContinualEnv, _d)
                except AttributeError as ae:
                    raise AttributeError(
                        "Cannot delete attribute '{}' from self! {} is neither ".format(_d,_d) +\
                        "an object nor class attribute of this object!\n{}".format(ae))
        
        #Update the internal delegation list with ALL members delegated for this task
        self._DELEGATION_LIST = [e for e in to_delegate if e not in to_remove]
        for _d in self._DELEGATION_LIST:
            # these checks are needed here again because the initial _DELEGATION_LIST
            # list shall be kept unchanged but can have entries not in current self.env
            if _d in self._DELEGATE_NEVER:
                continue
            if not hasattr(self._curr_env, _d):
                continue
            try:
                setattr(self, _d, getattr(self._curr_env, _d))
            except AttributeError:
                try:        #if the member is a class attribute
                    setattr(ContinualEnv, _d, getattr(self._curr_env, _d))
                except AttributeError as ae:
                    raise AttributeError(
                        "Cannot delegate '{}' from self.env to self! {} is neither ".format(_d, _d) +\
                        "an object nor class attribute of the current env!\n{}".format(ae))

      
    def _set_task_done(self):
        try:
            self._set_stats_property(self._curr_task_id, "finished", True)
        except TypeError as te:
            raise TypeError('Looks like the env is not initialized! Did you forget to call next_task()?\n{}'.format(te))

    @classmethod
    def _make_task_list_entry(cls, env_id_or_creator, env_config):
        """Combine the env ID and config to a valid task list entry"""
        if isinstance(env_id_or_creator,str):
            task_name = env_id_or_creator
            env_creator = get_env_creator(env_id_or_creator, env_config)
        elif callable(env_id_or_creator):
            task_name = env_id_or_creator.__name__
            env_creator = env_id_or_creator    #must be an env creator callable: f(env_config)->env obj
        else:
            raise ValueError("Invalid object to add as task! Must be string or callable, is {}".format(type(env_id_or_creator)))

        task_entry = {
                "_task": env_id_or_creator,
                "task_name": task_name,
                "env_config": env_config,
                "env_creator": env_creator,
                "task_stats": deepcopy(cls.default_per_env_metrics)
            }
        return task_entry

    def _reset_task(self, task_id):
        default_entry = self._make_task_list_entry(
            self.get_task_property(task_id, "_task"), 
            self.get_task_property(task_id, "env_config"))
        try:
            self.task_list[task_id] = default_entry
        except IndexError:
            return



