###################################################
#
# Environment wrapper functionality for the Gym
# Atari games.
# from OpenAI Baselines https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
#
###################################################
import shutil, pickle, copy
import gym

from common.util import setup_logger, remove_gridsearch

_logger = setup_logger(name=__name__,verbose=True)

ENV_SPEC_TABLE = 'env_spec_looku.p'


def get_env_creator(env_id, env_name, fact_config, logger=_logger):
    '''Returns an env factory according to fact_config and env ID and sets other
    config values such that the env is not wraped again by Rllib.
    '''
    #check for special envs specified by ID string
    from envs import PONG_SOUP_ID
    if env_id in PONG_SOUP_ID:
        from envs.pong_soup.pong_env_wrapper import make_pong_soup
        env_creator = make_pong_soup(env_id, fact_config, logger)
    else:   #default atari factory, ignores the config
        env_creator = lambda config: gym.make(env_id)
    
    if fact_config['env_config'].get('use_custom_env'):
        # use custom env, disable Rllib wrapping
        from envs.atari_env_wrapper import wrap_atari
        logger.debug('Using custom env wrapping; disables Rllib wrappers via config')
        fact_config.update(preprocessor_pref='NoPreprocessor')
        # env_config = fact_config['env_config']['per_env_config'][env_id]
        env_creator = wrap_atari(env_creator, fact_config['env_config'], logger)
    else:
        #use default Rllib wrapping
        logger.debug('Using default Rllib wrappers in the RolloutWorkers')

    if fact_config['env_config'].get('normalize_observations'):
        logger.debug('Adding observation normalization wrapper to env')
        if not fact_config['env_config'].get('use_custom_env'):
            logger.warning('WARNING: Adding observation normalization is order-dependent!\n'\
                'Please also enable `use_custom_env` in the config, else env sampling will '\
                'fail because Rllib adds its wrappers only AFTER this one which will cause '\
                'image operations to fail on the the normalized frames.')

    if fact_config['env_config'].get('use_memento_env'):
        logger.debug('Adding Memento wrapper to env for state initialization')
    
    def build_env(config):
        #this factory ensures that each respective `per_env_config` settings
        #are added to the root namespace of the accessable config 
        env_config = config.copy()
        env_config.update(config['per_env_config'].get(env_name, {}))
        env = env_creator(env_config)
        if env_config.get('normalize_observations'):
            from envs.gym_wrappers import NormObservation
            env = NormObservation(env,
                                  mode=env_config.get('normalize_obs_mode'),
                                  **env_config)
        if env_config.get('use_memento_env'):
            from envs.atari_env_wrapper import MementoEnv
            env = MementoEnv(env, env_config['memento_state_buffer_path'])
        return env
    
    return build_env



def register_envs(config, logger=_logger):
    '''Registers custom envs in Rllib if needed and update the task list names 
    and per-env-config keys for the registered envs.
    '''
    from ray.tune.registry import register_env
    new_task_list = []
    for env_id in config['env_config']['task_list']:
        env_name = env_id
        #get list of all registered gym envs
        # is_base_atari = True
        # if env_id not in [e.id for e in gym.envs.registry.all()]:
        #     is_base_atari = False

        # # env needs registration
        # if not is_base_atari or \
        #     config['env_config'].get('use_custom_env') or \
        #     config['env_config'].get('use_memento_env'):

        if config['env_config'].get('use_custom_env'):
            #custom wrapping is applied to env
            env_name = env_name + "_custom"
        elif config['env_config'].get('use_memento_env'):
            #no custom wrapping but Memento env is used
            env_name = env_name + "_memento"

        if env_name != env_id:
            #update the per-env-config key names
            env_id_config = config['env_config']['per_env_config'].pop(env_id)
            config['env_config']['per_env_config'][env_name] = env_id_config
        if config.get('env', '') == env_id:
            #update the global env key if needed (relevant for single task)
            config.update(env=env_name)

        logger.debug(f'Registering env: {env_name}')
        env_factory = get_env_creator(env_id, env_name, config, logger)
        register_env(env_name, env_factory)
        
        new_task_list.append(env_name)

    config['env_config'].update(task_list=new_task_list)



def get_env_infos(tasklist, config, logger=_logger) -> list:
    '''Retrieves the Env spaces needed for each task for 
    continual model initialization/restoring.
    After a continual model (like PNN) checkpoint is saved,
    the same info is dumped and can be loaded w/o this function.
    Note:
        * this is an expensive function 
        * custom envs must be registered in Tune before and 
          Ray must already be initialized!
    Returns a list (same order at given tasklist) of
    env infos.
    '''
    from ray.rllib.agents.ppo import PPOTrainer
    tmp_config = copy.deepcopy(config)
    #change the temp config to reduce Trainer creation overhead as far as possible
    tmp_config['log_level'] = 'ERROR'
    tmp_config['monitor'] = False
    tmp_config['num_workers'] = 0
    tmp_config['num_envs_per_worker'] = 1
    tmp_config['num_gpus'] = 0
    tmp_config['model']['custom_model_config']['checkpoint'] = None
    tmp_config = remove_gridsearch(tmp_config)

    task_envSpec = []
    for task_id, env_id in enumerate(tasklist):
        trainer = None
        # env spec not in the lookup table, needs to be evaluated
        try:
            #NOTE: for now only PPOTrainer considered
            trainer = PPOTrainer(config=tmp_config, env=env_id)
            obs = trainer.workers.local_worker().env.observation_space
            act = trainer.workers.local_worker().env.action_space
        except RuntimeError as re:
            logger.error('Tried to create a Trainer befor ray.init() was called?')
            raise RuntimeError from re
        except gym.error.Error as gymerr:
            logger.error(f'The env ID {env_id} is not a Gym env or registered in Tune!')
            raise gym.error.Error from gymerr
        except Exception as e:
            logger.error(f'Error during Trainer creation for env {env_id}:\n{e}')
            raise Exception from e
        else:
            #TODO: even with trainer stop this causes (non-crashing) exceptions in 
            # the main process at random times saying that a tempfile for these 
            # trainers is missing (env monitor dirs)
            if trainer:
                trainer.stop()
                shutil.rmtree(trainer.logdir, ignore_errors=True)
            # del trainer  #maybe dont manually delete the trainer and let gc pick it up
            
            envSpec = {
                'task_id': task_id,
                'observation_space': obs,
                'action_space': act
            }

        task_envSpec.append(envSpec)
    
    return task_envSpec     #returns only the env specs for the current task list

    