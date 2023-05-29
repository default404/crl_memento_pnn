
"""General utility functions"""
import os
import logging
import yaml, collections, copy
from pathlib import Path

import ray
from ray.tune import grid_search
from ray.rllib.agents.trainer import COMMON_CONFIG as RLlib_CONFIG



def setup_logger(name=None, 
                log_path=None, 
                file_mode='w', 
                verbose=False):
    """Sets the logger to log info in terminal and file `log_path`.
    `mode` specifies wether a possibly existing log file shall be overwritten or appended.

    Args:
        name: (str) name of the logger
        log_path: (str) where the logfile goes
        file_mode: (str) write mode of the logger, e.g. "w" (overwrite old file) "a" (append)
        verbose: (bool) the log level displayed on stdout
    """

    if name:
        logger = logging.getLogger(name)    #get / create logger with name
    else:    
        logger = logging.getLogger()    #no name = get the root logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = 0

    # if logger with that name has no handlers = uninitialized
    if not logger.handlers:
        if log_path is not None:
            # Logging to a file
            log_path = Path(log_path)
            if not log_path.suffix:
                if not log_path.is_dir():
                    raise NotADirectoryError(
                        'LoggerSetup: Given log dir [{}] is not a directory!'.format(log_path))
                log_path = log_path/"experiment.log"

            file_handler = logging.FileHandler(str(log_path), mode=file_mode)
            file_handler.setFormatter(
                logging.Formatter(fmt='%(asctime)s : %(name)s : %(levelname)-7s :: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")
                )
            file_handler.setLevel(logging.DEBUG) #filelogger always catches everything
            logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter(fmt='%(asctime)s : %(levelname)-7s :: %(message)s',
                datefmt="%H:%M")
            )
        if verbose:
            stream_handler.setLevel(logging.DEBUG)
        else:
            stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

    return logger

_logger = setup_logger(name=__name__,verbose=True)


def make_or_clean_dir(path, check_only=False, logger=_logger):
    """Checks for the given path to be a directory and also removes all files from it.
    If `check_only` is true, then no files will be removed from an existing path.
    Returns whether the given directory is empty.
    """
    is_empty = True
    path = Path(path)
    if path.is_dir():
        #collect all files in the dir
        files, dirs = [], []
        for root,ds,fs in os.walk(str(path), topdown=False):
            for name in fs:
                files.append(os.path.join(root, name))
            for name in ds:
                dirs.append(os.path.join(root, name))
        
        if files or dirs:
            logger.debug("Directory {} is not empty!".format(path))
            if check_only:
                is_empty = False
            else:
                logger.debug("Removing remaining files...")
                for e in files:
                    os.remove(e)
                for e in dirs:
                    os.rmdir(e)

    else:
        logger.info("The given output path does not exist! It will be created for you now...")
        os.makedirs(str(path))
    
    return is_empty

def make_new_dir(path, override=False, logger=_logger):
    """Checks if the given path already exists and is empty.
    If not existing it will be created.
    If not empty a new output directory will be created based
    on the original name with increasing number.
    If override is True, the content of existing directories will
    be cleared.
    """
    orig_path = Path(path)
    path = orig_path
    if path.is_file():
        raise ValueError('Given output path must be a directory! Got [{}]'.format(path))
    i = 2
    while path.is_dir():
        #collect all files in the dir
        files, dirs = [], []
        for root,ds,fs in os.walk(str(path), topdown=False):
            for name in fs:
                files.append(os.path.join(root, name))
            for name in ds:
                dirs.append(os.path.join(root, name))
        
        if files or dirs:       # dir not empty
            logger.debug('Output path is not empty...')
            if override:        # remove all content
                logger.debug("...removing remaining files")
                for e in files:
                    os.remove(e)
                for e in dirs:
                    os.rmdir(e)
            else:               # find a new one
                path = orig_path.parent / "_".join([orig_path.name, str(i)])
                logger.debug('...trying alt. dir [{}]'.format(path))
                i += 1
        else:
            #dir exists but is empty
            break

    if not path.exists():
        logger.info("The output path does not exist! It will be created for you now...")
        os.makedirs(str(path))
    
    return path


def dict_deep_update(d, u):
    """Updates `d` with `u` without overriding (top-)level keys entirely
    when already present in `d`.
    """
    if u:
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = dict_deep_update(d.get(k, {}), v)
            else:
                d[k] = v
    return d


def load_yml_config(path):
    path = Path(path)
    if not path.is_file() or path.suffix  not in ['.yml','.yaml']:
        raise ValueError("LoadConfig: Given path is not a valid YML file!")
    with open(path, 'r') as cf:
        return yaml.safe_load(cf)


def load_config_and_update(baseYaml, updatesYaml=None, updatesDict=None):
    """Loads the base config and updates its fields with an additional
    config file (`updatesYaml`) or and dictionary (`updatesDict`).
    The parent config is determined from which the resulting config
    originates from and stored in 
        ["auto_managed"]["parent_config"]
    Update order is: 
        baseYaml -> updatesYaml -> updatesDict
    """
    #load and return the base config always
    config = load_yml_config(baseYaml)
    parent_config = str(baseYaml)

    #update base with additional yaml
    if updatesYaml is not None:
        config = dict_deep_update(config, load_yml_config(updatesYaml))
    
    #update base with additional dict
    if isinstance(updatesDict,dict):
        config = dict_deep_update(config, updatesDict)

    try:
        parent_config = config["auto_managed"].get("parent_config")
    except KeyError:
        pass
    else:   #if auto_managed field is there
        if not parent_config:   #if the field is there but empty
            parent_config = str(updatesYaml) if updatesYaml else str(baseYaml)
        config["auto_managed"]["parent_config"] = parent_config

    return config


def list_to_gridsearch(d):
    """Recursively loops though all the elements 
    of a dictionary and converts any list elements 
    to a tune.gridsearch object. 
    """
    ret = {}    #return a copy of d to not change the inital param
    if isinstance(d, dict):
        for k,v in d.items():
            if k == 'grid_search':  #dont convert grirdsearch entries again
                ret[k] = v
            else:
                ret[k] = list_to_gridsearch(v)
    elif isinstance(d, list):
        ret = grid_search(d)
    else:
        ret = d

    return ret


def remove_gridsearch(d):
    """Recursively loops though all the elements 
    of a dictionary and removes any gridsearch obj 
    from it by picking the first element from the search. 
    """
    ret = {}    #return a copy of d to not change the inital param
    if isinstance(d, dict):
        if 'grid_search' in d.keys():
            ret = d['grid_search'][0]
        else:
            for k,v in d.items():
                if isinstance(v, dict):
                    ret[k] = remove_gridsearch(v)
                else:
                    ret[k] = v
    else:
        ret = d

    return ret

def convert_to_tune_config(config, inclUnknownKeys=False, logger=_logger):
    """Converts the given config to a compatible Ray config that can be used by 
    the Tune runner.

    If inclUnknownKeys is True, algorithm and continual params are included in
    the resulting config (note that this can raise errors in the agent if it
    does not allow unknown config keys).
    It is allowed to add sub-keys to the following config keys by default:
        "tf_session_args", "local_tf_session_args", "env_config", "model",
        "optimizer", "multiagent", "custom_resources_per_worker",
        "evaluation_config", "exploration_config",
        "extra_python_environs_for_driver", "extra_python_environs_for_worker"
    """
    algo_in_use = config.get('algorithm')
    #if missing, no continual params are written to config
    cont_params = config.get('Continual_params', {})    
    ray_config = copy.deepcopy(RLlib_CONFIG)
    # settings needed for CL but cannot easily inserted in the Ray config
    additional_params = {
        "algorithm": algo_in_use,
        "algo_config": cont_params.get('algo_params', {})
        }

    # convert any list parameter in the config to a grid search object
    rllib_params = list_to_gridsearch(config.get('Trainer_params',{}))
    ray_config.update(rllib_params)

    # add custom model options to ray config (only used if custom algo is selected)
    # handle special keys in the algo configs to be converted to a gridsearch obj for HP optimization by Tune
    algo_config = additional_params['algo_config'].get(algo_in_use, None)
    if not isinstance(algo_config, dict): 
        algo_config = {}
    special_keys = ['alpha_value', 'fcnet_activation', 'conv_activation']   #extend this to your desire
    for k in algo_config.keys():
        if k in special_keys:
            algo_config[k] = list_to_gridsearch(algo_config[k])
    dict_deep_update(ray_config['model']['custom_model_config'], algo_config)
    
    # update the ray env_config field with the additional env configs
    dict_deep_update(ray_config['env_config'], cont_params.get('env_config', {}))

    task_list = cont_params.get('task_list',[]) #for now this is just a list of env strings
    ray_config['env_config'].update({'task_list': task_list, 'per_env_config': {}})    
    
    #check which algo will be run
    if algo_in_use == "vanilla_ppo" and len(task_list) > 1:
        logger.warning('Selected algorithm {} not suitable for training '.format(algo_in_use) +\
            'on multiple tasks. You maybe experience unexpected behavior, like e.g. only ' \
            'the first task will be run, or the agent will be trained on each task with ' \
            'catastrophic forgetting effects, etc...')

    # multiple tasks are trained, their configs are stored in per-env-config 
    if task_list:
        #update the env key depending on how many tasks are given
        if len(task_list) > 1:  # continual learning mode
            ray_config.update(env=None)
        else:   # single task training using the task list
            ray_config.update(env=task_list[0])
        #add the per-env-config entries to ray config
        for env_id in task_list:
            if env_id not in cont_params.get('per_env_config',{}).keys():
                logger.warning("Env ID '{}' from the task list is missing its configuration in "
                "the [per_env_config] settings. The default configuration will be used.".format(env_id))
                ray_config['env_config']['per_env_config'][env_id] = cont_params['default_wrapper_params']
            else:
                conf = cont_params['per_env_config'][env_id]
                if isinstance(conf,dict):
                    ray_config['env_config']['per_env_config'][env_id] = conf
                else:
                    ray_config['env_config']['per_env_config'][env_id] = cont_params['default_wrapper_params']
    # no task list specified (single env in ray config)
    elif ray_config.get('env'):
        #add the single env to the task list
        single_env = ray_config.get('env')
        ray_config['env_config'].update({'task_list':[single_env],
                                         'per_env_config':{
                                             single_env: cont_params['default_wrapper_params']
                                         } })
    else:
        raise ValueError('Neither a task-list, nor a single Env ID in the Trainer params '
                         'is specified. Please give an Env to train on!')

    # for single task, env configuration could be controlled by Rllib Trainer
    # -> inject the custom env params into the env-config to find it
    if len(ray_config['env_config']['task_list']) == 1 and \
            cont_params['env_config'].get('use_custom_env'):
        t_name = ray_config['env_config']['task_list'][0]
        ray_config['env_config'].update(
            ray_config['env_config']['per_env_config'][t_name]
        )
        
    if inclUnknownKeys:
        ray_config = dict_deep_update(ray_config, additional_params)
    return ray_config, additional_params


def get_experiment_info(config, **kwargs):
    """Returns some of the experiment configurations in a pretty 
    formated string.
    """ 
    info_str = ''
    trainable = kwargs.get('trainable', '-')
    devices = '{} CPU(s) | {} GPU(s)'.format(config.get('num_workers', 0)+1, config.get('num_gpus', 0))
    task_env_infos = ['\n  - {}'.format(t) for t in config['env_config'].get('task_list',[])]
    if 'additional_env_info' in kwargs:
        task_env_infos = ['{:<30} : {} -> {}'.format(t, info['observation_space'], 
                                                     info['action_space']) 
                          for t,info in zip(task_env_infos, kwargs['additional_env_info'])]
    clbk = config.get('callbacks', 'Default')
    if callable(clbk):
        clbk = clbk.__name__ 
    elif type(clbk) in (dict,list,tuple):
        clbk = ''.join(['\n  - {}'.format(c.__name__ if callable(c) else c) for c in clbk])
    stop = kwargs.get('stop') or config['env_config'].get('stopping_criteria',{})

    info_str += 'Trainable:     {}\n'.format(trainable.__name__ if callable(trainable) else str(trainable))
    info_str += 'Algorithm:     {}\n'.format(config.get('algorithm') or kwargs.get('algorithm', '-'))
    info_str += 'Custom Model:  {}\n'.format(config['model'].get('custom_model') or 'No')
    info_str += 'Task List:     {}\n'.format(''.join(task_env_infos))
    info_str += 'Env Type:      {}\n'.format('custom' if config['env_config'].get('use_custom_env') else 'rllib')
    info_str += 'Stopping:      {}\n'.format(''.join(['\n  - {}: {}'.format(k,v) for k,v in stop.items()]))
    info_str += 'Devices:       {}\n'.format(devices)
    info_str += 'Callbacks:     {}\n'.format(clbk)
    info_str += 'Debug Mode:    {}\n'.format('ON' if kwargs.get('debug') else 'OFF')

    if kwargs.get('debug'):
        info_str += 'INFO: Running Ray in serialization mode (debugging), make sure to disable for actual runs.\n'
    
    return info_str


def get_slurm_resources(config, logger=_logger):
    """Looks up the resources reserved by SLURM for the job,
    checks if the Ray configuration exeeds the limit and 
    corrects the config if necessary.
    """
    slurm_cpus = os.environ.get("SLURM_JOB_CPUS_PER_NODE")
    if slurm_cpus:
        ray_thread_demand = int(config.get('num_workers', -1))
        max_worker_threads = int(slurm_cpus)*2 - 1
        if ray_thread_demand > max_worker_threads:
            logger.warning('You specified more workers than threads are assigned '
                           'for the job by SLURM! Adjusting worker number to '
                           '{} to match available resources'.format(max_worker_threads))
            config.update(num_workers=max_worker_threads)
    else:
        logger.info('Cannot determine max CPUs assigned for job by SLURM. Using '
                    'more workers than max number of threads can decrease performance!')
    
    slurm_gpus = os.environ.get("SLURM_GPUS")
    if slurm_gpus:
        ray_gpu_demand = int(config.get('num_gpus', 0))
        assigned_gpus = int(slurm_gpus)
        if ray_gpu_demand > assigned_gpus:
            logger.warning('You specified more GPUs than SLURM assigned '
                           'to the job! Adjusting number of GPUs to '
                           '{} to match available resources'.format(assigned_gpus))
            config.update(num_gpus=assigned_gpus)
    else:
        logger.info('Cannot determine number of assigned GPUs for job by SLURM. Using '
                    'more GPUs than available will crash Ray!')
    return slurm_cpus, slurm_gpus


def save_model_stats(model, out_path, framework='torch', input_shape=None):
    '''Generates a summary string similar to Keras model summary.
    '''
    output_file = Path(out_path)/f'{framework}_model_summary.txt'
    if framework == 'torch':
        try:
            from common.torchsummary import summary_string
        except ImportError:
            ret_str = 'Could not import torch-summary function from common.torchsummary'
            return ret_str, (None, None)
        
        summary_str, (total_params, trainable_params) = \
            summary_string(model, input_size=input_shape, 
                           device=next(model.parameters()).device)
        
        with open(output_file,'w') as f:
            f.write(summary_str)

    elif framework == 'tf':
        from ray.rllib.models.modelv2 import ModelV2
        with open(output_file,'w') as f:
            if isinstance(model, ModelV2):
                model.base_model.summary(print_fn=lambda x: f.write(x + '\n'))
            else:
                model.summary(print_fn=lambda x: f.write(x + '\n'))


def find_last_ckpt(directory, suffix='.h5', logger=_logger):
    """retrieves the LAST model checkpoint from a directory which 
    has multiple checkpoints. This does NOT check taskID's in the 
    file name nor does it any kind of ordering.
    """
    import re
    path = Path(directory)

    #last checkpoint is determined by highest number in the file name
    last_ckpt = None
    last_step = -1
    for ckpt in path.glob(f'*{suffix}'):
        try:
            new_step = max(list(map(int,re.findall(r'[0-9]+',ckpt.stem))))
        except ValueError:
            logger.warning(f'No number in the checkpoint file name \'{ckpt.stem}\' to determine last one!')
        else:
            if new_step > last_step:
                last_step = new_step
                last_ckpt = str(ckpt)

    logger.debug(f'Found last model checkpoint with iteration/timestep {last_step}')
    return last_ckpt


def find_model_ckpts_per_task(directory, suffix='.h5', logger=_logger):
    """Get the model checkpoints from a given directory recursively.
    Returns a dictionary with: task ID -> path to last checkpoint
    """
    import re
    path = Path(directory)

    checkpoints = {}
    #sort by common base dir
    base_to_ckpt = {}
    for ckpt in path.rglob(f'*{suffix}'):
        if str(ckpt.parent) not in base_to_ckpt.keys():
            base_to_ckpt[str(ckpt.parent)] = [ckpt]
        else:
            base_to_ckpt[str(ckpt.parent)].append(ckpt)

    for base, ckpts in base_to_ckpt.items():
        tmp = ckpts[0]
        try:
            taskStr = re.findall(r'T[s]?[-]?[0-9]+',str(tmp.name))[-1]
            task_id = int(re.findall(r'[0-9]+',taskStr)[0])
        except:
            logger.warning(f'Could not determine task ID for checkpoints in base dir: {base}.\n '
                           f'Tested {tmp}. \nCheckpoints will be skipped!')
        else:
            try:
                checkpoints[task_id]
            except KeyError:
                pass
            else:
                logger.warning(f'Found multiple directories with ckpt-files that match the same task ID: {task_id}\n'+\
                    f'Current store: {checkpoints[task_id]}\nNext dir match: {base}\n'+\
                    'Please check the ckpt directory for ambiguous files!')
            finally:
                checkpoints[task_id] = find_last_ckpt(base, suffix, logger)

    if not checkpoints:
        logger.warning(f'Found no checkpoint files ({suffix}) in the given directory {directory}')

    return checkpoints
