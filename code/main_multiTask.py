###################################################
#
# Main function of the continual reinforcement 
# learner. The environments are designed to be 
# provided by OpenAI Gym and the agents are using a
# different non-standart ploicies with a comination
# of EWC penalty or PNN arichtecture or both.
#
###################################################

import os, sys, inspect
__MAINDIR__ = os.path.dirname(os.path.realpath(inspect.getfile(lambda: None)))
print('Main dir:', __MAINDIR__)
if not __MAINDIR__ in sys.path: #insert the project dir in the sys path to find modules
    sys.path.insert(0, __MAINDIR__)
#NOTE: The RolloutWorkers do not receive the Driver system paths!
print('System Path:\n',sys.path)

import argparse, yaml, logging, pickle
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import numpy as np

import ray
if tuple(map(int, ray.__version__.split('.'))) < (0,8,6):
    raise EnvironmentError('This version of the training script only supports Ray >= 0.8.6. '
                           'Please update the package and all associated dependencies.')
from ray import tune
# from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print

# utilities
from common.util import (setup_logger, make_new_dir, get_experiment_info,
    load_config_and_update, convert_to_tune_config, dict_deep_update, 
    get_slurm_resources, save_model_stats, find_last_ckpt, remove_gridsearch)
from common.custom_callbacks import UnifiedCallbacks
import common.exp_analysis as expAnalysis

# atari envs
from envs.ray_env_util import register_envs, get_env_infos

# continual policies / models
from continual_atari.agents.policies.pnn.pnn_model import PNN


# handle the cwd for script execution
__ORIG_CWD__ = os.getcwd()
__NEW_CWD__ = __MAINDIR__


#The path to the default config file (used if no config is given at runtime)
_DEFAULT_CONFIG = Path("continual_atari/configs/default_config.yml")

def parse_args():
    parser = argparse.ArgumentParser(prog='Continual RL training pipeline',
        description='Continual reinforcement learning with EWC and PNN on PPO',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', dest='config', type=str, default=_DEFAULT_CONFIG,
        help='Path to the config file to update the default config for the experiment.\n' 
        'If not set, only the default config [{}] is used'.format(_DEFAULT_CONFIG))
    parser.add_argument('--output', dest='output_path', required=True, type=str, 
        help='Path where the output (= logs, configs, savepoints,...) will be stored. '
        'If the destination does not exist it will be created.')
    parser.add_argument('--override', dest='override_old', action="store_true", default=False,
        help='If specified, output directories with content in them will be overridden '
        'instead of a new directory with a new name created.')
    parser.add_argument("--debug-ray", dest='debug_ray', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']),
        help='If specified, Ray will be run serialized in debug mode (local without parallelization).')
    parser.add_argument("--stop-iters", dest='training_iteration', type=int,
        help='Stopping criterion for the agent training. Terminates after `stop-iters` Trainer iterations.')
    parser.add_argument("--stop-timesteps", dest='timesteps_total', type=int,
        help='Stopping criterion for the agent training. Terminates after `stop-timesteps` env steps.')
    parser.add_argument("--stop-reward", dest='episode_reward_mean', type=float,
        help='Stopping criterion for the agent training. Terminates after `stop-reward` reward is achieved.')
    parser.add_argument('--task_min_max', dest='task_min_max', type=yaml.load, default="{}",
        help='Dict with the min and max performance (e.g. reward) for each task for plotting purposes.\n'
        'Perfromance values are given as low, high for each task ID. E.g.:\n "{0: [-21., 21.], 1: [0., 3000.]}"\n'
        'If not given, no normalization of the individual tasks is done which CAN lead to tilted graphs for envs '
        'with greatly different reward ranges.')

    args = parser.parse_args()
    return args

### Reward Table:
#(gym leaderboard best 100-ep)
# Pong-v0:      20.81 ± 0.04
# Breakout-v0:  760.07 ± 18.37



def get_args_dict(args):
    """returns the dict view from the command line args from the argparse object.
    Also renames / removes some of the arguments, that are given on cmd.
    """
    d = vars(args)
    d.pop('override_old', None)
    d.pop('debug_ray', None)
        
    return d 


def trial_str_creator(trial):
    """Generator function for a custom trial name. 
    Can use the Trial information:
        trainable_name (str): Name of the trainable object to be executed.
        config (dict): Provided configuration dictionary with evaluated params.
        trial_id (str): Unique identifier for the trial.
        local_dir (str): Local_dir as passed to tune.run.
        logdir (str): Directory where the trial logs are saved.
        evaluated_params (dict): Evaluated parameters by search algorithm,
        experiment_tag (str): Identifying trial name to show in the console.
        resources (Resources): Amount of resources that this trial will use.
        status (str): One of PENDING, RUNNING, PAUSED, TERMINATED, ERROR/
        error_file (str): Path to the errors that this trial has raised.
    """
    time = datetime.now().strftime("%H:%M:%S")
    return "{}_{}_{}".format(time, trial.trainable_name, trial.trial_id)



def main(args):
    '''
    The config is used to specifiy the experiment parameters.
    For an experiment, the config is READ and COPIED to the output destination in order
    to provide traceback and pick-up information.

    All tasks to be trained on are stored in config->env_config->task_list, even when 
    only single task training is done.
    Use config->env_config->per_env_config[env_id] to get the configuration for a 
    specific environment.

    If `use_custom_env` is set, the task_list will be exchanged with customized env IDs
    under which the custom env creators will be registered in Ray.
    '''
    date = datetime.today().strftime('%Y-%m-%d')
    p_output = Path(args.output_path)
    exp_name = "{}_{}".format(p_output.name, date)
    debug_ray = args.debug_ray

    p_output = make_new_dir(p_output, override=args.override_old)

    #update default config keys and values with commandline args
    # dict_args = get_args_dict(args)
    # setup logger
    logger = setup_logger(name=exp_name, log_path=p_output, verbose=True)
    logger.info("+--------------------------------------------------------------+")
    logger.info("|  STARTING CONTINUAL LEARNING: {:^30} |".format(exp_name))
    logger.info("+--------------------------------------------------------------+")

    # initialize the ray cluster and connect to it (when the cluster is the machine itself)
    ray.init(
        ignore_reinit_error=False,   #ignore error thrown if ray is initialized multiple times
        log_to_driver=True,     #whether the logs shall be send to the driver
        local_mode=debug_ray,    #whether to run ray in debug mode
        include_dashboard=False,    #manage dashbord
        # _lru_evict=True,
        # _memory=1*1024*1024*1024,    #given in byte
        # object_store_memory=1*1024*1024*1024,
        # _driver_object_store_memory=0.5*1024*1024*1024,
        _temp_dir=str(Path('~/no_backup/d1346/tmp/ray').expanduser()),      #the root dir for the ray temp files and logs (default: /tmp/ray/)
        )

    user_config = load_config_and_update(args.config)
    ray_config, aux_dict = convert_to_tune_config(user_config, False, logger)

    # add the stopping criteria which are not None(=disabled)
    stoppers = dict(ray_config['env_config'].get('stopping_criteria', {}))
    stoppers.update({k:getattr(args, k) for k in stoppers.keys() if getattr(args, k) is not None})
    stoppers = {k:v for k,v in stoppers.items() if v is not None}
    if not stoppers:
        logger.warning("\nWARNING: No stopping criteria given, the first task will be trained infinitely!\n")
    ray_config['env_config'].update(stopping_criteria=stoppers)

    # save the updated config yml
    with open(p_output/'exp_config.yml','w') as f:
        yaml.safe_dump(user_config, f, default_flow_style=False)

    #if custom env or own env shall be used
    logger.debug('Registering custom environments if needed...')
    register_envs(ray_config)
        
    # get custom policy/model options if applicable
    algo = aux_dict.get('algorithm')
    trainable = None
    custom_model = None
    if algo == 'vanilla_ppo':
        #the standart Rllib model for single task training
        trainable = 'PPO'
        assert ray_config.get('env') or ray_config['env_config'].get('env'), \
            'missing environment for standart PPO'
    else:
        logger.debug("Registering custom model: {}".format(algo))
        if algo == 'pnn':
            #for PNN only the model must be adapted
            trainable = 'PPO'
            custom_model = PNN
            ray_config['model'].update({
                'custom_model': algo,
                # 'custom_action_dist': None
            })
            ModelCatalog.register_custom_model(algo, custom_model)

        else:
            raise ValueError('The selected algorithm [{}] in the config is not implemented!'.format(algo))
        

    # add a callbacks to Trainer class
    logger.debug('Adding callbacks to Trainer...')
    ray_config.update(callbacks=UnifiedCallbacks)

    # determine HW assignment from SLURM
    get_slurm_resources(ray_config, logger)

    # get env space infos for each task
    logger.info('Collecting environment infos for each task...')
    env_infos = get_env_infos(ray_config['env_config']['task_list'], ray_config, logger)

    logger.info('Running Tune with configuration:\n{}'.format(
                get_experiment_info(ray_config, 
                                    trainable=trainable, 
                                    debug=debug_ray,
                                    additional_env_info=env_infos,
                                    **aux_dict)
                ))

    # create a small lookup file to quickly find the dir of the best experiment for each task
    # if the best exp for a task cannot be determined the dir is stored from which the checkpoint 
    # for the next task was used 
    best_exps_file = p_output / 'best_experiments.txt'
    with open(best_exps_file, 'w') as f:
        f.write('# Paths to the best experiments of each task:\n')
    
    # PNN specific checkpoint loading behavior
    task_id_offset = 0      # if we load a checkpoint we start actually from a greater task ID
    first_task_ckpt = ray_config['model']['custom_model_config'].get('checkpoint')
    if first_task_ckpt:
        #if we have a checkpoint for the first task already, we also need a env spec dump to get info from 
        logger.info('Loading checkpoint for first task: \n\t{}\n'.format(first_task_ckpt) +\
                    'to initialize the continual model.')
        env_spec_file = ray_config['model']['custom_model_config'].get('build_columns', None)
        if not (env_spec_file and Path(env_spec_file).is_file()):
            env_spec_file = list(Path(first_task_ckpt).parent.glob('*env.spec'))[0] #there should only be one
        with open(env_spec_file, 'rb') as f:
            ckpt_env_info = pickle.load(f)
        # if we want to retrain the last column of the PNN with the first task we need to remove the last
        # env spec entry from the checkpoint
        if ray_config['model']['custom_model_config'].get('retrain_last_col', False):
            logger.info('Configuring model to retrain last column loaded from the checkpoint')
            ckpt_env_info = ckpt_env_info[:-1]
        task_id_offset = len(ckpt_env_info)
        #update the task ids with the offset
        for env_spec in env_infos:
            env_spec['task_id'] += task_id_offset
        env_infos = ckpt_env_info + env_infos

    task_list = ray_config['env_config']['task_list']
    analysis_list = []
    start_time = datetime.now().replace(microsecond=0)
    exp_start_time = start_time
    for task_id, task_name in enumerate(task_list):
        task_id += task_id_offset
        logger.info('---')
        logger.info('Running Tune experiment for task {:>2}/{:>2}: {}'.format(task_id+1, len(task_list)+task_id_offset, task_name))
        task_config = deepcopy(ray_config)

        #handle first task specific behavior
        if task_id == 0:
            # remove PNN HP gridsearch from first column / task config bc it has no effect for 
            # the first task and greatly increases the opt space! 
            no_hpOpt_model_config = remove_gridsearch(
                task_config['model']['custom_model_config']['alpha_value']
            )
            task_config['model']['custom_model_config']['alpha_value'] = no_hpOpt_model_config
        
        #configure behavior after the first task was completed
        if task_id > task_id_offset:
            #for the next tasks, the model shall not retrain the last column anymore 
            task_config['model']['custom_model_config']['retrain_last_col'] = False

        task_config['model']['custom_model_config'].update(build_columns=env_infos[:task_id+1])

        if not stoppers:
            # task will be trained indefinitely and stopped manually
            exp_name = f'inf__{task_name}'
        else:
            # we have a (possible) multi-task training
            exp_name = f"T{task_id}_{task_name}"

        # set the current task in the task config
        task_config.update(env = task_name)

        # run the trainable
        analysis = tune.run(
            trainable,      #just PPO for now
            name=exp_name,
            local_dir=str(p_output),
            # checkpoint_freq=10,       #FIXME: this is popbably not applicable with custom trainables
            checkpoint_at_end=True,
            # keep_checkpoints_num=10,
            # checkpoint_score_attr="episode_reward_mean",    #metric to determine the best checkpoints
            # resources_per_trial={"cpu": 1, "gpu": 0},  #default
            config=task_config,
            stop=stoppers,
            # search_alg=BasicVariantGenerator(),   #default
            # scheduler=FIFOScheduler(),       #default
            # with_server=False,     Starts a background Tune server. Needed for using the Client API
            # server_port=,         Port number for launching TuneServer.
            # resume=False | "LOCAL" / True / "PROMPT",  #resume experiment from checkpoint_dir (=local_dir/name); PROMT requires CLI input
            verbose=2   #0 = silent, 1 = only status updates, 2 = status and trial results
        )
        analysis_list.append(analysis)

        # get the dir of the best experiment
        #   criteria is currently: episode_reward_mean
        best_dir = None
        try:
            # in case we ran multiple experiments with HP sweeps try to get the best dir first
            best_dir = analysis.get_best_logdir(metric='episode_reward_mean', mode='max')
            logger.info('Location of best Trial: {}'.format(best_dir))
        except Exception as e:
            logger.error('Couldnt find the best trial directory for '\
                         'task ID {} ({})!\nError: {}'.format(task_id, task_name, e))
            #in case we cannot determine the best experiment, we try to get it by finding the
            #last checkpoint from a dir scan
            ckpt = find_last_ckpt(p_output / exp_name, '.h5', logger=logger)
            best_dir = str(Path(ckpt).parent)
        else:
            ckpt = find_last_ckpt(best_dir, '.h5', logger=logger)

        # write the best trial dir to the overview file
        with open(best_exps_file, 'a') as f:
            f.write(f'Task {task_id} ({task_name}): {best_dir}\n')

        if not ckpt and task_id < len(task_list)+task_id_offset-1:
            raise ValueError(f'Could not find a checkpoint for task {task_id} ({task_name})!\n'
                             'Either the checkpointing callback of the Trainer did not work '
                             f'or the ckpt-file was stored in a differnt location than {p_output}!\n'
                             'Without a checkpoint of the previous task, the model cannot train '
                             'on the next one!')
        ray_config['model']['custom_model_config'].update(checkpoint=str(ckpt))

        exp_time_passed = datetime.now().replace(microsecond=0)
        logger.info('Finished Experiment for {}. Time needed: {}'.format(task_name, exp_time_passed-exp_start_time))
        exp_start_time = exp_time_passed

    logger.info('')
    logger.info(''.join(['=']*40))
    logger.info('All Experiments finished. Total time for the run: {}'.format(datetime.now().replace(microsecond=0)-start_time))

    if first_task_ckpt:
        exp_state_file = expAnalysis.EXPERIMENT_STATE_FILE
        try:
            exp_state_path = list(Path(first_task_ckpt).parent.parent.parent.glob(exp_state_file))[0]
        except:
            logger.warning(f'Couldn\'t fetch the experiment state ({exp_state_file}) from the inital '
                            'checkpointed experiment for consecutive plotting')
        else:
            analysis_list.insert(0, tune.ExperimentAnalysis(str(exp_state_path)))
        
    logger.info('Starting plotting functions for the experiment:')
    # create the default plots from the experiment
    logger.info('')
    logger.info(f'Creating individual plots for each Trial in: {p_output}...')
    try:
        expAnalysis.run_trial_plotter(p_output, 
                                      output=None, 
                                      override=args.override_old, 
                                      logger=logger)
    except Exception as e:
        logger.error(f'Error while running the trial plotter!')
        logger.error(f'Traceback: {e}')
    
    # create averaging plots for one experiment if applicable
    logger.info('')
    logger.info(f'Creating averaging plots for each Tune experiment...')
    options = {'color':'xkcd:blood red', 'height':4, 'aspect':3, 'save':True}
    try:
        expAnalysis.run_exp_average_plotter(analysis_list, 
                                            output=p_output / 'avg_tuneExp_plots', 
                                            override=args.override_old, 
                                            logger=logger, 
                                            **options)
    except Exception as e:
        logger.error(f'Error while running the averaging plotter for each experiment!')
        logger.error(f'Traceback: {e}')
    logger.info(f'Creating trial summary plots for each Tune experiment...')
    try:
        options.update(legend=False)
        for anl in analysis_list:
            expAnalysis.run_task_progress_plotter(anl, 
                                                  task_perf_dict=args.task_min_max,
                                                  trial_selection='all',
                                                  output=p_output / 'summary_tuneExp_plots', 
                                                  override=args.override_old, 
                                                  save_csv=False,
                                                  logger=logger,
                                                  **options)
    except Exception as e:
        logger.error(f'Error while running the trial summary plotter for each experiment!')
        logger.error(f'Traceback: {e}')
    
    # create progress plots for the whole multi-task experiment
    if len(analysis_list) > 1:
        logger.info('')
        logger.info(f'Creating progress plots for all tasks of the experiment...')
        options = {'height':4, 'aspect':3, 'save':True}
        try:
            expAnalysis.run_task_progress_plotter(analysis_list, 
                                                task_perf_dict=args.task_min_max,
                                                trial_selection='all',
                                                output=p_output / 'task_progress_plots', 
                                                override=args.override_old, 
                                                save_csv=True,
                                                logger=logger,
                                                **options)
        except Exception as e:
            logger.error(f'Error while plotting the multi-task progress metrics!')
            logger.error(f'Traceback: {e}')

    ray.shutdown()

    try:
        exp_dir = Path(analysis.trials[0].logdir).parent
    except KeyError:
        logger.warning('Could not find any trial in Tune analysis!')
        exp_dir = p_output
    logger.info("Visualize results with TensorBoard with e.g. `tensorboard --logdir {}`".format(exp_dir))
    logger.info("+--------------------------------------------+")
    logger.info("|             END OF EXPERIMENT              |")
    logger.info("+--------------------------------------------+")



if __name__ == '__main__':
    try:
        os.chdir(__NEW_CWD__)
        main(parse_args())
    finally:
        os.chdir(__ORIG_CWD__)