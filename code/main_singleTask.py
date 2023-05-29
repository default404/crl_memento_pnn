
###############################################################
# This trains an agent on a single RL task (env) using either
# a standart Rllib implementation or a custom model/agent.
# The used Loss criterion is PPO.
###############################################################

import os, sys, inspect
__MAINDIR__ = os.path.dirname(os.path.realpath(inspect.getfile(lambda: None)))
print('Main dir:', __MAINDIR__)
if not __MAINDIR__ in sys.path: #insert the project dir in the sys path to find modules
    sys.path.insert(0, __MAINDIR__)
print('System Path:\n',sys.path)

import argparse, yaml, pickle
from datetime import datetime
from pathlib import Path

import ray
from ray import tune
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog

# utilities
from common.util import (setup_logger, make_new_dir, get_experiment_info,
    load_config_and_update, convert_to_tune_config, dict_deep_update,
    get_slurm_resources, save_model_stats, remove_gridsearch)
from common.custom_callbacks import UnifiedCallbacks
import common.exp_analysis as expAnalysis
# agents
from ray.rllib.agents.ppo import PPOTrainer
# from continual_atari.agents.ppo_agent import ContinualPPOTrainer
# continual policies / models
from continual_atari.agents.policies.pnn.pnn_model import PNN
#custom envs
from envs.ray_env_util import register_envs, get_env_infos

# handle the cwd for script execution
__ORIG_CWD__ = os.getcwd()
__NEW_CWD__ = __MAINDIR__


#The path to the default config file (used if no config is given at runtime)
_DEFAULT_CONFIG = Path(__MAINDIR__) / "single_task_atari/single_task_config.yml"

# input args
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument('--output', type=str, default='~/no_backup/d1346/tune_out/single_task')
parser.add_argument('--config', type=str, default=_DEFAULT_CONFIG)
parser.add_argument("--debug-ray", default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
parser.add_argument("--override", dest='override_old', action="store_true", default=False)
parser.add_argument("--stop-iters", dest='training_iteration', type=int,
    help='Stopping criterion for the agent training. Terminates after `stop-iters` Trainer iterations.')
parser.add_argument("--stop-timesteps", dest='timesteps_total', type=int,
    help='Stopping criterion for the agent training. Terminates after `stop-timesteps` env steps.')
parser.add_argument("--stop-reward", dest='episode_reward_mean', type=float,
    help='Stopping criterion for the agent training. Terminates after `stop-reward` reward is achieved.')
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


def main(args):

    date = datetime.today().strftime('%Y-%m-%d')
    p_output = Path(args.output).expanduser()
    exp_name = "{}_{}".format(p_output.name, date)
    debug_ray = args.debug_ray

    p_output = make_new_dir(p_output, override=args.override_old)

    # setup logger
    logger = setup_logger(name=exp_name, log_path=p_output, verbose=True)
    logger.info("+-------------------------------------------------------------+")
    logger.info("|  START SINGLE TASK TRAINING: {:^30} |".format(exp_name))
    logger.info("+-------------------------------------------------------------+")

    logger.info("Generating unified config dict for Ray...")
    # normal way to load the config yaml of the tool 
    # dict_args = get_args_dict(args)
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
    
    logger.info("Initializing Ray...")
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
    
    logger.info('Registering custom environments if needed...')
    register_envs(ray_config, logger=logger)

    # get custom model options if applicable
    if ray_config['model'].get('custom_model_config', {}):
        logger.info("Registering custom model...")
        if aux_dict["algorithm"] == 'pnn':
            custom_model = PNN
            ray_config['model'].update({"custom_model": aux_dict["algorithm"],
                                # "custom_model_config": ray_config["model"]["custom_model_config"]
                                })
            ModelCatalog.register_custom_model(aux_dict["algorithm"], custom_model)
        else:
            raise ValueError('Unknown custom alogrithm to use!')
    else: 
        #the standart Rllib model
        custom_model = None
    
    # determine HW assignment from SLURM
    get_slurm_resources(ray_config, logger)

    #add a callback to get model summary
    logger.info('Adding callbacks to Trainer...')
    ray_config.update(callbacks=UnifiedCallbacks)

    # get env space infos for each task
    logger.info('Collecting environment infos for each task...')
    env_infos = get_env_infos(ray_config['env_config']['task_list'], ray_config, logger)
    
    logger.info('Running Tune with configuration:\n{}'.format(
                get_experiment_info(ray_config, 
                                    trainable=args.run,
                                    debug=debug_ray, 
                                    stop=stoppers,
                                    additional_env_info=env_infos,
                                    **aux_dict)
                ))
    
    if not stoppers:
        # task will be trained indefinitely and stopped manually
        exp_name = f'inf__{exp_name}'

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
        ray_config['model']['custom_model_config'].update(build_columns=env_infos)
    else:
        # remove HP gridsearch from custom model which has no effect for 
        # the first task and greatly increases the opt space! 
        no_hpOpt_model_config = remove_gridsearch(
            ray_config['model']['custom_model_config']['alpha_value']
        )
        ray_config['model']['custom_model_config']['alpha_value'] = no_hpOpt_model_config

    analysis = tune.run(
        args.run,
        name=exp_name,
        local_dir=str(p_output),
        # checkpoint_freq=10,
        checkpoint_at_end=True,     #default
        # keep_checkpoints_num=10,
        # checkpoint_score_attr="min-validation_loss",    #metric to determine the best checkpoints
        stop=stoppers,
        # resources_per_trial={"cpu": 1, "gpu": 0},  #default
        num_samples=1,      #number of trials started for the experiment, each trial requires the specified resources!
        config=ray_config,
        # with_server=False,     Starts a background Tune server. Needed for using the Client API
        # server_port=,         Port number for launching TuneServer.
        # resume=False | "LOCAL" / True / "PROMPT",  #resume experiment from checkpoint_dir (=local_dir/name); PROMT requires CLI input
        verbose=2   #0 = silent, 1 = only status updates, 2 = status and trial results
    )
    analysis_list = [analysis]

    best_dir = None
    try:
        best_dir = analysis.get_best_logdir(metric='episode_reward_mean', mode='max')
        logger.info('Location of best Trial: {}'.format(best_dir))
    except Exception as e:
        logger.error('Could find the best trial directory for '\
                        'task {}!\nError: {}'.format(ray_config['env'], e))

    if first_task_ckpt:
        exp_state_file = expAnalysis.EXPERIMENT_STATE_FILE
        try:
            exp_state_path = list(Path(first_task_ckpt).parent.parent.parent.glob(exp_state_file))[0]
        except:
            logger.warning(f'Couldn\'t fetch the experiment state ({exp_state_file}) from the inital '
                            'checkpointed experiment for consecutive plotting')
        else:
            analysis_list.insert(0, tune.ExperimentAnalysis(str(exp_state_path)))

    logger.info('')
    logger.info(''.join(['=']*40))
    logger.info('Starting plotter functions for the experiment:')
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
    logger.info(f'Creating averaging plots for the Tune experiment...')
    options = {'color':'xkcd:blood red', 'height':4, 'aspect':3, 'save':True}
    try:
        expAnalysis.run_exp_average_plotter(analysis_list, 
                                            output=p_output / 'avg_tuneExp_plots', 
                                            override=args.override_old, 
                                            logger=logger, 
                                            **options)
    except Exception as e:
        logger.error(f'Error while running the averaging plotter for the experiment!')
        logger.error(f'Traceback: {e}')
    logger.info(f'Creating trial summary plots for the Tune experiment...')
    try:
        options.update(legend=False)
        expAnalysis.run_task_progress_plotter(analysis, 
                                              task_perf_dict=None,
                                              trial_selection='all',
                                              output=p_output / 'summary_tuneExp_plots', 
                                              override=args.override_old, 
                                              save_csv=False,
                                              logger=logger,
                                              **options)
    except Exception as e:
        logger.error(f'Error while running the trial summary plotter for the experiment!')
        logger.error(f'Traceback: {e}')
    
    # create progress plots for the whole multi-task experiment
    if len(analysis_list) > 1:
        logger.info('')
        logger.info(f'Creating progress plots for all tasks of the experiment...')
        options = {'height':4, 'aspect':3, 'save':True}
        try:
            expAnalysis.run_task_progress_plotter(analysis_list, 
                                                task_perf_dict=None,
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
        
    logger.info('')
    logger.info("Visualize results with TensorBoard with e.g. `tensorboard --logdir {}`".format(exp_dir))
    logger.info("+--------------------------------------------+")
    logger.info("|             END OF EXPERIMENT              |")
    logger.info("+--------------------------------------------+")



if __name__ == '__main__':
    try:
        os.chdir(__NEW_CWD__)
        main(parser.parse_args())
    finally:
        os.chdir(__ORIG_CWD__)