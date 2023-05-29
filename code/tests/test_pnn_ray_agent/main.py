
import os, sys, inspect
__MAINDIR__ = os.path.dirname(os.path.realpath(inspect.getfile(lambda: None)))
__PRJ_DIR__ = os.path.dirname(os.path.dirname(__MAINDIR__))
if not __PRJ_DIR__ in sys.path: #insert the project dir in the sys path to find modules
    sys.path.insert(0, __PRJ_DIR__)

import argparse, yaml, logging
from datetime import datetime
from pathlib import Path
import numpy as np

import ray
from ray import tune
from ray.tune import grid_search
from ray.rllib.models import ModelCatalog
# from ray.rllib.agents.ppo import PPOTrainer

# utilities
from common.util import (setup_logger, make_or_clean_dir, 
    load_config_and_update, convert_to_tune_config, dict_deep_update)
# agents
from continual_atari.agents.vanilla_ppo_agent import PPOAgent
# from continual_atari.agents.ppo_agent import ContinualPPOTrainer
# continual policies / models
from continual_atari.agents.policies.pnn.pnn_model import PNN



# handle the cwd for script execution
__ORIG_CWD__ = os.getcwd()
__NEW_CWD__ = __MAINDIR__


#The path to the default config file (used if no config is given at runtime)
_DEFAULT_CONFIG = Path(__ORIG_CWD__) / "continual_atari/configs/default_config.yml"

# input args
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument('--output', type=str, default='/home/dennis/tune_out/pnn_test2')
parser.add_argument("--debug-ray", action="store_true", default=True)
parser.add_argument("--override", dest='override_old', action="store_true", default=True)
parser.add_argument("--env", type=str, default="Pong-v0")
parser.add_argument("--stop-iters", type=int, default=50)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=10.0)
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
    p_output = Path(args.output)
    exp_name = "{}_{}".format(p_output.name, date)
    debug_ray = args.debug_ray

    is_empty = make_or_clean_dir(p_output, check_only=(not args.override_old))
    if not is_empty:
        raise IOError("The given output directory is not empty! " 
            "Running an new experiment would override old files. "
            "Please choose a different location or use the -override flag!")

    # setup logger
    logger = setup_logger(name=exp_name, log_path=p_output, verbose=True)
    logger.info("+--------------------------------------------+")
    logger.info("|  STARTING EXPERIMENT: {:<20} |".format(exp_name))
    logger.info("+--------------------------------------------+")

    logger.info("Generating unified config dict for Ray...")
    # normal way to load the config yaml of the tool 
    # dict_args = get_args_dict(args)
    tune_config = load_config_and_update(_DEFAULT_CONFIG,None,{'algorithm':'pnn'})
    tune_config, unused = convert_to_tune_config(tune_config, False, logger)
    # let the PNN create the first column instantly to be able to run in tune
    tune_config["model"]["custom_model_config"].update({
        'init_on_creation': True,
        'output_basedir': str(p_output),
    })

    # only use the special model parameters for the test config 
    config = {
        "env": args.env,
        "env_config": {
            
        },
        "model": {
            "custom_model": "PNN",
            'custom_model_config': tune_config["model"]["custom_model_config"]
        },
        "lr": grid_search([1e-2, 1e-4]),  # try different lrs
        "num_workers": 1,  # parallelism
        "framework": '"torch',
        "log_level": "DEBUG",
        "monitor": True,
        #tuned example params
        "no_done_at_end": False,
        "normalize_actions": False,
        "clip_rewards": True,
        "clip_actions": True,
        "preprocessor_pref": "deepmind",
        "use_critic": True,
        "use_gae": True,
        "lambda": 0.95,
        "kl_coeff": 0.5,
        "vf_share_layers": True,
        "vf_loss_coeff": 1.0,
        "entropy_coeff": 0.01,
        "entropy_coeff_schedule": None,
        "clip_param": 0.1,
        "vf_clip_param": 10.0,
        "grad_clip": None,
        "kl_target": 0.01,
    }
    # config = dict_deep_update(config, override)

    #stoping criteria
    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.timesteps_total,
        "episode_reward_mean": args.episode_reward_mean,
    }

    logger.info("Initializing Ray...")
    # initialize the ray cluster and connect to it (when the cluster is the machine itself)
    ray.init(
        ignore_reinit_error=False,   #ignore error thrown if ray is initialized multiple times
        log_to_driver=True,     #whether the logs shall be send to the driver
        local_mode=debug_ray,    #whether to run ray in debug mode
        webui_host='localhost',     #host to reach the wb UI of ray from
        # temp_dir=str(p_output /'tmp'),      #the root dir for the ray temp files and logs (default: /tmp/ray/)
        )

    #register custom model
    ModelCatalog.register_custom_model("PNN", PNN)

    analysis = tune.run(
        args.run,
        name=exp_name,
        local_dir=str(p_output),
        checkpoint_freq=10,
        checkpoint_at_end=False,     #default
        # keep_checkpoints_num=10,
        # checkpoint_score_attr="min-validation_loss",    #metric to determine the best checkpoints
        stop=stop,
        # resources_per_trial={"cpu": 1, "gpu": 0},  #default
        config=config,
        # with_server=False,     Starts a background Tune server. Needed for using the Client API
        # server_port=,         Port number for launching TuneServer.
        # resume=False | "LOCAL" / True / "PROMPT",  #resume experiment from checkpoint_dir (=local_dir/name); PROMT requires CLI input
        verbose=2   #0 = silent, 1 = only status updates, 2 = status and trial results
    )




    # TBC: append logging / metrics; post-process metrics; etc
    # accumulate all the continual trained agents
    global_analysis = tune.Analysis(str(p_output))
    #do stuff
    if analysis.trials[0].last_result["episode_reward_mean"] < args.episode_reward_mean:
        logger.info("`stop-reward` of {} not reached!".format(args.episode_reward_mean))
    logger.info("ok")

    ray.shutdown()

    logger.info("+--------------------------------------------+")
    logger.info("|             END OF EXPERIMENT              |")
    logger.info("+--------------------------------------------+")



if __name__ == '__main__':
    try:
        os.chdir(__NEW_CWD__)
        main(parser.parse_args())
    finally:
        os.chdir(__ORIG_CWD__)