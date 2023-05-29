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

import argparse, yaml, logging, csv, tempfile
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
from ray.tune.utils import flatten_dict
from ray.tune.logger import pretty_print, UnifiedLogger

# utilities
from common.util import (setup_logger, make_new_dir, get_experiment_info,
    load_config_and_update, convert_to_tune_config, dict_deep_update, 
    get_slurm_resources, save_model_stats)
from common.custom_callbacks import ModelSummaryCallback
from ray.tune.logger import CSVLogger
# agents
from continual_atari.agents.vanilla_ppo_agent import PPOAgent
from ray.rllib.agents.ppo import PPOTrainer
# from continual_atari.agents.ppo_agent import ContinualPPOTrainer
# atari envs
from envs.ray_env_util import register_envs, get_env_infos
# from envs.continual_env import ContinualEnv
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
    # parser.add_argument('-algo', dest='algorithm', type=str, 
    #     choices=['vanilla_ppo','ewc','pnn','progress_compress'],
    #     help='Selection of the algorithm to use (overrides config value).')
    # parser.add_argument('-num_gpu', default=-2, type=int, 
    #     help='Cuda devices to use: \n'
    #     '\t-2 (default) for automatic handling \n\t-1 for none; \n\t0,1... for specific device\n' 
    #     'It\'s recommended to let `ray` handle the device allocation, but if only a specific Cuda device shall '
    #     'be visible to the program or no GPU shall be used at all, then `ray` will be configured to only '
    #     'propagate that device to the allocator.')
    # parser.add_argument('-num_cpu', default=-2, type=int, 
    #     help='CPU\'s to use: \n'
    #     '\t-2 (default) for automatic handling \n\t-1 for none; \n\t0,1... for specific number of CPUs')
    parser.add_argument('--output', dest='output_path', required=True, type=str, 
        help='Path where the output (= logs, configs, savepoints,...) will be stored. '
        'If the destination does not exist it will be created.')
    parser.add_argument('--override', dest='override_old', action="store_true", default=False,
        help='If specified, output directories with content in them will be overridden '
        'instead of a new directory with a new name created.')
    parser.add_argument("--debug-ray", dest='debug_ray', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']),
        help='If specified, Ray will be run serialized in debug mode (local without parallelization).')
    #TODO: unused right now, add to env_config and use in trainable
    parser.add_argument("--stop-iters", dest='training_iteration', type=int,
        help='Stopping criterion for the agent training. Terminates after `stop-iters` Trainer iterations.')
    parser.add_argument("--stop-timesteps", dest='timesteps_total', type=int,
        help='Stopping criterion for the agent training. Terminates after `stop-timesteps` env steps.')
    parser.add_argument("--stop-reward", dest='episode_reward_mean', type=float,
        help='Stopping criterion for the agent training. Terminates after `stop-reward` reward is achieved.')

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


#functions to override the TorchPolicy class with since they are not implemented!
def export_model(self, export_dir):
    """
    Example:
        >>> trainer = MyTrainer()
        >>> for _ in range(10):
        >>>     trainer.train()
        >>> trainer.export_policy_model("/tmp/")
    """
    os.makedirs(export_dir, exist_ok=True)
    self.model.export_model(export_dir)

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
    from ray.rllib.utils import try_import_torch
    torch, _ = try_import_torch()
    os.makedirs(export_dir, exist_ok=True)

    export_file = os.path.join(export_dir, filename)
    save_dict = {}
    for i, opt in enumerate(self._optimizers):
        save_dict["opt_{}_state_dict".format(i)] = opt.state_dict()
    save_dict["model_state_dict"] = self.model.state_dict()
    
    torch.save(save_dict, export_file)

#NOTE: checkpointing and model export are not implemented for TochPolicy currently (Ray=0.8.6)...
# so a running implementation needs to be added
#class-wide override
# from ray.rllib.policy.torch_policy import TorchPolicy
# TorchPolicy.export_checkpoint = export_checkpoint
# TorchPolicy.export_model = export_model


def trainable_for_PNN(config):
    """The training callable for the Tune run.
    For the continual learning this handles the training on multiple environments
    and checkpointing, as well as logging.
    """
    checkpoints = []
    prev_env_infos = None
    tracker_logdir = Path(tune.get_trial_dir()).parent

    stoppers = config['env_config'].get('stopping_criteria',{})
    if not stoppers or all([v is None for v in stoppers.values()]):
        print('\n')
        print("WARNING: No stopping criteria given, the first task will be trained infinitely!")
        print('\n')

    def logger_creator(config, taskdir, prefix=""):
        """Creates a Unified logger with a default logdir prefix
        containing the agent name and the env id
        """
        if not os.path.exists(taskdir):
            os.makedirs(taskdir)
        logdir = tempfile.mkdtemp(
            prefix=prefix, dir=taskdir)
        return UnifiedLogger(config, logdir, loggers=None)

    task_list = config['env_config']['task_list']

    for task_id, task_name in enumerate(task_list):
        task_config = config.copy()
        #update the env config with task specific settings
        task_env_spec = config['env_config'].get('per_env_config',{}).get(task_name)
        dict_deep_update(task_config['env_config'], task_env_spec)

        #update the model setup info
        #FIXME: this is only for PNN right now!
        task_config['model']['custom_model']['build_columns'] = \
            config['model']['custom_model']['build_columns'][:task_id+1]
        
        task_dir = str(tracker_logdir / "T-{}_{}".format(task_id, task_name))
        make_new_dir(task_dir, override=True)
        dt = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

        trainer = PPOTrainer(config=task_config, 
                             env=task_name, 
                             logger_creator=lambda c: logger_creator(c, task_dir, f"{task_name}_{dt}"))
        progressLogger = CSVLogger(task_config, task_dir)   #TODO: maybe retrieve trial object (somehow) to give to logger when syncing fails

        #NOTE: Monkey patch TorchPolicy since those functions are not implemented in Ray 0.8.6
        def add_checkpointing_to_policy(policy, pol_id, model_fn, ckpt_fn):
            from ray.rllib.policy.torch_policy import TorchPolicy
            policy.export_model = model_fn.__get__(policy, TorchPolicy)
            policy.export_checkpoint = ckpt_fn.__get__(policy, TorchPolicy)
        trainer.workers.foreach_policy(lambda p, pid: 
                                       add_checkpointing_to_policy(p, pid, export_model, export_checkpoint))

        #FIXME: this is maybe not necessary anymore after the PNN initialization update
        # #NOTE: use ONLY objects in worker functions that are serializable! Everything else needs to be given
        # # as static parameter like obs_sp/action_sp here!
        # obs_sp = trainer.workers.local_worker().env.observation_space
        # action_sp = trainer.workers.local_worker().env.action_space
        # def add_pnn_column(policy, pol_id, obs_space, act_space):
        #     policy.model.freeze_columns()
        #     policy.model.next_task(obs_space, act_space)
        #     return
        # if task_id > 0:
        #     assert  len(checkpoints) >= task_id, \
        #         f"Missing checkpoints to load the PNN. Got task ID {task_id} but only {len(checkpoints)} checkpoints."
        #     for i in range(task_id):   #rebuild the PNN structure
        #         osp = prev_env_infos[i]['observation_space']
        #         asp = prev_env_infos[i]['action_space']
        #         try:
        #             trainer.workers.foreach_policy(lambda p, pid: add_pnn_column(p, pid, osp, asp))
        #         except AttributeError as ae:
        #             raise AttributeError("The trainer model has no freeze_columns() or "
        #                 "next_task() function! Make sure to use a suitable policy for "
        #                 "continual learning. \n{}".format(ae))
        #     #re-init with the previous training
        #     trainer.import_policy_model_from_h5(checkpoints[i])

        # #add the next task
        # trainer.workers.foreach_policy(lambda p, pid: add_pnn_column(p, pid, obs_sp, action_sp))

        #log the model summary once per task
        obs = task_config['model']['custom_model']['build_columns'][task_id]['observation_space']
        save_model_stats(trainer.get_policy().model,
                        task_dir, 
                        framework=task_config['framework'],
                        input_shape=obs.shape)

        while True:
            #train on the env
            #each train() step runs:
            #  1. collect [train_batch_size] samples from remote workers (is not always excactly that number)
            #  2. for num_sgd_iter times do:
            #    4. random shuffle all samples (if no RNN state)
            #    3. for [train_batch_size/sgd_minibatch_size] num of mini-batches do: 
            #       5. run SGD on [sgd_minibatch_size] samples
            # => performs [num_sgd_iter * train_batch_size / sgd_minibatch_size] number of SGD steps
            #    on ~[num_sgd_iter * train_batch_size] num of samples (=steps in environment)
            result = trainer.train()
            
            #collect some metrics
            #TODO: this can may be deleted since it writes the exact same thing in a different location
            progressLogger.on_result(result)
            
            # check the stopping conditions
            should_stop = False
            for n in stoppers.keys():
                try:
                    if stoppers[n] is not None and result[n] >= stoppers[n]:
                        should_stop = True
                        print(f"\nStopping criterion [{n} = {stoppers[n]}] reached!\n")
                        break
                except KeyError:
                    print(f"Stopping criterion {n} is not in the training result and cannot be evaluated!")
                    continue
            if should_stop:
                break

        ckpt_name = "model_ckpt_T-{}.h5".format(task_id)
        trainer.export_policy_checkpoint(task_dir, ckpt_name)
        export_path = str(Path(task_dir) / ckpt_name)
        # trainer._save(export_path)  #this saves the whole trainer obj (filters, policies, optimizer) as pickle
        # trainer.workers.local_worker().for_policy(
        #     lambda p: p.model.export_checkpoint(export_path))  #lowlevel, direct model fn access
        #add the last trainer model checkpoint to load for next task
        checkpoints.append(export_path)     #TODO: list may be irrelevant now
        config['model']['custom_model_config'].update(last_checkpoint=export_path)
        progressLogger.flush()
        progressLogger.close()
        
        #save previous env info
        prev_env_infos = trainer.get_policy().model.task_specs
        # saves the whole trainer (__get_state__) with all workers
        state = trainer.save()  #TODO:check if this can be used, since the env and model structure is changed by trainers
        #stop the trainer
        trainer.stop()

    # #save the continual stats
    # flat_Stats = flatten_dict(continual_stats, delimiter="/")
    # f_name = os.path.join(tracker_logdir, "tasks_progress.csv")
    # cont = os.path.exists(f_name)
    # with open(f_name, 'a') as f:
    #     writer = csv.DictWriter(f, flat_Stats.keys())
    #     if not cont:
    #         writer.writeheader()
    #     writer.writerow({k: v for k, v in flat_Stats.items() if k in writer.fieldnames})
    
    #TODO: maybe move this out of the fn, fn is maybe called multiple times by tune
    # export whole serialized pytorch model
    # NOTE: this is project structure sensitive!
    t_names = "-".join(task_list)
    model_save_path = str(tracker_logdir / f"final_model_Ts{task_id}")
    trainer.export_policy_model(model_save_path)




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
    # if not is_empty:
    #     raise IOError("The given output directory is not empty! " 
    #         "Running an new experiment would override old files. "
    #         "Please choose a different location or use the -override flag!")

    #update default config keys and values with commandline args
    dict_args = get_args_dict(args)
    # setup logger
    logger = setup_logger(name=exp_name, log_path=p_output, verbose=True)
    logger.info("+----------------------------------------------------+")
    logger.info("|  STARTING CONTINUAL LEARNING: {:<20} |".format(exp_name))
    logger.info("+----------------------------------------------------+")

    # initialize the ray cluster and connect to it (when the cluster is the machine itself)
    ray.init(
        ignore_reinit_error=False,   #ignore error thrown if ray is initialized multiple times
        log_to_driver=True,     #whether the logs shall be send to the driver
        local_mode=debug_ray,    #whether to run ray in debug mode
        webui_host='localhost',     #host to reach the wb UI of ray from
        lru_evict=True,
        # memory=1*1024*1024*1024,    #given in byte
        # object_store_memory=0.8*1024*1024*1024,
        # driver_object_store_memory=0.5*1024*1024*1024,
        temp_dir=str(Path('~/tmp/ray').expanduser()),      #the root dir for the ray temp files and logs (default: /tmp/ray/)
        )

    user_config = load_config_and_update(args.config)
    ray_config, aux_dict = convert_to_tune_config(user_config, False, logger)

    with open(p_output/'exp_config.yml','w') as f:
        yaml.safe_dump(user_config, f, default_flow_style=False)

    
    #if custom env or own env shall be used
    logger.info('Registering custom environments if needed...')
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
        logger.info("Registering custom model: {}".format(algo))
        if algo == 'pnn':
            #for PNN only the model must be adapted
            trainable = trainable_for_PNN
            custom_model = PNN
            ray_config['model'].update({
                'custom_model': algo,
                # 'custom_action_dist': None
            })
            ModelCatalog.register_custom_model(algo, custom_model)

        else:
            raise ValueError('The selected algorithm [{}] in the config is not implemented!'.format(algo))
        

    #add a callback to get model summary
    #NOTE: not needed since the trainable can call summaries between tasks
    # logger.info('Adding callbacks to Trainer...')
    # ray_config.update(callbacks=ModelSummaryCallback)

    #add the stoping criterias
    stop_updates = {k:getattr(args,k) for k in ray_config['env_config']['stopping_criteria'].keys()
                    if getattr(args,k) is not None}
    dict_deep_update(ray_config['env_config']['stopping_criteria'], stop_updates)

    # get env space infos for each task
    logger.info('Collecting environment infos for each task...')
    env_infos = get_env_infos(ray_config['env_config']['task_list'], ray_config, logger)
    ray_config['model']['custom_model']['build_columns'] = env_infos

    # determine HW assignment from SLURM
    get_slurm_resources(ray_config, logger)

    logger.info('Running Tune with configuration:\n{}'.format(
                get_experiment_info(ray_config, 
                                    trainable=trainable, 
                                    debug=debug_ray,
                                    additional_env_info=env_infos,
                                    **aux_dict)
                ))
    
    
    #register the ContinualEnv
    # register_env("ContinualAtari", lambda config: ContinualEnv(config))

    # run the trainable
    #TODO: add restore flag for continued experiments?
    analysis = tune.run(
        trainable,
        name=exp_name,
        local_dir=str(p_output),
        #for trainable API the resources are not found correctly if not set here ... (each trial will be blocking others)
        resources_per_trial={
            'gpu': ray_config['num_gpus'], 
            'cpu': 1, 
            'extra_cpu': ray_config['num_workers']}, 
        # checkpoint_freq=10,       #FIXME: this is popbably not applicable with custom trainables
        checkpoint_at_end=False,    #NOTE: if True, this raises error bc it calls the Trainable.save fn (not implemented) for some reason
        # keep_checkpoints_num=10,
        # checkpoint_score_attr="min-validation_loss",    #metric to determine the best checkpoints
        # resources_per_trial={"cpu": 1, "gpu": 0},  #default
        config=ray_config,
        # search_alg=BasicVariantGenerator(),   #default
        # scheduler=FIFOScheduler(),       #default
        # with_server=False,     Starts a background Tune server. Needed for using the Client API
        # server_port=,         Port number for launching TuneServer.
        # resume=False | "LOCAL" / True / "PROMPT",  #resume experiment from checkpoint_dir (=local_dir/name); PROMT requires CLI input
        verbose=2   #0 = silent, 1 = only status updates, 2 = status and trial results
    )

    #return the row for each trial experiment with best metric
    try:
        max_rew_df = analysis.dataframe(metric='episode_reward_mean', mode='max')
        logger.info('Best MEAN REWARD results:\n{}'.format(max_rew_df))
        max_rew_config = analysis.get_best_config(metric='episode_reward_mean', mode='max')
        logger.info('Best HP setting:\n{}'.format(pretty_print(max_rew_config)))
        best_dir = analysis.get_best_logdir(metric='episode_reward_mean', mode='max')
        logger.info('Location of best Trial: {}'.format(best_dir))
    except AttributeError as ae:
        logger.error('Could not print trial info of best result!\nError: {}'.format(ae))

    # TBC: append logging / metrics; post-process metrics; etc
    # accumulate all the continual trained agents
    # global_analysis = tune.Analysis(str(p_output))

    ray.shutdown()

    try:
        exp_dir = Path(analysis.trials[0].logdir).parent
    except KeyError:
        logger.warning('Could not find any trial in Tune analysis!')
        exp_dir = p_output
    logger.info("Visualize results with TensorBoard with `tensorboard --logdir {}`".format(exp_dir))
    logger.info("+--------------------------------------------+")
    logger.info("|             END OF EXPERIMENT              |")
    logger.info("+--------------------------------------------+")



if __name__ == '__main__':
    try:
        os.chdir(__NEW_CWD__)
        main(parse_args())
    finally:
        os.chdir(__ORIG_CWD__)