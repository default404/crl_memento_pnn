
###############################################################
# This trains an agent on a single RL task (env) using either
# a standart Rllib implementation or a custom model/agent.
# The used Loss criterion is PPO.
###############################################################

import os, sys, inspect
__MEMENTO_DIR__ = os.path.dirname(os.path.realpath(inspect.getfile(lambda: None)))
__MAINDIR__ = os.path.dirname(__MEMENTO_DIR__)
print('Main dir:', __MAINDIR__)
if not __MAINDIR__ in sys.path: #insert the project dir in the sys path to find modules
    sys.path.insert(0, __MAINDIR__)
print('System Path:\n',sys.path)

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import pickle

#data science & DL packages
import numpy as np
import pandas as pd
import torch
# from torch.autograd import Variable
import gym
import ray
from ray.rllib.models import ModelCatalog
# from ray.rllib.models.model import flatten

# project utilities
from common.util import (setup_logger, make_new_dir, load_yml_config, 
    convert_to_tune_config, dict_deep_update)
# continual policies / models
from continual_atari.agents.policies.pnn.pnn_model import PNN
#custom envs
from envs.ray_env_util import register_envs
#eval plotting
from memento_experiment.plot_eval_csv import main as eval_plotter, plot_eval_metrics
#helpers from state buffer creation
from memento_experiment.create_state_buffer import (convertToNonTorchTensor,
    convertToTorchTensor, fake_ray_env_creation)


# handle the cwd for script execution
__ORIG_CWD__ = os.getcwd()
__NEW_CWD__ = __MAINDIR__

# PATH AND FILE NAME CONSTANTS
AGENT_BUFFER_NAME = 'agent_observations.pkl'    #dict of episode id to observations list mapping
STATE_BUFFER_NAME = 'statebuffer_observations.pkl'  #normal list of observations
from memento_experiment import MEMENTO_BUFFER, EVAL_CSV_NAME, MEMENTO_CSV_NAME
# MEMENTO_BUFFER = 'memento.buffer'
# EVAL_CSV_NAME = 'evaluation.csv'
# MEMENTO_CSV_NAME = 'memento_info.csv'

#The default logger
_logger = setup_logger(name=__name__,verbose=True)


def total_obs(list_dict):
    return sum(len(l) for l in list_dict.values())

def get_single_obs_image(obs):
    obs = np.asarray(obs)
    if len(obs.shape) == 3:
        if obs.shape[-1] == 1:       #assume greyscale
            return obs
        elif obs.shape[-1] == 3:     #assume RGB image
            return obs
        else:                               #assume some stacked greyscale
            return obs[:,:,:1]
    elif len(obs.shape) == 2:
        return np.expand_dims(obs, -1)
    else:
        raise ValueError('Unknown observation type to get single image from!')


#################################################################

def collect_agent_obs(args):
    """Main eval function to collect some observations from an agent. 
    """
    date = datetime.today().strftime('%Y-%m-%d')

    output_dir = Path(args.output).expanduser()
    exp_name = output_dir.name
    output_dir = make_new_dir(output_dir, override=args.override_old)

    # setup logger
    logger = setup_logger(name=exp_name, log_path=output_dir/'eval_sb.log', verbose=True)
    logger.info("+---------------------------------------------------------------+")
    logger.info("|  START MODEL EVAL & CREATE STATE BUFFER: {:^20} |".format(exp_name))
    logger.info("+---------------------------------------------------------------+")

    #update the eval config with CLI args
    eval_config = { 'checkpoint': args.checkpoint,
                    'env_spec': args.env_spec,
                    'exp_config': args.exp_config,
                  }

    logger.info("Loading training config from experiment dir...")
    exp_config_file = eval_config.get('exp_config')
    if not exp_config_file or not Path(exp_config_file).is_file():
        raise FileNotFoundError('Could not find the config for the experiment to be evaluated! '
                        'Without it the model and env cannot be reconstructed the same way as in the training.')
    exp_config = load_yml_config(exp_config_file)
    ray_config, aux_dict = convert_to_tune_config(exp_config, False, logger)

    # update config fields for evaluation
    ray_config['model']['custom_model_config'].update(in_training=False)
    checkpoint = eval_config.get('checkpoint')
    if not checkpoint or not os.path.isfile(checkpoint):
        logger.error(f'WARNING: given checkpoint file to load the model from is invalid ({checkpoint})! '
                        'A valid file path must be given either in the eval config or on the command line.')
        raise ValueError('Invalid checkpoint file!')
    ray_config['model']['custom_model_config'].update(checkpoint=checkpoint)
        
    env_spec = eval_config['env_spec'] if eval_config['env_spec'] and os.path.isfile(eval_config['env_spec']) \
                else ray_config['model']['custom_model_config']['build_columns']
    ray_config['model']['custom_model_config'].update(build_columns=env_spec)
    
    # create all the envs to evaluate
    logger.info('Reconstructing Ray envs from training config...')
    ray.init(configure_logging=False, local_mode=True)  #a running Ray instance is needed to read from the registry
    logger.info('Registering custom environments if needed...')
    register_envs(ray_config, logger=logger)
    eval_last_env_only = True
    #this fakes the rllib env creation but disables the EpisodicLife from the deepmind wrapper by default
    envs = fake_ray_env_creation(ray_config, monitordir=output_dir, 
                                 last_only=eval_last_env_only, 
                                 disable_episodicLife=False, logger=logger)
    ray.shutdown()

    # create the PNN model
    logger.info('Creating agent model and action distribution class...')
    logger.info(f'Using checkpoint: {checkpoint}')
    logger.info(f'Using env spec: {env_spec}')
    model = PNN(envs[-1]['env'].observation_space,
                envs[-1]['env'].action_space,
                num_outputs=-1,
                model_config=ray_config['model'], 
                name='PNN', 
                **ray_config['model']['custom_model_config'])
    #device allocation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f'Assigning model to device: {device}')
    model = model.to(device)
    
    # constant defs
    collect_num_obs = args.num_obs  #number of obsaervations to collect
    explore = args.explore     #define exploring behavior for action sampling from model logit distribution
    epsilon = args.epsilon      #probability of exploration
    render_env = args.render_env   #shows the agent during evaluation
    eval_df_file = output_dir / EVAL_CSV_NAME
    logger.info(f'Using flags: \n\tExploration: {explore}, epsilon: {epsilon} '
                f'\n\tStallment detection: {args.use_stall_detection} '
                f'\n\tEnv rendering: {render_env}')

    logger.info('\n########################################################################')
    logger.info('Starting evaluation and state buffer creation...\n')
    t_start_eval = datetime.now().replace(microsecond=0)
    ### for each task / env
    for env_dict in envs:
        tid = env_dict['task_id']
        env_id = env_dict['env_id']
        env = env_dict['env']

        logger.info(f'Running evaluation for task {tid}: {env_id} ({str(env)})')

        observations = defaultdict(list)   #the collected observations per env
        stat_dict = defaultdict(list)

        if len(envs) == model.columns:
            if eval_last_env_only:
                #the loaded PNN must automatically have the same tid as the created env
                assert model._cur_task_id == tid
            else:
                model.set_task_id(tid)

        env_dir = output_dir / f'{tid}_{env_id}'
        if not env_dir.exists():
            os.makedirs(str(env_dir))
        curr_obs_buffer = env_dir / AGENT_BUFFER_NAME

        dist_class, _ = ModelCatalog.get_action_dist(
                    env.action_space, ray_config["model"], framework="torch")

        input_dict = {"obs": None, "is_training": False}
        ### for each episode
        t_start_env = datetime.now().replace(microsecond=0)
        curr_ep = 0
        observations[curr_ep]
        logger.info(f'Running evaluation until collecting {collect_num_obs} observations...')
        while total_obs(observations) < collect_num_obs:
            logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>')
            logger.info('Running episode: {:>4d}'.format(curr_ep))

            t_start_ep = datetime.now().replace(microsecond=0)
            obs = env.reset()
            try:
                lifes = env.unwrapped.ale.lives()
            except:
                lifes = None
            done = False
            step = 0    #timestep t
            reward = 0  #reward from env step
            if hasattr(env, 'reward_offset'):
                R_t = getattr(env, 'reward_offset')
            else:
                R_t = 0     #return of timestep t
            

            observations[curr_ep].append(get_single_obs_image(obs))

            stat_dict['Task'].append(tid)
            stat_dict['Environment'].append(env_id)
            stat_dict['Timesteps'].append(step)
            stat_dict['Episode'].append(curr_ep)
            stat_dict['Step Rewards'].append(reward)
            stat_dict['Episode Return'].append(R_t)
            stat_dict['ALE.Lifes'].append(lifes)

            fifo_stall = []  #FIFO reward queue for stallment detection
            tried_reset = False  #control flag for fire reset after life loss 
            old_Rt = R_t
            t_infer = []
            t_step = []
            while not done:         # for each env step
                if render_env:
                    env.render()
                ### Sample Action ###
                with torch.no_grad():
                    t_start_infer = datetime.now()   #track inference & sample time
                    if explore and np.random.random() <= epsilon:   # e-greedy sampling
                        #NOTE: different exploration behavior could be introduced here
                        # see https://github.com/ray-project/ray/blob/releases/0.8.6/rllib/policy/torch_policy.py#L149
                        logits = torch.log(torch.Tensor([[1/model.num_outputs]*model.num_outputs])) #logits of equal prob
                        action_dist = dist_class(logits, model)
                        action = action_dist.sample()
                        logp = action_dist.sampled_action_logp()
                    else:   #deterministic arg-max sampling
                        batched_obs = np.expand_dims(obs, axis=0)   #make a pseudo batch of the single observation
                        obs_tensor = convertToTorchTensor(batched_obs, device=device)
                        input_dict["obs"] = obs_tensor
                        logits,_ = model(input_dict)    #generate logits of the model fwd pass
                        action_dist = dist_class(logits, model)    #gen distribution over actions from specific dist type and model output
                        action = action_dist.deterministic_sample()
                        logp = torch.zeros((action.size()[0], ), dtype=torch.float32)
                    act = convertToNonTorchTensor(action)
                    t_infer.append(datetime.now()-t_start_infer)

                ### Step Environment ###
                t_start_step = datetime.now()   #track env step time
                if args.use_stall_detection and \
                        len(fifo_stall)>=1000 and all(e == R_t for e in fifo_stall):
                    if 'FIRE' in env.unwrapped.get_action_meanings()[1]:
                        if not old_Rt == R_t:
                            tried_reset = False
                        if not tried_reset:
                            logger.debug('Env / agent seems to be stuck (1000 steps w/o reward change). Trying fire reset')
                            fire_act = env.unwrapped.get_action_meanings().index('FIRE')
                            obs, reward, done, info = env.step(fire_act)
                            tried_reset = True
                            old_Rt = R_t
                            fifo_stall = fifo_stall[-500:]   #remove 500 elements from queue
                        else:
                            logger.debug('After trying fire reset, the env / agent is still stuck. Terminating episode')
                            done = True
                    else:
                        logger.debug('Env / agent seems to be stuck (1000 steps w/o reward change) and no reset action known. Terminating episode')
                        done = True
                else:
                    #NOTE: some wrappers mask the true ending or terminate ealier
                    obs, reward, done, info = env.step(act)
                
                observations[curr_ep].append(get_single_obs_image(obs))

                t_step.append(datetime.now()-t_start_step)
                step += 1
                R_t += reward
                if args.use_stall_detection: 
                    fifo_stall.append(R_t)
                    if len(fifo_stall) > 1000: fifo_stall = fifo_stall[-1000:]

                if lifes is not None:
                    _lifes = env.unwrapped.ale.lives()
                    trigger_info = (lifes - _lifes)!=0
                    lifes = _lifes
                else:
                    trigger_info = False
                if step % 500 == 0 or trigger_info:
                    logger.debug('\tEnv step: {:^7d} :: Current R_t: {:<8.2f} :: Lifes: {}'.format(step, R_t, 'NaN' if lifes is None else lifes))

                stat_dict['Task'].append(tid)
                stat_dict['Environment'].append(env_id)
                stat_dict['Timesteps'].append(step)
                stat_dict['Episode'].append(curr_ep)
                stat_dict['Step Rewards'].append(reward)
                stat_dict['Episode Return'].append(R_t)
                stat_dict['ALE.Lifes'].append(lifes)

                #opt out when obs objective is reached
                if total_obs(observations) >= collect_num_obs:
                    logger.info(f'\nNumber of observations to collect ({collect_num_obs}) is reached!')
                    break

                #emergency exit for broken done's
                if step > 1e5:  #the episode should have been finished long ago!
                    logger.warning('\nThe current episode never finished (no DONE return)! Terminating manually...\n')
                    break
            
            curr_ep += 1
            if render_env:
                env.close()

            logger.debug('Episode finished after {} steps ({})'.format(step, datetime.now().replace(microsecond=0)-t_start_ep))
            logger.debug('Episode mean model inference & action sample time: {}'.format(np.mean(t_infer)))
            logger.debug('Episode mean environment step time: {}'.format(np.mean(t_step)))

            if (curr_ep+1) % 10 == 0:
                # frequent checkpointing to prevent data loss if job takes too long
                logger.debug('Writing temporary checkpoint...')
                with open(curr_obs_buffer, 'wb') as f:
                    pickle.dump(observations, f)
                pd.DataFrame.from_dict(stat_dict).to_csv(str(eval_df_file), index=False, na_rep='nan')
        
        logger.info('<<<<<<<<<<<<<<<<<<<<<<<<<<')
        logger.info('Finished running the env: {}. Elapsed time: {}'.format(env_id, datetime.now().replace(microsecond=0)-t_start_env))
        with open(curr_obs_buffer, 'wb') as f:
            pickle.dump(observations, f)

    #final csv saves
    pd.DataFrame.from_dict(stat_dict).to_csv(str(eval_df_file), index=False, na_rep='nan')

    logger.info('Generating plots for the evaluation metrics...')
    eval_plotter(output_dir, logger)

    logger.info('')
    logger.info("+-----------------------------+")
    logger.info("|     FINISHED MODEL EVAL     |")
    logger.info("+-----------------------------+")
    logger.info('Execution time: {}'.format(datetime.now().replace(microsecond=0)-t_start_eval))


def collect_statebuffer_obs(args):
    """Main eval function to collect observations from a state buffer. 
    """
    date = datetime.today().strftime('%Y-%m-%d')

    output_dir = Path(args.output).expanduser()
    exp_name = output_dir.name
    output_dir = make_new_dir(output_dir, override=args.override_old)

    # setup logger
    logger = setup_logger(name=exp_name, log_path=output_dir/'eval_sb.log', verbose=True)
    logger.info("+---------------------------------------------------------------+")
    logger.info("|  START MODEL EVAL & CREATE STATE BUFFER: {:^20} |".format(exp_name))
    logger.info("+---------------------------------------------------------------+")

    logger.info("Loading training config from experiment dir...")
    exp_config_file = args.exp_config
    if not exp_config_file or not Path(exp_config_file).is_file():
        raise FileNotFoundError('Could not find the config for the experiment to be evaluated! '
                        'Without it the model and env cannot be reconstructed the same way as in the training.')
    exp_config = load_yml_config(exp_config_file)
    ray_config, aux_dict = convert_to_tune_config(exp_config, False, logger)

    mem_buffer = args.buffer_path
    if not mem_buffer or not Path(mem_buffer).is_file():
        raise FileNotFoundError('Invalid path to the state buffer for the Memento env! '
                        'Without it the env cannot be initialized with the states to retrieve.')
    ray_config['env_config']['use_memento_env'] = True
    ray_config['env_config']['memento_state_buffer_path'] = mem_buffer
    
    # create all the envs to evaluate
    logger.info('Reconstructing Ray envs from training config...')
    ray.init(configure_logging=False, local_mode=True)  #a running Ray instance is needed to read from the registry
    logger.info('Registering custom environments if needed...')
    register_envs(ray_config, logger=logger)
    eval_last_env_only = True
    ray_config["monitor"] = False   #disable monitor to be able to reset the env without being done
    #this fakes the rllib env creation but disables the EpisodicLife from the deepmind wrapper by default
    envs = fake_ray_env_creation(ray_config, monitordir=None, 
                                 last_only=eval_last_env_only, 
                                 disable_episodicLife=True, logger=logger)
    ray.shutdown()
    
    # constant defs
    collect_num_obs = args.num_obs  #number of obsaervations to collect

    logger.info('\n########################################################################')
    logger.info('Starting evaluation and state buffer creation...\n')
    t_start_eval = datetime.now().replace(microsecond=0)
    ### for each task / env
    for env_dict in envs:
        tid = env_dict['task_id']
        env_id = env_dict['env_id']
        env = env_dict['env']
        logger.info(f'Running evaluation for task {tid}: {env_id} ({str(env)})')
        if "NOOP" not in env.unwrapped.get_action_meanings():
            logger.warning('The env does not seem to have a NOOP action! A random action will be applied instead.')
            use_action = env.action_space.sample()
        else:
            use_action = env.unwrapped.get_action_meanings().index('NOOP')

        observations = []   #the collected observations per env

        env_dir = output_dir / f'{tid}_{env_id}'
        if not env_dir.exists():
            os.makedirs(str(env_dir))
        curr_obs_buffer = env_dir / STATE_BUFFER_NAME

        ### for each episode
        t_start_env = datetime.now().replace(microsecond=0)
        update_time = t_start_env
        curr_ep = 0
        logger.info(f'Running evaluation until collecting {collect_num_obs} observations...')
        while len(observations) < collect_num_obs:
            obs = env.reset()
            obs, _, _, _ = env.step(use_action)
            observations.append(get_single_obs_image(obs))
            curr_ep += 1

            if datetime.now() - update_time > timedelta(seconds=30):
                update_time = datetime.now().replace(microsecond=0)
                running_since = update_time - t_start_env
                logger.info(f'({running_since}) Collected {len(observations)} observations so far')

            if (curr_ep+1) % 1000 == 0:
                # frequent checkpointing to prevent data loss if job takes too long
                logger.debug('Writing temporary checkpoint...')
                with open(curr_obs_buffer, 'wb') as f:
                    pickle.dump(observations, f)
        
        logger.info('<<<<<<<<<<<<<<<<<<<<<<<<<<')
        logger.info('Finished running the env: {}. Elapsed time: {}'.format(env_id, datetime.now().replace(microsecond=0)-t_start_env))
        with open(curr_obs_buffer, 'wb') as f:
            pickle.dump(observations, f)

    logger.info('')
    logger.info("+-----------------------------+")
    logger.info("|     FINISHED MODEL EVAL     |")
    logger.info("+-----------------------------+")
    logger.info('Execution time: {}'.format(datetime.now().replace(microsecond=0)-t_start_eval))

#TODO: 
# write a visu notebook for iterating though the obs list to select the images to plot

# input args
parser = argparse.ArgumentParser()
parser.add_argument('--collect-from', '-from', dest='collect_from', type=str, 
    default='agent', choices=['agent', 'statebuffer'])
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--num-obs', dest='num_obs', type=int, required=True,
    help='Number of observations to collect. The model will run and restart the env if '
    'required, until this number of steps is reached.')
parser.add_argument("--exp_config", dest='exp_config', type=str, default=None,
    help='Path to the experiment config that was used to train the loaded agent. If not given '
         'it is tried to be loaded from the experiment dir.')
parser.add_argument("--buffer-path", dest='buffer_path', type=str, default=None,
    help='Path to the state buffer to load into the Memento env if --collect-from=statebuffer.')        
parser.add_argument("--checkpoint", dest='checkpoint', type=str, default=None,
    help='Checkpoint to init the model with.')
parser.add_argument("--env_spec", dest='env_spec', type=str, default=None,
    help='Path to the environment specification dump that contains the env obs and action spaces.')
parser.add_argument("--use_exploration", dest='explore', action="store_true", default=False,
    help='Whether to use epsilon-greedy exploration for action sampling')
parser.add_argument("--epsilon", dest='epsilon', type=float, default=0.001,
    help='Probability for epsilon-greedy exploration')
parser.add_argument("--use_stall_detection", dest='use_stall_detection', action="store_true", default=False,
    help='Whether to use stallment detection in the environment. This monitors the reward changes '
    'over the last 1k env steps and tries a reset action (e.g. FIRE) if the agent is stuck. If this '
    'doesnt work, the episode will be terminated shortly after.')
parser.add_argument("--override", dest='override_old', action="store_true", default=False)
parser.add_argument("--render_env", dest='render_env', action="store_true", default=False,
    help='Enable env rendering if obs are collected from an agent.')


if __name__ == '__main__':
    try:
        os.chdir(__NEW_CWD__)
        args = parser.parse_args()
        if args.collect_from == 'agent':
            collect_agent_obs(args)
        elif args.collect_from == 'statebuffer':
            collect_statebuffer_obs(args)
    finally:
        os.chdir(__ORIG_CWD__)