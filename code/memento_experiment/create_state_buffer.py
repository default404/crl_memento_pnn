
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

import argparse, yaml
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pickle, queue

#data science & DL packages
import numpy as np
import pandas as pd
import seaborn as sns
import torch
# from torch.autograd import Variable
import gym
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
# from ray.rllib.models.model import flatten

# project utilities
from common.util import (setup_logger, make_new_dir, load_yml_config, convert_to_tune_config, dict_deep_update,
    remove_gridsearch)
import common.exp_analysis as expAnalysis
# continual policies / models
from continual_atari.agents.policies.pnn.pnn_model import PNN
#custom envs
from envs.ray_env_util import register_envs
#eval plotting
from memento_experiment.plot_eval_csv import main as eval_plotter


# handle the cwd for script execution
__ORIG_CWD__ = os.getcwd()
__NEW_CWD__ = __MAINDIR__

# PATH AND FILE NAME CONSTANTS
_DEFAULT_CONFIG = Path(__MEMENTO_DIR__) / "memento_eval_config.yml"
from memento_experiment import MEMENTO_BUFFER, EVAL_CSV_NAME, MEMENTO_CSV_NAME
# MEMENTO_BUFFER = 'memento.buffer'
# EVAL_CSV_NAME = 'evaluation.csv'
# MEMENTO_CSV_NAME = 'memento_info.csv'

#The default logger
_logger = setup_logger(name=__name__,verbose=True)


#NOTE: this could be moved to utility
def convertToNonTorchTensor(tensor):
    """Converts a Torch tensor to its numpy value type.
    Un-mapped clone of rllib.utils.torch_ops.convert_to_non_torch_type
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().item() if len(tensor.size()) == 0 else \
            tensor.cpu().detach().numpy()
    else:
        return tensor

#NOTE: this could be moved to utility
def convertToTorchTensor(item, device=None):
    """Converts an element to a Torch tensor and maps to device if given.
    Un-mapped clone of rllib.utils.torch_ops.convert_to_torch_tensor
    """
    if torch.is_tensor(item):
        return item if device is None else item.to(device)
    tensor = torch.from_numpy(np.asarray(item))
    # Floatify all float64 tensors.
    if tensor.dtype == torch.double:
        tensor = tensor.float()
    return tensor if device is None else tensor.to(device)


class RLlibPreprocFilter(gym.ObservationWrapper):
    """Convenience wrapper for RLlib preprocessor and filter handling
    of a given env.
    Env sampling is called with the proprocs/filters in RLlib (for refencence): 
    https://github.com/ray-project/ray/blob/releases/0.8.6/rllib/evaluation/sampler.py#L353
    """
    def __init__(self, env, _preprocessor=None, _filter=None):
        super(RLlibPreprocFilter, self).__init__(env)
        self._preprocessor = _preprocessor
        self._filter = _filter
        if self._preprocessor:
            self.observation_space = self._preprocessor.observation_space
        else:
            self.observation_space = env.observation_space

    def observation(self, obs):
        if self._preprocessor:
            obs = self._preprocessor.transform(obs)
        if self._filter:
            obs = self._filter(obs)
        return obs


#NOTE: this could be moved to env utility
def fake_ray_env_creation(config, 
                          *, 
                          monitordir=None, 
                          last_only=True, 
                          disable_episodicLife=True,
                          logger=_logger):
    '''Reconstructs the identical environments as created by RLlib Trainers.
    If last_only is True, only the lst env in the task list will be created.
    Returns the actual env for each registered env ID in the Tune `config`.
    '''
    from ray.tune.registry import ENV_CREATOR, _global_registry
    from ray.rllib.models.preprocessors import NoPreprocessor
    from ray.rllib.utils.filter import get_filter
    from ray.rllib.env.atari_wrappers import is_atari
    # the list of actual envs in the same order as the task list 
    # if multiple envs shall be evaluated
    envs = []
    _env_ids = config['env_config'].get('task_list') or [config['env']]
    if not _env_ids:
        raise ValueError('No env ID(s) in the config->task_list/env fields! '
                         'Cannot create any evaluation environment')
    if last_only:
        envs_to_create = [_env_ids[-1]]
    else:
        envs_to_create = _env_ids

    for tid, env_id in enumerate(envs_to_create):
        if _global_registry.contains(ENV_CREATOR, env_id):
            env_creator = _global_registry.get(ENV_CREATOR, env_id)
        elif env_id in [e.id for e in gym.envs.registry.all()]:
            env_creator = lambda env_config: gym.make(env_id)
        else:
            err_str = f'Env ID {env_id} is not registered in the Tune registry nor a basic Gym env!'
            logger.error(err_str)
            raise ValueError(err_str)

        if last_only:   #if we only eval the last task, get the correct task id still
            tid = len(_env_ids)-1

        ########### create the base env ###########
        env = env_creator(config["env_config"])

        ########### add monitoring and check Preprocessor behavior ###########
        if config["monitor"]:
            if monitordir:
                monitordir = Path(monitordir) / f'{tid}_{env_id}' / 'episode_monitor'
                monitor_path = str(make_new_dir(monitordir, override=True, logger=logger))
            else:
                logger.warning('The config was set to add a monitoring wrapper to the env '
                    f'but no monitor output path was given! Monitoring will be diabled.')
                monitor_path = None
        else:
            monitor_path = None
        preprocessing_enabled = True

        if is_atari(env) and \
                not config["model"].get("custom_preprocessor") and \
                config.get("preprocessor_pref") == "deepmind":

            # Deepmind wrappers already handle all preprocessing
            preprocessing_enabled = False

            if config["clip_rewards"] is None:  #unused for now
                config["clip_rewards"] = True

            def wrap(env):
                from envs.atari_env_wrapper import wrap_deepmind
                env = wrap_deepmind(
                    env,
                    dim=config["model"].get("dim"),
                    framestack=config["model"].get("framestack"),
                    episodic_life=(not disable_episodicLife))
                if monitor_path:
                    from gym import wrappers
                    env = wrappers.Monitor(env, monitor_path, resume=True)
                return env
        
        else:
            def wrap(env):
                if monitor_path:
                    from gym import wrappers
                    env = wrappers.Monitor(env, monitor_path, resume=True)
                return env

        env = wrap(env)
        #TODO: check if the returned env from the Tune registry is a BaseEnv or a Gym env!
        
        ########### get the needed Preprocessor ###########
        obs_space = env.observation_space
        if preprocessing_enabled:
            preproc_cls = ModelCatalog.get_preprocessor_for_space(
                obs_space, config.get("model"))
            preprocessor = preproc_cls
        else:
            preprocessor = NoPreprocessor(obs_space, None)

        ########### get the needed Filter ###########
        #NOTE: this will be moved into filter manager in later Ray versions 
        filtr = get_filter(config["observation_filter"], preprocessor.observation_space.shape) 

        #NOTE: no reward or action clipping applied for evaluation now
        # see clip_rewards: https://github.com/ray-project/ray/blob/releases/0.8.6/rllib/evaluation/sampler.py#L433
        # see clip_actions: https://github.com/ray-project/ray/blob/releases/0.8.6/rllib/evaluation/sampler.py#L839

        ########### wrap the env with preprocessor and filter wrapper ###########
        env = RLlibPreprocFilter(env, preprocessor, filtr)
        
        envs.append({'task_id': tid, 'env_id': env_id,'env': env})
        
    logger.debug('Created {} env(s) for evaluation:\n\t{}'.format(len(envs), 
        '\n\t'.join([f"{e['task_id']}: {e['env_id']} :: {str(e['env'])}" for e in envs])))

    return envs


def get_real_done(env):
    '''Get the true done flag for the current env.
    This can be different from the flag returned by env.step
    if specific wrappers are used.
    '''
    true_done = None
    if 'EpisodicLifeEnv' in str(env):
        true_done = env.was_real_done
    return true_done


#################################################################

def eval_pnn(args):
    """Main eval function for the PNN model. 
    This loads 2 configs - one for this tool and the original config from the agent training
    and updates all config fields for eval mode. Then it creates a PNN model from a 
    checkpoint and re-constructs the last environment from the task list in the training 
    config in RLlib-flavor and evaluates it for nb of episodes until the target state buffer 
    size is reached. 
    Along, the Memento state buffer is created that stores state-return pairs from each
    episode where the max return was reached first. This buffer can be used with the Memento
    environment to train on defined env states.
    """
    date = datetime.today().strftime('%Y-%m-%d')
    print("Loading eval config...")
    eval_config = load_yml_config(args.config)
    #update the eval config with CLI args
    #TODO: update eval_config with ALL CLI args 
    eval_config = dict_deep_update(eval_config, 
        {
            'experiment_dir': args.exp_dir if args.exp_dir else eval_config.get('experiment_dir'),
            'checkpoint': args.checkpoint if args.checkpoint else eval_config.get('checkpoint'),
            'env_spec': args.env_spec if args.env_spec else eval_config.get('env_spec'),
            'exp_config': args.exp_config if args.exp_config else eval_config.get('exp_config'),
        })


    if not args.output:
        exp_dir = Path(eval_config.get('experiment_dir')).expanduser()
        if not exp_dir.is_dir():
            raise NotADirectoryError(
                'Given experiment dir [{}] is not a directory!'.format(exp_dir))
        exp_name = exp_dir.name
        output_dir = exp_dir / 'evaluation_w_statebuffer'
    else:
        output_dir = Path(args.output).expanduser()
        exp_name = output_dir.name
    output_dir = make_new_dir(output_dir, override=args.override_old)


    # setup logger
    logger = setup_logger(name=exp_name, log_path=output_dir/'eval_sb.log', verbose=True)
    logger.info("+---------------------------------------------------------------+")
    logger.info("|  START MODEL EVAL & CREATE STATE BUFFER: {:^20} |".format(exp_name))
    logger.info("+---------------------------------------------------------------+")

    #TODO: store eval_config to output dir to be able to see the used checkpoint, etc

    logger.info("Loading training config from experiment dir...")
    exp_config_file = eval_config.get('exp_config')
    if not Path(exp_config_file).is_file():
        exp_config_file = list(exp_dir.glob('exp_config*.yml'))
        if len(exp_config_file) > 0:
            exp_config_file = exp_config_file[0]
        else:
            exp_config_file = list(exp_dir.glob('*.yml'))
            if len(exp_config_file) == 1:
                exp_config_file = exp_config_file[0]
            else:
                raise FileNotFoundError('Could not find a unique config for the experiment to be evaluated! '
                        'Without it the model and env cannot be reconstructed the same way as in the training.')
    exp_config = load_yml_config(exp_config_file)
    ray_config, aux_dict = convert_to_tune_config(exp_config, False, logger)

    # get the available SLURM resources if applicable
    # num_cpus, num_gpus = get_slurm_resources(ray_config, logger)

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

    # save the eval config yml
    # with open(output_dir/'eval_exp_config.yml','w') as f:
    #     yaml.safe_dump(exp_config, f, default_flow_style=False)
    
    # create all the envs to evaluate
    logger.info('Reconstructing Ray envs from training config...')
    ray.init(configure_logging=False, local_mode=True)  #a running Ray instance is needed to read from the registry
    logger.info('Registering custom environments if needed...')
    register_envs(ray_config, logger=logger)
    eval_last_env_only = True
    envs = fake_ray_env_creation(ray_config, monitordir=output_dir, 
                                 last_only=eval_last_env_only, 
                                 disable_episodicLife=True, logger=logger)
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
    explore = args.explore     #define exploring behavior for action sampling from model logit distribution
    epsilon = args.epsilon      #probability of exploration
    render_env = args.render_env   #shows the agent during evaluation
    save_episodes = args.save_episodes
    eval_df_file = output_dir / EVAL_CSV_NAME
    momento_df_file = output_dir / MEMENTO_CSV_NAME
    logger.info(f'Using flags: \n\tExploration: {explore}, epsilon: {epsilon} '
                f'\n\tStallment detection: {args.use_stall_detection} '
                f'\n\tEnv rendering: {render_env}'
                f'\n\tEpisode buffer saves: {save_episodes}')

    logger.info('\n########################################################################')
    logger.info('Starting evaluation and state buffer creation...\n')
    t_start_eval = datetime.now().replace(microsecond=0)
    ### for each task / env
    for env_dict in envs:
        tid = env_dict['task_id']
        env_id = env_dict['env_id']
        env = env_dict['env']

        logger.info(f'Running evaluation for task {tid}: {env_id} ({str(env)})')

        stat_dict = defaultdict(list)
        mem_stat_dict = defaultdict(list)

        if len(envs) == model.columns:
            if eval_last_env_only:
                #the loaded PNN must automatically have the same tid as the created env
                assert model._cur_task_id == tid
            else:
                model.set_task_id(tid)

        env_dir = output_dir / f'{tid}_{env_id}'
        if not env_dir.exists():
            os.makedirs(str(env_dir))

        state_buffer_path = env_dir / 'state_buffers'
        if not state_buffer_path.exists():
            os.makedirs(str(state_buffer_path))
        recent_tmp_buffer = state_buffer_path

        dist_class, _ = ModelCatalog.get_action_dist(
                    env.action_space, ray_config["model"], framework="torch")

        state_buffer = []       #state buffer that will be stored
        input_dict = {"obs": None, "is_training": False}
        ### for each episode
        t_start_env = datetime.now().replace(microsecond=0)
        for i in range(eval_config['state_buffer_size']):
            logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>')
            logger.info('Running episode {:>4d}/{:>4d}'.format(i+1, eval_config['state_buffer_size']))

            episode_buffer = []     #whole buffer for one episode
            t_start_ep = datetime.now().replace(microsecond=0)
            obs = env.reset()
            done = False
            step = 0    #timestep t
            reward = 0  #reward from env step
            if hasattr(env, 'reward_offset'):
                R_t = getattr(env, 'reward_offset')
            else:
                R_t = 0     #return of timestep t
            try:
                lifes = env.unwrapped.ale.lives()
            except:
                lifes = None

            stat_dict['Task'].append(tid)
            stat_dict['Environment'].append(env_id)
            stat_dict['Timesteps'].append(step)
            stat_dict['Episode'].append(i)
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
                    if 'FIRE' in env.unwrapped.get_action_meanings():
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
                    # done = get_real_done(env)
                    # if done is None:    #if no special condition must be checked for ep. end
                    #     done = _done

                t_step.append(datetime.now()-t_start_step)
                step += 1
                R_t += reward
                fifo_stall.append(R_t)
                if len(fifo_stall) > 1000: fifo_stall = fifo_stall[-1000:]

                episode_buffer.append( {'state':env.clone_state(), 'return':R_t, 'step':step} )

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
                stat_dict['Episode'].append(i)
                stat_dict['Step Rewards'].append(reward)
                stat_dict['Episode Return'].append(R_t)
                stat_dict['ALE.Lifes'].append(lifes)

                #emergency exit for broken done's
                if step > 1e5:  #the episode should have been finished long ago!
                    logger.warning('\nThe current episode never finished (no DONE return)! Terminating manually...\n')
                    break
            
            if render_env:
                env.close()
            logger.debug('Episode finished after {} steps ({})'.format(step, datetime.now().replace(microsecond=0)-t_start_ep))
            logger.debug('Episode mean model inference & action sample time: {}'.format(np.mean(t_infer)))
            logger.debug('Episode mean environment step time: {}'.format(np.mean(t_step)))

            if save_episodes:
                # also store the full episode buffers, maybe we need them later
                ep_buffer_path = state_buffer_path / 'episode_state_buffers'
                if not ep_buffer_path.exists():
                    os.makedirs(str(ep_buffer_path))
                with open(ep_buffer_path / 'episode_{:04d}.buffer'.format(i),'wb') as eb:
                    pickle.dump(episode_buffer, eb)

            # Fill the Memento buffer (i.e. store the fist episode buffer element with max return)
            logger.debug('Retrieving memento state from episode...')
            best_state = max(episode_buffer, key=lambda x: x['return'])
            state_buffer.append(best_state)

            mem_stat_dict['Episode'].append(i)
            mem_stat_dict['Return'].append(best_state['return'])
            mem_stat_dict['Task'].append(tid)
            mem_stat_dict['Environment'].append(env_id)

            if (i+1) % 10 == 0:
                # frequent checkpointing to prevent data loss if job takes too long
                logger.debug('Writing Memento buffer temporary checkpoint...')
                if recent_tmp_buffer.is_file():
                    recent_tmp_buffer.unlink()      #remove old buffer
                recent_tmp_buffer = state_buffer_path / f'tmp_{i}-{MEMENTO_BUFFER}'
                with open(recent_tmp_buffer,'wb') as sb:
                    pickle.dump(state_buffer, sb)
                pd.DataFrame.from_dict(stat_dict).to_csv(str(eval_df_file), index=False, na_rep='nan')
                pd.DataFrame.from_dict(mem_stat_dict).to_csv(str(momento_df_file), index=False, na_rep='nan')
        
        logger.info('<<<<<<<<<<<<<<<<<<<<<<<<<<')
        logger.info('Finished evaluation of env: {}. Elapsed time: {}'.format(env_id, datetime.now().replace(microsecond=0)-t_start_env))
        logger.info('Writing final state buffer for memento env...')
        with open(state_buffer_path / MEMENTO_BUFFER,'wb') as sb:
            pickle.dump(state_buffer, sb)
        # remove the temp memento buffer
        if recent_tmp_buffer.is_file():
            recent_tmp_buffer.unlink()

    #final csv saves
    pd.DataFrame.from_dict(stat_dict).to_csv(str(eval_df_file), index=False, na_rep='nan')
    pd.DataFrame.from_dict(mem_stat_dict).to_csv(str(momento_df_file), index=False, na_rep='nan')
    # eval_df.to_csv(str(eval_df_file), index=False, na_rep='nan')
    # mem_df.to_csv(str(momento_df_file), index=False, na_rep='nan')

    logger.info('Generating plots for the evaluation metrics...')
    eval_plotter(output_dir, logger)

    logger.info('')
    logger.info("+-----------------------------+")
    logger.info("|     FINISHED MODEL EVAL     |")
    logger.info("+-----------------------------+")
    logger.info('Execution time: {}'.format(datetime.now().replace(microsecond=0)-t_start_eval))



# input args
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', dest='exp_dir', type=str, default=None)
parser.add_argument('--output', type=str, default=None)
parser.add_argument('--config', type=str, default=_DEFAULT_CONFIG)
parser.add_argument("--checkpoint", dest='checkpoint', type=str, default=None,
    help='Checkpoint to init the model with.')
parser.add_argument("--env_spec", dest='env_spec', type=str, default=None,
    help='Path to the environment specification dump that contains the env obs and action spaces.')
parser.add_argument("--exp_config", dest='exp_config', type=str, default=None,
    help='Path to the experiment config that was used to train the loaded agent. If not given '
         'it is tried to be loaded from the experiment dir.')
parser.add_argument("--use_exploration", dest='explore', action="store_true", default=False,
    help='Whether to use epsilon-greedy exploration for action sampling')
parser.add_argument("--epsilon", dest='epsilon', type=float, default=0.001,
    help='Probability for epsilon-greedy exploration')
parser.add_argument("--use_stall_detection", dest='use_stall_detection', action="store_true", default=False,
    help='Whether to use stallment detection in the environment. This monitors the reward changes '
    'over the last 1k env steps and tries a reset action (e.g. FIRE) if the agent is stuck. If this '
    'doesnt work, the episode will be terminated shortly after.')
parser.add_argument("--override", dest='override_old', action="store_true", default=False)
parser.add_argument("--render_env", dest='render_env', action="store_true", default=False)
parser.add_argument("--save_episodes", dest='save_episodes', action="store_true", default=False)


if __name__ == '__main__':
    try:
        os.chdir(__NEW_CWD__)
        eval_pnn(parser.parse_args())
    finally:
        os.chdir(__ORIG_CWD__)