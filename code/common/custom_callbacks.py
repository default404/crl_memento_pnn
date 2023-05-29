import os
from typing import Dict
from pathlib import Path

import ray
from ray import tune
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.tune.result import (SHOULD_CHECKPOINT, DONE, TRAINING_ITERATION, 
                            TIMESTEPS_TOTAL)

from common.util import save_model_stats
        

class ModelSummaryCallback(DefaultCallbacks):

    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super(ModelSummaryCallback, self).__init__(legacy_callbacks_dict)

        self.log_model_summary_done = False
    
    def on_train_result(self, *, trainer, result: dict, **kwargs):
        if not self.log_model_summary_done:
            print(f'Writing model summary to {trainer.logdir}')
            fw = trainer.config['framework']
            # fw = trainer.framework
            input_shape = trainer.workers.local_worker().env.observation_space.shape
            save_model_stats(trainer.get_policy().model,
                             trainer.logdir,
                             framework=fw,
                             input_shape=input_shape)
            self.log_model_summary_done = True


class ModelCheckpointCallback(DefaultCallbacks):

    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super(ModelCheckpointCallback, self).__init__(legacy_callbacks_dict)
 
    
    def on_train_result(self, *, trainer, result: dict, **kwargs):
        trainer_config = trainer.get_config()
        if self._should_checkpoint(trainer_config, result):
            policy = trainer.get_policy()
            model = policy.model

            save_dir = Path(trainer.logdir) / 'checkpoints'
            if not save_dir.exists():
                os.makedirs(str(save_dir), exist_ok=True)
            task_id = getattr(model, 'cur_task_id', None)
            if task_id is not None:     #we train a custom model and can assume that model checkpointing is implemented
                it = result.get(TRAINING_ITERATION,'NaN')
                ts = result.get(TIMESTEPS_TOTAL,'NaN')
                model.export_checkpoint(str(save_dir), f'model_ckpt_T-{task_id}_it-{it}_st-{ts}.h5')
            else:
                try:
                    policy.export_model(save_dir)
                except NotImplementedError as nie:
                    print(f'\nWARNING: Training a default Rllib model ({model.__class__.__name__}) where '
                          f'model export is not implemented!\nTraceback: {nie}\n')


    def _should_checkpoint(self, config, result):
        # check the stopping conditions
        # checkpointing is queued by trainer
        should_checkpoint = result.get(SHOULD_CHECKPOINT,False) or result[DONE]
        if not should_checkpoint:
            # checkpointing because checkpoint frequency is reached
            ckpt_frequ = max(config['env_config'].get('checkpoint_frequency',0),0)
            should_checkpoint = (bool(ckpt_frequ) and result.get(TRAINING_ITERATION,float('inf')) % ckpt_frequ == 0)
        if not should_checkpoint:
            #always checkpoint at the end of training
            #since there is no way to access the Trial from the Trainer
            #the stopping criteria is checked manually
            stoppers = config['env_config'].get('stopping_criteria', {})
            for n in stoppers.keys():
                try:
                    if result[n] >= stoppers[n]:
                        should_checkpoint = True
                        print(f"\nStopping criterion [{n} = {stoppers[n]}] reached!\n")
                        break
                except KeyError:
                    print(f"Stopping criterion {n} is not in the training result and cannot be evaluated!")
                    continue
        return should_checkpoint
        

class UnifiedCallbacks(DefaultCallbacks):
    callback_classlist = [ModelSummaryCallback, ModelCheckpointCallback]

    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super(UnifiedCallbacks, self).__init__(legacy_callbacks_dict)

        self.cbk_list = [c(legacy_callbacks_dict) for c in self.callback_classlist 
                         if type(c)==type]

    def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        for c in self.cbk_list:
            c.on_episode_start(worker=worker, base_env=base_env, policies=policies, 
                               episode=episode, env_index=env_index, **kwargs)

    def on_episode_step(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        for c in self.cbk_list:
            c.on_episode_step(worker=worker, base_env=base_env, episode=episode, 
                              env_index=env_index, **kwargs)
    
    def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode, env_index: int, **kwargs):
        for c in self.cbk_list:
            c.on_episode_end(worker=worker, base_env=base_env, policies=policies,
                             episode=episode, env_index=env_index, **kwargs)

    def on_sample_end(self, *, worker: "RolloutWorker", samples: SampleBatch,
                      **kwargs):
        for c in self.cbk_list:
            c.on_sample_end(worker=worker, samples=samples, **kwargs)
    
    def on_train_result(self, *, trainer, result: dict, **kwargs):
        for c in self.cbk_list:
            c.on_train_result(trainer=trainer, result=result, **kwargs)

    def on_postprocess_trajectory(
            self, *, worker: "RolloutWorker", episode: MultiAgentEpisode,
            agent_id: AgentID, policy_id: PolicyID,
            policies: Dict[PolicyID, Policy], postprocessed_batch: SampleBatch,
            original_batches: Dict[AgentID, SampleBatch], **kwargs):
        for c in self.cbk_list:
            c.on_postprocess_trajectory(worker=worker, episode=episode, agent_id=agent_id,
                                        policy_id=policy_id, policies=policies, 
                                        postprocessed_batch=postprocessed_batch,
                                        original_batches=original_batches, **kwargs)




# def callback_inject_wrapper(cls, **kwargs):
#     '''Injects class attributes to the given cls'''
#     for att,val in kwargs.items():
#         setattr(cls, att, val)
    
#     return cls