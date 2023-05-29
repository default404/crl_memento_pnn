

import os, sys, inspect
__PRJ_DIR__ = os.path.dirname(os.path.dirname(os.path.realpath(inspect.getfile(lambda: None))))
if not __PRJ_DIR__ in sys.path: #insert the project dir in the sys path to find modules
    sys.path.insert(0, __PRJ_DIR__)

import unittest, time
from pathlib import Path
from copy import deepcopy

import gym
from ray.rllib.env.atari_wrappers import MonitorEnv, get_wrapper_by_cls

from envs.continual_env import ContinualEnv
from common.util import load_config_and_update, convert_to_tune_config
from envs.ray_env_util import wrap_atari


class TestContinualEnv(unittest.TestCase):
    """NOTE: 
        - To narrow possible errors down, the utility test should 
            be run before this class.
        - This will implicitly test the atari builder functions.
    """
    _DEFAULT_CONFIG_PATH = Path(__PRJ_DIR__) / "continual_atari/configs/default_config.yml"
    default_wrapper = {
            "env_style": "deepmind",
            "image_dim": 84,
            "dm_enable_frameSkip": True,
            "enable_noopInit": False,
            "noop_start_max": 30,
            "enable_frameSkip": False,
            "skip_frames": 4
    }

    def setUp(self):
        assert self._DEFAULT_CONFIG_PATH.is_file(), "The config file is not at the expected location!"
        updates = {
            "algorithm": "pnn",
            "Continual_params": {
                "task_list": ["BreakoutNoFrameskip-v4", "BeamRiderNoFrameskip-v4", "QbertNoFrameskip-v4"],
                "per_env_config": {
                    "BreakoutNoFrameskip-v4": None,
                    "BeamRiderNoFrameskip-v4": None,
                    "QbertNoFrameskip-v4": None
                },
                'env_config': {
                    "add_monitor": True,
                    "init_on_creation" : False
                },
                "default_wrapper_params": self.default_wrapper
            }
        }
        default_config = load_config_and_update(self._DEFAULT_CONFIG_PATH, updatesDict=updates)
        self.config = convert_to_tune_config(default_config)
        self.env = ContinualEnv(self.config['env_config'])


    def test_0_setup(self):
        task_names = ["BreakoutNoFrameskip-v4", "BeamRiderNoFrameskip-v4", "QbertNoFrameskip-v4"]

        self.assertIsInstance(self.env, gym.Env)

        #check that the task names were extracted successfully
        self.assertListEqual(self.env.get_task_names(), task_names)

        #check that the _task property was fetches correctly
        _task_list = [self.env.get_task_property(i,"_task") for i in range(len(self.env.get_task_names()))]
        self.assertListEqual(_task_list, task_names)

        #check that the env creators were created correctly
        for env_creator in [self.env.get_task_property(i, 'env_creator') for i in range(len(self.env.get_task_names()))]:
            self.assertTrue(callable(env_creator))
        
        #check that the stats dict is initialized to the default one
        self.assertDictEqual(self.env.get_task_stats(0), ContinualEnv.default_per_env_metrics)
        self.assertEqual(len(self.env.get_stats()), 3)
        
    
    def test_1_nextTask(self):
        
        #set the first task
        self.env.next_task()
        #check that env is now init with the first task and the order is contained
        self.assertEqual(self.env._curr_task_id, 0)
        self.assertEqual(self.env.get_task_name(), "BreakoutNoFrameskip-v4")

        #check that the current env has the Monitor attached
        self.assertIsInstance(get_wrapper_by_cls(self.env.env, MonitorEnv), MonitorEnv)

        #check that the spec finished flag is not set yet
        spec = self.env.collect_env_stats()
        self.assertFalse(spec['finished'])
        
        #set the next task
        self.env.next_task()
        self.assertEqual(self.env._curr_task_id, 1)
        self.assertEqual(self.env.get_task_name(), "BeamRiderNoFrameskip-v4")

        #check that the spec finished flag is set for the first task
        # Note that spec is updated in-place 
        self.assertTrue(spec['finished'])


    def test_2_raiseConditions(self):
        #check that env is uninitialized
        self.assertIsNone(self.env._curr_task_id)
        with self.assertRaises(ValueError):
            #check that task with incorrect ID cannot be set 
            self.env.set_task(10)
        
        with self.assertRaises(IndexError):
            #check that stats from a not existing task id cannot be accessed
            self.env.get_stats_property(10, 'finished')

        with self.assertRaises(TypeError):
            #check that the uninit env can not return infos of the current task
            self.env.get_task_name()
        
        #init first env
        self.env.next_task()
        with self.assertRaises(ValueError):
            #check that current task can not be set again w/o the revisit flag
            self.env.set_task(0)


    def test_3_addTask(self):
        task_names = ["BreakoutNoFrameskip-v4", "BeamRiderNoFrameskip-v4", 
            "BreakoutNoFrameskip-v4", "QbertNoFrameskip-v4"]
        # #set the first task
        # self.env.next_task()
        #add a task to the continual env task list at the correct position
        self.env.add_task("BreakoutNoFrameskip-v4", self.default_wrapper, 2)
        self.assertListEqual(self.env.get_task_names(), task_names)


    def test_4_methodDelegation(self):
        """NOTE: This test is only valid iff:
            * the base env is a gym.Env
            * first task is "BreakoutNoFrameskip-v4"
            * second task is "BeamRiderNoFrameskip-v4"
            * most-outer env wrapper is a 'FrameStack'
        """
        #test some members that are not implemented explicitly in the continual env
        attr_to_delegate = ['unwrapped', 'metadata', 'observation_space', 'action_space', 
            'reward_range', 'class_name', 'close', 'compute_reward', 'frames', 
            'k', 'render', 'seed', 'spec', 'get_action_meanings']
        
        #set the first task
        self.env.next_task() # = BreakoutNoFrameskip-v4

        #check that the unwrapper returns the same base env than the current task
        correct_base_env = self.env.env.unwrapped.__class__.__name__
        actual_env_name = self.env.unwrapped.__class__.__name__
        self.assertEqual(actual_env_name, correct_base_env)

        #check each delegated attribute
        for a in attr_to_delegate:
            # ContinualEnv obj now has this member
            self.assertTrue(hasattr(self.env, a), "Delegated attribute {} is missing!".format(a))
            self.assertIsNotNone(getattr(self.env, a), 
                "Delegated attribute {} was not forwarded correctly (is None)!".format(a))
            #double-check that the correct task is used
            if a == 'spec':
                self.assertEqual(getattr(self.env, a).id, "BreakoutNoFrameskip-v4")

        # === Tests for task switch ===
        #store a dict of attribute id's to check that these are correctly overridden
        # when the task is switched
        t1_ids = {}
        for a in attr_to_delegate:
            mem = getattr(self.env, a)
            if callable(mem):
                t1_ids[a] = id(mem)
            else:
                t1_ids[a] = 'static_type'
            
        # #add an auxillary member to env that should be removed when switched to the next task
        # self.env.aux_task = lambda: "I should be removed on task switch"
        # #now it looks like the came from the current task
        # self.env._DELEGATION_LIST.append('aux_task')
        # =============================

        #set the next task
        self.env.next_task() # = BeamRiderNoFrameskip-v4

        # 'aux_task' should be not in the list anymore!
        # self.assertNotIn('aux_task', self.env._DELEGATION_LIST)
        # self.assertFalse(hasattr(self.env, 'aux_task'))
        
        #check each delegated attribute
        for a in attr_to_delegate:
            # ContinualEnv obj now has this member
            self.assertTrue(hasattr(self.env, a), "Delegated attribute {} is missing!".format(a))
            # check that the methods from the old task where overridden
            # self.assertNotEqual(id(getattr(self.env, a)), t1_ids[a], 
            #     "The member {} has the same id as for the previous task!".format(a))
            #double-check that the correct task is used
            if a == 'spec':
                self.assertEqual(getattr(self.env, a).id, "BeamRiderNoFrameskip-v4")

        for a in attr_to_delegate:
            if not t1_ids[a] == 'static_type':
                self.assertNotEqual(id(getattr(self.env, a)), t1_ids[a], 
                    "Member {} has same obj id".format(a))


    def test_5_runEnv(self):

        def run_env_random(steps=100, render=False, pause=0.05):
            obs = self.env.reset()
            for i in range(steps):
                if render:
                    self.env.render()
                    if pause > 0: time.sleep(pause) #pause for some time  
                act = self.env.action_space.sample()
                obs, rew, done, info = self.env.step(act)
                if done:
                    obs = self.env.reset()

            self.env.close()
            return self.env.collect_env_stats()


        #set the first task and run shortly (should be less than one episode)
        self.env.next_task()
        spec = deepcopy(run_env_random(50, True))

        #check that after reset (but no step) the spec are still the same
        #NOTE: On live lose, the done sig is send by the EpisodicLife 
        # wrapper to the outer wrappers but it doesnt truly reset the env 
        # until the step func from the underlying env sends the done sig.
        #NOTE: For most of the stats we need to compare if they are *almost*
        # equal because
        self.env.reset()
        spec_after_reset = deepcopy(self.env.collect_env_stats())
        for k,v in spec_after_reset.items():
            if k == 'finished':
                self.assertFalse(v, "The "+k+" flag should now be True!")
            elif k == 'total_episodes':
                self.assertEqual(spec[k], v, 
                    "The number of total episodes directly before and after env reset should still match!")
            elif isinstance(v,list):
                for i,vv in enumerate(v):
                    self.assertAlmostEqual(spec[k][i], vv, delta=40,
                        msg="The metric "+k+" directly before and after env reset should be almost the same!")
            else:
                self.assertAlmostEqual(spec[k], v, delta=40,
                    msg="The metric "+k+" directly before and after env reset should be almost the same!")
        
        #set next task
        self.env.next_task()

        #check that after task switch the metrics are still valid
        spec_after_switch = deepcopy(self.env.get_task_stats(0))
        for k,v in spec_after_switch.items():
            if k == 'finished':
                self.assertTrue(v, "The "+k+" flag should now be True!")
            elif k == 'total_episodes':
                self.assertEqual(spec[k], v, 
                    "The number of total episodes directly before and after env reset should still match!")
            elif isinstance(v,list):
                for i,vv in enumerate(v):
                    self.assertAlmostEqual(spec[k][i], vv, delta=40,
                        msg="The metric "+k+" directly before and after env reset should be almost the same!")
            else:
                self.assertAlmostEqual(spec[k], v, delta=40,
                    msg="The metric "+k+" directly before and after env reset should be almost the same!")

        #run the next env for a longer time (should be longer than one episode)
        self.env.next_task() #= QbertNoFrameskip-v4
        spec = run_env_random(500, True, 0.01)

        #check that the episode number is increased correctly
        self.assertGreater(spec['total_episodes'], 1, 
            "The num of episodes is still only 1! "
            "Note that his test can fail sometimes, if the agent randomly performs too well...")
        

if __name__ == "__main__":
    unittest.main()