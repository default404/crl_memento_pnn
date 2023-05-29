
import os, sys, inspect
__PRJ_DIR__ = os.path.dirname(os.path.dirname(os.path.realpath(inspect.getfile(lambda: None))))
if not __PRJ_DIR__ in sys.path: #insert the project dir in the sys path to find modules
    sys.path.insert(0, __PRJ_DIR__)

import unittest
from pathlib import Path

from common.util import load_config_and_update, convert_to_tune_config
from ray.tune import grid_search


class TestUtils(unittest.TestCase):

    _DEFAULT_CONFIG_PATH = Path(__PRJ_DIR__) / "continual_atari/configs/default_config.yml"

    def setUp(self):
        assert self._DEFAULT_CONFIG_PATH.is_file(), "The config file is not at the expected location!"

    def test_0_loadConfig(self):
        config = load_config_and_update(self._DEFAULT_CONFIG_PATH)

        #inspect the top keys of the config
        top_keys = ["auto_managed", "algorithm", "Continual_params", "Trainer_params"]
        self.assertListEqual(top_keys, list(config.keys()))

        #inspect the auto_managed fields
        self.assertIn("parent_config", list(config["auto_managed"].keys()))
        #check that the parent_config field was added correct
        self.assertEqual(str(self._DEFAULT_CONFIG_PATH), config["auto_managed"]["parent_config"])

        #inspect the Continual_params fields
        self.assertIn("task_list", list(config["Continual_params"].keys()))
        self.assertIsInstance(config["Continual_params"]["task_list"], list)
        self.assertIsInstance(config["Continual_params"]["per_env_config"], dict)


    def test_1_loadConfigWithUpdates(self):

        updates = {
            "Continual_params": {"task_list": ["T1", "T2"]}
        }

        config = load_config_and_update(self._DEFAULT_CONFIG_PATH, updatesDict=updates)
        #check that the update worked
        self.assertIn("task_list", list(config["Continual_params"].keys()))
        self.assertListEqual(["T1", "T2"], config["Continual_params"]["task_list"])

    def test_2_convertToTune(self):
        #Here task "T3" is intentionally missing so it should be added to
        #the env_config with default params. This will throw a logger warning btw.
        updates = {
            "Continual_params": {
                "task_list": ["T1", "T2", "T3"],
                "per_env_config": {
                    "T1": None,
                    "T2": {"env_style": "mystyle", "someParam": 42}
                },
                'env_config': {
                    "add_monitor": True,
                    "init_on_creation" : False
                },
                "default_wrapper_params":{
                    "env_style": "deepmind",
                    "image_dim": 84,
                    "dm_enable_frameSkip": True,
                    "enable_noopInit": False,
                    "noop_start_max": 30,
                    "enable_frameSkip": False,
                    "skip_frames": 4
                }
            }
        }
        config = load_config_and_update(self._DEFAULT_CONFIG_PATH, updatesDict=updates)
        #set the algo value to something continal-learnable
        config["algorithm"] = "progress_compress"
        #change one parameter to a list so it will become a gridsearch
        config["Trainer_params"]["kl_coeff"] = [0.2, 0.5, 0.9]
        #get the config in tune flavor
        tune_config = convert_to_tune_config(config)

        #check SOME of the top keys the tune config must have now
        for k in ["env","env_config","algorithm","num_workers","train_batch_size","kl_coeff"]:
            self.assertIn(k, list(tune_config.keys()))
        #check that the gridsearch conversion worked
        self.assertEqual(tune_config["kl_coeff"], grid_search([0.2, 0.5, 0.9]))
        #check that the Tune env_config field was updated with the Continual env_config
        self.assertEqual(tune_config["env_config"].get('init_on_creation', None), False)
        #check that the task list was copied successfully
        self.assertListEqual(tune_config["env_config"]["task_list"], ["T1", "T2", "T3"])
        #check that the per-env params were handled correctly
        self.assertDictEqual(
            tune_config['env_config']['per_env_config']["T1"], 
            updates["Continual_params"]["default_wrapper_params"])
        self.assertDictEqual(
            tune_config['env_config']['per_env_config']["T2"], 
            {"env_style": "mystyle", "someParam": 42})
        self.assertDictEqual(
            tune_config['env_config']['per_env_config']["T3"], 
            updates["Continual_params"]["default_wrapper_params"])

      
if __name__ == "__main__":
    unittest.main()

