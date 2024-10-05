import gym
import numpy as np
import Custom_envs
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EventCallback
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
import sys

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3.common.logger import Figure
import wandb
from wandb.sdk.lib import telemetry as wb_telemetry

# from wandb.integration.sb3 import WandbCallback
import logging


from fractions import Fraction
logger = logging.getLogger(__name__)
directory = 'lorenz/'

import argparse
import yaml

parser = argparse.ArgumentParser(description ='data')

parser.add_argument('-d', dest = 'path_to_config',action = 'store')

args = parser.parse_args()

path_to_config = args.path_to_config

with open(path_to_config, 'r') as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)

config = data['config']
wandb_config = data['wandb']


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


# class TensorboardCallback(BaseCallback):
#     """
#     Custom callback for plotting additional values in tensorboard.
#     """

#     def __init__(self, verbose=0):
#         super(TensorboardCallback, self).__init__(verbose)

#     def _on_step(self) -> bool:
#         # Log scalar value (here a random variable)
#         reward_lorenz = self.cost
#         self.logger.record('reward', reward_lorenz)
#         return True

# 



class WandbCallback(BaseCallback):
    """Callback for logging experiments to Weights and Biases.

    Log SB3 experiments to Weights and Biases
        - Added model tracking and uploading
        - Added complete hyperparameters recording
        - Added gradient logging
        - Note that `wandb.init(...)` must be called before the WandbCallback can be used.

    Args:
        verbose: The verbosity of sb3 output
        model_save_path: Path to the folder where the model will be saved, The default value is `None` so the model is not logged
        model_save_freq: Frequency to save the model
        gradient_save_freq: Frequency to log gradient. The default value is 0 so the gradients are not logged
        log: What to log. One of "gradients", "parameters", or "all".
    """

    def __init__(
        self,
        verbose: int = 0,
        model_save_path: Optional[str] = None,
        model_save_freq: int = 0,
        eval_freq: int = 10000,
        gradient_save_freq: int = 0,
        log: Optional[Literal["gradients", "parameters", "all"]] = "all",
    ) -> None:
        super().__init__(verbose)
        if wandb.run is None:
            raise wandb.Error("You must call wandb.init() before WandbCallback()")
        with wb_telemetry.context() as tel:
            tel.feature.sb3 = True
        self.model_save_freq = model_save_freq
        self.model_save_path = model_save_path
        self.eval_freq = eval_freq
        self.gradient_save_freq = gradient_save_freq
        if log not in ["gradients", "parameters", "all", None]:
            wandb.termwarn(
                "`log` must be one of `None`, 'gradients', 'parameters', or 'all', "
                "falling back to 'all'"
            )
            log = "all"
        self.log = log
        # Create folder if needed
  
        if self.model_save_path is not None:
            os.makedirs(self.model_save_path, exist_ok=True)
            self.path = os.path.join(self.model_save_path, "model.zip")
        else:
            assert (
                self.model_save_freq == 0
            ), "to use the `model_save_freq` you have to set the `model_save_path` parameter"
        
        self.first_link = []
        self.second_link = []
        self.first_velocity = []
        self.second_velocity = []
        self.action = []

    def _init_callback(self) -> None:
        d = {}
        if "algo" not in d:
            d["algo"] = type(self.model).__name__
        for key in self.model.__dict__:
            if key in wandb.config:
                continue
            if type(self.model.__dict__[key]) in [float, int, str]:
                d[key] = self.model.__dict__[key]
            else:
                d[key] = str(self.model.__dict__[key])
        if self.gradient_save_freq > 0:
            wandb.watch(
                self.model.policy,
                log_freq=self.gradient_save_freq,
                log=self.log,
            )
        wandb.config.setdefaults(d)

    def _on_step(self) -> bool:
        self.first_link.append(self.locals['infos'][0]['first_link'])
        self.second_link.append(self.locals['infos'][0]['second_link'])
        self.first_velocity.append(self.locals['infos'][0]['first_velocity'])
        self.second_velocity.append(self.locals['infos'][0]['second_velocity'])
        self.action.append(self.locals['infos'][0]['action'])
        # print ("here")
        if self.model_save_freq > 0:
            if self.model_save_path is not None:
                if self.n_calls % self.model_save_freq == 0:
                    # print ("here 2")
                    # self.logger.record("mean angle of first link", np.mean(np.abs(self.first_link)))
                    self.save_model()
                    # self.first_link = []
                    # self.logger.dump(self.num_timesteps)

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.logger.record("mean angle of first link", np.mean(np.abs(self.first_link)))
            self.logger.record("mean angle of second link", np.mean(np.abs(self.second_link)))
            self.logger.record("mean velocity of first link", np.mean(np.abs(self.first_velocity)))
            self.logger.record("mean velocity of second link", np.mean(np.abs(self.second_velocity)))
            self.logger.record("mean action", np.mean(np.abs(self.action)))

            self.logger.dump(self.num_timesteps)

            self.first_link = []
            self.second_link = []

        return True

    def _on_training_end(self) -> None:
        if self.model_save_path is not None:
            self.save_model()

    def save_model(self) -> None:
        self.model.save(self.path)
        wandb.save(self.path, base_path=self.model_save_path)
        if self.verbose > 1:
            logger.info(f"Saving model checkpoint to {self.path}")


run = wandb.init(
    project=wandb_config['project'],
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    mode = "offline",
    tags=wandb_config['tags']
)

env = gym.make("double_pendulum_le-v0", alpha = config["alpha"], reward_type=config['reward_type'], max = config['max'])

model = SAC(config["policy_type"], env ,gamma=config['gamma'], 
    learning_rate=config['learning_rate'],
    tensorboard_log="double_pendulum_tensorboard/")
model.learn(total_timesteps=config["total_timesteps"], callback=WandbCallback(model_save_path=f"models_for_paper/double_pendulum/{run.id}",eval_freq=100,verbose=1,),)

run.finish()
