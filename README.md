# TruncLE


## Installation

### Create Anaconda environment
```
conda create -n mastering_chaos python=3.10.9
pip install setuptools==65.5.0 pip==21
pip install wheel==0.38.0
```

### Install libraries and custom environments
```
conda activate mastering_chaos
pip install -r requirements.txt
pip install -e Custom_envs
pip install --upgrade numpy==1.24.2
```

### Run experiments

please create a wandb account at https://wandb.ai

```
export WANDB_MODE=offline

python stablebaselines3_double_pendulum_le_sac_wandb.py -d <path to config.yml>
```

After training is finished, sync wandb to view training statistics.

```
wandb sync --sync-all
```

### Config.yml file

config.yml file is where all parameters are specified.

- config: parameters for experiments:
    - policy_type: policy networks for SAC
    - total_timesteps: the total time step for training
    - alpha: parameters for quadratic reward.
        - Pendulum: alpha1
        - Cart Pole: alpha1
        - Double Pendulum: alpha1, alpha2, alpha3
        - Lorenz: alpha1, alpha2
    - Learning_rate: specified learning rate
    - Gamma: specified gamma value
    - reward_type: specified reward types: 'precal', 'sparse', 'quadratic'
    - max: specify whether it is using LE max or Sum of Positive. True if max, False if sum of positive.
- wandb: parameters for recording wandb statistics:
    - project: specify project's name
    - tags: specify tags of runs for easy lookup


config.yaml example:

```
config:
  policy_type: MlpPolicy
  total_timesteps: 100000
  alpha:
    alpha1: 0.020051244039549988
  learning_rate: 0.00040404947251807303
  gamma: 0.9707213813656388
  reward_type: quadratic
  max: False

wandb:
  project: sb3_pendulum
  tags: [quadratic, test]
```
