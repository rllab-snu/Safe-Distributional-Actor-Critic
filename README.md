# Trust Region-Based Safe Distributional Reinforcement Learning for Multiple Constraints

This is an official GitHub Repository for the following paper:
- Dohyeong Kim, Kyungjae Lee, and Songhwai Oh, "Trust Region-Based Safe Distributional Reinforcement Learning for Multiple Constraints," in Proc. of Neural Information Processing Systems (NeurIPS), Dec. 2023.

## Requirement

### 1. Install learning-related modules

⚠️ As `stable-baselines3` causes the `torch` installation to be incorrect, we recommend to install `stable-baselines3` and `sb3-contrib` first, and then `torch`.

- python 3.8 or greater
- stable-baselines3
- sb3-contrib
- torch==1.12.1
- wandb (Optional, just for logging)
- scipy
- qpsolvers==1.9.0
- opencv-python
- tensorflow-gpu==2.5.0 (Optional, for `OffTRC`, `CPO`, and `WCSAC`)
- tensorflow-probability==0.12.2 (Optional, for `OffTRC`, `CPO`, and `WCSAC`)
- tqdm (Optional, for `CVPO`)
- tensorboardX>=2.4 (Optional, for `CVPO`)
- cpprb==10.1.1 (Optional, for `CVPO`)
- mpi4py (Optional, for `WCSAC`)
- numpy==1.22

### 2. Install Safety Gym environment

1. Install `mujoco-py`: 
    - You can refer to [here](https://github.com/openai/mujoco-py).
2. Install `safety-gym`:
    - The official repository has some issues, so we recommend to install it as follows.
    - ```bash
        mv {sdac}/installation/safety-gym
        pip install -e .
        ```

### 3. Install WCSAC (Optional, if you want to run `WCSAC`)

- The official repository supports only `tensorflow 1.XX`, so to use `tensorflow 2.XX`, we recommend to install it as follows.
- ```bash
    mv {sdac}/installation/WCSAC
    pip install -e .
    ```

## Supported environment list

**Safety Gym**
- `Safexp-PointGoal1-v0`
- `Safexp-CarGoal1-v0`
- `Safexp-PointButton3-v0` (defined in `safety_gym/utils/register.py`)
- `Safexp-CarButton3-v0` (defined in `safety_gym/utils/register.py`)

**Locomotion**
- `MITCheetah-v0` and `MITCheetah-v1` (defined in `locomotion/utils/register.py`)
- `Laikago-v0` and `Laikago-v1` (defined in `locomotion/utils/register.py`)
- `Cassie-v0` and `Cassie-v1` (defined in `locomotion/utils/register.py`)

## How to train and test

### **Safety Gym**

1. `SDAC`
    - The constraint conservativeness $\alpha$ can be set by modifying the part corresponding to `--cost_alpha {float_number}` in each shell file.
    - ```bash
      # for train
      cd {sdac}/safety_gym/sdac
      bash train/{env_name}.sh # env_name: point_goal, point_button, car_goal, car_button.
      ```
    - ```bash
      # for test
      cd {sdac}/safety_gym/sdac
      bash test/{env_name}.sh # env_name: point_goal, point_button, car_goal, car_button.
      ```
2. `OffTRC` and `CPO`
    - The constraint conservativeness $\alpha$ for `OffTRC` can be set by modifying the part corresponding to `--cost_alpha {float_number}` in each shell file (For `CPO`, $\alpha$ should be fixed at $1.0$).
    - The source code is from https://github.com/rllab-snu/Off-Policy-TRC.
    - ```bash
      # for train
      cd {sdac}/safety_gym/offtrc
      bash train/{algo_name}_{env_name}.sh # algo_name: offtrc, cpo.
      ```
    - ```bash
      # for test
      cd {sdac}/safety_gym/offtrc
      bash test/{algo_name}_{env_name}.sh # algo_name: offtrc, cpo.
      ```
3. `CVPO`
    - The source code is from https://github.com/liuzuxin/cvpo-safe-rl.
    - See `{cvpo}/safety_gym/cvpo/README.md` for detailed configuration information.
    - ```bash
      # for train
      cd {sdac}/safety_gym/cvpo
      bash train/{env_name}.sh
      ```
4. `WCSAC`
    - The source code is from https://github.com/AlgTUDelft/WCSAC.
    - The constraint conservativeness $\alpha$ can be set by modifying the part corresponding to `--cl {float_number}` in each shell file.
    - ```bash
      # for train
      cd {sdac}/safety_gym/cvpo
      bash train/{env_name}.sh
      ```

### **Locomotion**
1. `SDAC`, `WCSAC`, and `OffTRC`
    - ```bash
      # for train
      cd {sdac}/locomotion/{algo_name} # algo_name: sdac, wcsac, offtrc
      bash train/{env_name}.sh # env_name: cheetah, laikago, cassie
      ```
    - ```bash
      # for test
      cd {sdac}/locomotion/{algo_name} # algo_name: sdac, wcsac, offtrc
      bash test/{env_name}.sh # env_name: cheetah, laikago, cassie
      ```

## Training curve

All algorithms leave log files using `{sdac}/safety_gym/utils/logger.py`.

To draw graph using the log files, you can run `visualize.py` in each algorithm directory.

For example, `WCSAC`:
```bash
cd {sdac}/safety_gym/wcsac
python integrate.py
python visualize.py
```
`SDAC`, `CPO`, `OffTRC`, and `CVPO`:
```bash
cd {sdac}/safety_gym/{algo_name}
python visualize.py
```


After run the python file, the figure file will be saved in the `imgs` folder.

In the `visualize.py`, you can modify the path of where the logs are saved.

## License

Distributed under the MIT License. See `LICENSE` for more information.
