# Training and Testing of RL algorithms for controlling robots' movements in CoppeliaSim

This project allows for either training models for robots in a CoppeliaSim environment using different Reinforcement Learning algorithms, or testing those pre-trained models. It automates the process of launching the CoppeliaSim software, loading the respective scene, and executing training or testing experiments for a given robot. The project is organized into several key directories:

- **`configs`**: Contains json files with configuration's parameters for the environment and the training/testing processes. By default, you will find a model file, which you can modify and make copies of to place in this same folder. Please do not delete the model file 'params_file.json'.
- **`dependencies`**: Contains external libraries or modules that the project depends on and that are not installable by pip. Specifically, here you will find the 'rl_spin_decoupler' package.
- **`src`**: Contains Python scripts for managing the environment, the processes of training and testing, and the interaction with the CoppeliaSim simulator. Here you will find two subfolders:
    - **`common`**: 
        - **`agent_copp.py`**: This script is not directly executable but is intended to update some of its variables whenever a training/testing process starts. Then, it's content is directly copied inside the selected Coppelia scene, so the user does not need to make any manual changes in the 'Agent_Script' inside Coppelia scenes.
        - **`coppelia_agents.py`**: Definition of CoppeliaAgent class for managing the interaction of the agent with CoppeliaSim simulation. If you want to add a new robot, just copy one of the current subclasses (e.g., BurgerBotAgent, TurtleBotAgent), and modify their key handles to adapt the robot to your scene.
        - **`coppelia_envs.py`**: Definition of CoppeliaEnv class for managing the interaction of the environments with CoppeliaSim simulator. Here you will find the step, reset, and calculate-reward functions. If you want to add a new robot, just copy one of the current subclasses (e.g., BurgerBotEnv, TurtleBotEnv). It's important to adapt ther action and observation spaces to your needs.
        - **`rl_coppelia_manager.py`**: Definition of the RLCoppeliaManager class, which is responsible for managing some of the initial processes of every core functionality of this project. It manages the creation of the environment, and the start/stop of the simulations.
        - **`utils.py`**: Utility scripts used across various processes. 
    - **`rl_coppelia.py`**: Core functionalities of the project.
        - **`cli.py`**: Handles command-line interface (CLI) functionality for all the possible processes of this project.
        - **`train.py`**: Manages the model training process.
        - **`test.py`**: Runs model evaluation on test data and logs performance metrics.
        - **`save.py`**: Saves the trained model and its associated data to a zip file inside the 'results' folder.
        - **`tf_start.py`**: Starts TensorFlow training logs and visualizations, handling TensorBoard. It automatically opens your browser using the right web address and port.
        - **`plot.py`**: Generates plots for visualizing diffeerent comparisons of training and testing metrics.
        - **`auto_training.py`**: Automates the training process, managing hyperparameter tuning. It is only necessary to have the different configurations to be tested (each in a json file of the type 'params_file') inside the subfolder robots/<robot_name>/auto_trainings/<session_name>.
        - **`sat_training.py`**: Automates the training process jsut changing one specific parameter, the action time. This makes possible to easily find the optimal action time for a specific setup.
        - **`auto_testing.py`**: Automates the testing of models. __Not tested yet__
        - **`retrain.py`**: It allows the user to continue the training of a pretrained model. __Not tested yet__
- **`scenes`**: It contains CoppeliaSim scene files (.ttt format) for each robot/task. __Important:__ The scene name must be <robot_name>_scene.ttt. 
- **`robots`**: Stores the data related to each robot, including model and callback files, tensorboard logs and generated metrics. All the subfolders inside 'robots' folder will be automatically created whenever you called the 'train' functionality for a new robot for the first time. This will be the structure that would be created for a robot called burgerBot:
```
    ├── robots/
      └── burgerBot/
          ├── auto_trainings/
          ├── callbacks/
          ├── models/
          ├── parameters_used/
          ├── sat_trainings/
          ├── script_logs/
          ├── testing_metrics/
          ├── tf_logs/
          └── training_metrics/
```
- **`results`**: Each zip file generated with the 'save' functionality will be saved here.

## Overview

The primary goal of this project is to facilitate robot training or testing models in the CoppeliaSim simulator by interacting with the robot through Python scripts. The training or testing process will automatically start CoppeliaSim, load a specific scene, and initiate training using a provided robot model.

Any new robot name used in the `main.py` script will generate a new folder within the `robots` directory. This folder contains all the data generated during training or testing for that robot, including logs, models, and any additional outputs.

## Notes Before Running anything

- **Note 1**: You need to clone this project with its submodule, `rl_spin_decoupler`, which is a repository located at https://github.com/uncore-team/rl_spin_decoupler.git. For doing that, please clone the repository and initialize the submodule (just the first time) using the next commands:

```bash
git clone --recurse-submodules git@github.com:uncore-team/rl_coppelia.git
```

```bash
git submodule init
```

```bash
git submodule update
```

At this point, the repository and it's submodule should be correctly cloned.

- **Note 2**: When running train/test functionalities, the content inside the `agent_copp` will be copied into the `Agent_Script` code section of the CoppeliaSim scene automatically. Just keep it in mind in case you need to make a backup of your scene.

## Installation

Before using this project, ensure that the following dependencies are installed:

- **Python 3.x** (preferably 3.6 or later). It has been tested with python 3.8.10.
- **CoppeliaSim**: The simulator must be installed and configured correctly for the project to work. The project has been tested with CoppeliaSim Edu v4.9.0 (rev. 6) 64bit.

To install the required Python libraries, you can directly use the `install.sh` file included in the root directory of the project. This will also add the rl_spin_decoupler package to the path:

```bash
chmod +x install.sh
```

```bash
./install.sh
```

```bash
source ~/.bashrc
```

Last command is important to refresh the changes made in the path. After that, don't forget to activate again your virtualenv (in case you were using one).

Everything should be already installed, including the `rl_coppelia` package. In fact, it is installed in editable mode (-e), so any changes you make in the code will be automatically reflected without needing to reinstall the package.

From now on, you will need to operate from the `src` subfolder.

## Running the Script

To start training a model for a robot, execute the train option of the `rl_coppelia` package. You do not need to have CoppeliaSim opened, in fact, by default a new instance of the programm will be opened if you do not use set the `dis_parallel_mode` to True. 

```bash
rl_coppelia train --robot_name turtleBot --verbose 2
```

- **`--robot_name`**: The name of the robot you wish to train or test for. This will create a folder for the robot in the `robots` directory. If no 'robot_name' is provided, it will be 'burgerBot' by default.

- **`--verbose`**: Level of verbosity. For your first steps with this repository, it's recommended to set it to 2, so you can check all the logs generated during the process.

For the training, as well as for creating the environment and for testing any model, there are some parameters needed which are assigned within the `configs/params_file.json` file. The user can replicate this file and change the parameters' values, and then use the argument `--params_file` indicating the absolute or relative path of the new json file (it's recommended to keep them in the same `configs` folder).

In case of having any trained model (for example, a model called `burgerBot_model_15`), the user can test it using the next command:

```bash
rl_coppelia test --model_name burgerBot_model_15
```

If you have any questions about the possible input arguments for any functionality (e.g., `plot`), please refer to the help option for more information.

```bash
rl_coppelia plot -h
```
