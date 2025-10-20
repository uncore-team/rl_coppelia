# Training and Testing of RL Algorithms for robots in CoppeliaSim

This project provides a complete framework to train and test reinforcement learning (RL) policies for mobile robots in CoppeliaSim. It automates launching CoppeliaSim, loading the right scene, and managing training/testing loops from Python. 


## 📁 Project Structure

The project is organized into several key directories:

- **`configs`**: JSON files with configuration parameters for the environment and the training/testing processes. By default, you will find a model file, which you can modify and make copies of to place in this same folder. Please do not delete the model file `params_default_file.json`. Also, every time you create a new robot using `create_robot`, a new default parameters file will be created for that new robot.
- **`dependencies`**: External libraries or modules not installable via pip. In particular, it includes the `rl_spin_decoupler` package (added as a submodule).
- **`src`**: Python source code. It contains:
    - **`common`**: 
        - **`coppelia_agents.py`**: Definition of CoppeliaAgent class for managing the interaction of the agent with CoppeliaSim simulation. If you want to add a new robot, just copy one of the current subclasses (e.g., BurgerBotAgent, TurtleBotAgent), and modify their key handles to adapt the robot to your scene. If you run `create_robot`, the child class will be created automatically.
        - **`coppelia_envs.py`**: Definition of CoppeliaEnv class for managing the interaction of the environments with CoppeliaSim simulator. Here you will find the step, reset, and calculate-reward functions. If you want to add a new robot, just copy one of the current subclasses (e.g., BurgerBotEnv, TurtleBotEnv). It's important to adapt their action and observation spaces to your needs. Again, this will be automatically done running the script for creating a new robot.
        - **`rl_coppelia_manager.py`**: Definition of the RLCoppeliaManager class, which is responsible for managing some of the initial processes of every core functionality of this project. It manages the creation of the environment, and the start/stop of the simulations.
        - **`utils.py`**: Utility scripts used across various processes. 
        - **`robot_generator.py`**: Helpers used by the interactive robot creator.
    - **`coppelia_scripts`**:
        - **`rl_script_copp.py`**: Not directly executable. Its variables are updated at the start of the experiment and its content is automatically copied into a script within the CoppeliaSim scene called 'Agent_Script'. This way you don’t need to edit the scene manually. This script is responsible for the communication between the Python RL agent and the simulation, as well as managing the reinforcement learning loop.
        - **`robot_script_copp.py`**: Similarly to the previous file, it is automatically copied into a CoppeliaSim scene script called `Robot_Script`. This script contains functions for controlling the robot and the scene.
    - **`plugins`**: `envs`and `agents` are lightweight plugin registries. Add <robot>.py modules that call register_env(...) / register_agent(...) to plug any robot in cleanly. These files are automatically managed by the `create_robot`functionality.
    - **`rl_coppelia.py`**: Core functionalities of the project. It contains the core entry points for command-line usage:
        - **`cli.py`**: Handles command-line interface (CLI) functionality for all the possible processes of this project.
        - **`train.py`**: Manages the model training process.
        - **`test.py`**: Runs model evaluation and logs performance metrics.
        - **`save.py`**: Saves the trained model and its associated data to a zip file inside the 'results' folder.
        - **`tf_start.py`**: Starts TensorFlow training logs and visualizations, handling TensorBoard. It automatically opens your browser using the right web address and port.
        - **`plot.py`**: Generates plots for visualizing different comparisons of training and testing metrics.
        - **`auto_training.py`**: Automates the training process, managing hyperparameter tuning. It is only necessary to have the different configurations to be tested (each in a json file of the type 'params_file') inside the subfolder robots/<robot_name>/auto_trainings/<session_name>.
        - **`sat_training.py`**: Automates the training process just changing one specific parameter, the action time. This makes possible to easily find the optimal action time for a specific setup. __Not tested yet__
        - **`auto_testing.py`**: Automates the testing of models. __Not tested yet__
        - **`retrain.py`**: It allows the user to continue the training of a pretrained model. __Not tested yet__
- **`scenes`**: It contains CoppeliaSim scene files (.ttt format) for each robot/task.
- **`robots`**: Stores the data related to each robot, including model and callback files, tensorboard logs and generated metrics. All the subfolders inside 'robots' folder will be automatically created whenever you called the 'train' functionality for a new robot for the first time. This will be the structure that would be created for a robot called burgerBot:
```
    ├── robots/
      └── burgerBot/
          ├── auto_trainings/
          ├── callbacks/
          ├── models/
          ├── parameters_used/
          ├── sat_trainings/
          ├── scene_configs/
          ├── script_logs/
          ├── testing_metrics/
          ├── tf_logs/
          └── training_metrics/
```
- **`results`**: Each zip file generated with the `save` functionality will be saved here.
- **`gui`**: A PyQt5 GUI with tabs for Train, Test, Auto-Train, SAT-Train (action-time sweep), Test Scene, Plot, and Manage (import/export, cleanup). __Not fully tested yet__

## 🎯 Overview

The primary goal of this project is to facilitate robot training or testing models in the CoppeliaSim simulator by interacting with the robot through Python scripts. The training or testing process will automatically start CoppeliaSim, load a specific scene, and initiate training using a provided robot model.

Any new robot name used in the `main.py` script will generate a new folder within the `robots` directory. This folder contains all the data generated during training or testing for that robot, including logs, models, and any additional outputs.

## ⚙️ Notes Before Running anything

- **Note 1**: You need to clone this project with its submodule, `rl_spin_decoupler`, which is a repository located at https://github.com/uncore-team/rl_spin_decoupler.git. For doing that, please clone the repository using the next command:

```bash
git clone --recurse-submodules git@github.com:uncore-team/rl_coppelia.git
```

If you already cloned it before reading these instructions (don't worry, I never read them neither), please use the next commands:

```bash
git submodule init
git submodule update
```

At this point, the repository and it's submodule should be correctly cloned.

- **Note 2**: When running train/test functionalities, the content inside `rl_script_copp` and `agent_script_copp` will be copied into the `Robot_Script` and `Agent_Script` CoppeliaSim scripts, respectively. Just keep it in mind in case you need to make a backup of your scene.

## 🧩 Installation

Before using this project, ensure that the following dependencies are installed:

- **Python 3.x** (preferably 3.6 or later). It has been tested with python 3.8.10.
- **CoppeliaSim**: The simulator must be installed and configured correctly for the project to work. The project has been tested with CoppeliaSim Edu v4.9.0 (rev. 6) 64bit.

To install the required Python libraries, you can directly use the `install.sh` file included in the root directory of the project. This will also add the rl_spin_decoupler package to the path:

```bash
chmod +x install.sh
./install.sh
source ~/.bashrc
```

Last command is important to refresh the changes made in the path. After that, don't forget to activate again your virtualenv (in case you were using one).

Everything should be already installed, including the `rl_coppelia` package. In fact, it is installed in editable mode (-e), so any changes you make in the code will be automatically reflected without needing to reinstall the package.

From now on, you will need to operate from the `src` subfolder.

## 🚀 Usage

To start training a model for a robot, execute the train option of the `uncore_rl` package. You do not need to have CoppeliaSim opened, as a new instance of the program will be opened if you do not set the `dis_parallel_mode` to True. 

```bash
uncore_rl train --robot_name turtleBot --verbose 2
```
**Key arguments**:
- **`--robot_name`** (required): The name of the robot you wish to train or test for. This will create a folder for the robot in the `robots` directory. 
- **`--params_file`**: Name of the custom JSON with training/env parameters (defaults to configs/params_default_file_\<robot\>.json).
- **`--scene_path`**: Name of the CoppeliaSim scene used for the experiment (defaults to scenes/\<robot\>_scene.ttt)
- **`--no_gui`**: Flag to disable CoppeliaSim GUI. It's recommended for trainings.
- **`--dis_parallel_mode`**: Force to use an already opened CoppeliaSim scene. 
- **`--timestamp`**: External argument used by the GUI to name the terminal of the experiment.  
- **`--verbose`**: Level of verbosity. For your first steps with this repository, it's recommended to set it to 3, so you can check all the logs generated during the process.

For the training, as well as for creating the environment and for testing any model, there are some parameters needed which are assigned within the `configs/params_default_file_\<robot\>.json` file. The user can replicate this file and change the parameters values, and then use the argument `--params_file` indicating the absolute or relative path of the new json file (it's recommended to keep them in the same `configs` folder).

After every training, two models will be saved: the last and the one with the best training reward obtained. Moreover, you will have callbacks available every 10K steps (with the default parameters). In case that you want to test the last model saved for an specific experiment (for example, a model called `burgerBot_model_15`), the user can test it using the next command:

```bash
uncore_rl test --model_name burgerBot_model_15/burgerBot_model_15_last
```

If you need to check the possible key arguments for any functionality (e.g., `test`), please refer to the help option for more information.

```bash
uncore_rl test -h
```

## 🖥️ Graphical Interface (GUI)

The project also includes a **PyQt5-based GUI** that allows users to interact with the same core functionalities through a graphical interface.  
It includes tabs for **Training**, **Testing**, **Auto-Training**, **SAT-Training**, **Test Scene**, **Plotting**, and **Manage** (import/export, cleanup).

To launch the GUI, run:

```bash
uncore_rl_gui
```

This interface allows you to configure experiments, run and monitor training/testing processes, and visualize logs in real time. ⚠️ **Note:** The GUI is still under active development, and some features may not be fully tested yet.