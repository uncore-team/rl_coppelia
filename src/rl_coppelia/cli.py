import os

from common import utils
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress TensorFlow warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
import argparse

def main(argv=None):
    """
    Entry point for the rl_coppelia CLI. Handles argument parsing and dispatches to the correct subcommand.

    Args:
        argv (list[str], optional): List of CLI arguments (for programmatic use). Defaults to None.
    """
    parser = argparse.ArgumentParser(prog="rl_coppelia", description="Training and testing CLI")
    subparsers = parser.add_subparsers(dest="command")

    create_robot_parser = subparsers.add_parser("create_robot", help="Create an environment and its corresponding agent for CoppeliaSim simulation.")

    train_parser = subparsers.add_parser("train", help="Train a RL algorithm for robot movement in CoppeliaSim.")
    train_parser.add_argument("--robot_name", type=str, help="Name for the robot.", required=True)
    train_parser.add_argument("--scene_path", type=str, help="Path to the CoppeliaSim scene file.", required=False)
    train_parser.add_argument("--dis_parallel_mode", action="store_true", help="Disables the parallel training or testing.", required=False)
    train_parser.add_argument("--no_gui", action="store_true", help="Disables Coppelia GUI, it will just show the terminal", required=False)
    train_parser.add_argument("--params_file", type=str, help="Path to the configuration file.",required=False)
    train_parser.add_argument("--obstacles_csv_folder", type=str, help="Path to scene configuration folder in case that we want to train with fixed obstacles. Please just indicate the folder not the whole path (e.g. /Scene014)",required=False)
    train_parser.add_argument("--dis_save_notes", action="store_true", help="Flag to save some notes for the experiment.", default = False, required=False)
    train_parser.add_argument("--rl_side", action="store_true", help="Flag to just execute rl side code.", default = False, required=False)
    train_parser.add_argument("--agent_side", action="store_true", help="Flag to just execute agent side code.", default = False, required=False)
    train_parser.add_argument("--timestamp", type=str, help="Timestamp provided externally (e.g., from GUI).", required=False)
    train_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. \
                             1: just a progress bar and save warnings. 2: just a progress bar and save everything. 3: all logs shown and saved for debugging. Other: just terminal, logs are not saved", default=0, required=False)

    test_parser = subparsers.add_parser("test", help="Test a trained RL algorithm for robot movement in CoppeliaSim.")
    test_parser.add_argument("--model_name", type=str, help="Name of the trained model is required (it must be located under 'models' folder)", required=True)
    test_parser.add_argument("--robot_name", type=str, help="Name for the robot. Default will be burgerBot.", required=False)
    test_parser.add_argument("--scene_path", type=str, help="Path to the CoppeliaSim scene file.", required=False)
    test_parser.add_argument("--save_scene", action="store_true", help="Enables saving scene mode.", required=False, default=False)
    test_parser.add_argument("--save_traj", action="store_true", help="Enables saving trajectory mode.", required=False, default=False)
    test_parser.add_argument("--obstacles_csv_folder", type=str, help="Path to scene configuration folder in case that we want to test with fixed obstacles. Please just indicate the folder not the whole path (e.g. /Scene014)",required=False)
    test_parser.add_argument("--dis_parallel_mode", action="store_true", help="Disables the parallel training or testing.", required=False)
    test_parser.add_argument("--no_gui", action="store_true", help="Disables Coppelia GUI, it will just show the terminal", required=False)
    test_parser.add_argument("--params_file", type=str, help="Path to the configuration file.",required=False)
    test_parser.add_argument("--iterations", type=int, help="Number of iterations for the test. If set, it will override the parameter from the parameters' json file.",required=False)
    test_parser.add_argument("--dis_save_notes", action="store_true", help="Flag to save some notes for the experiment.", default = False, required=False)
    test_parser.add_argument("--timestamp", type=str, help="Timestamp provided externally (e.g., from GUI).", required=False)
    test_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. \
                             1: just a progress bar and save warnings. 2: just a progress bar and save everything. 3: all logs shown and saved for debugging. Other: just terminal, logs are not saved", default=0, required=False)

    test_scene_parser = subparsers.add_parser("test_scene", help="Test a trained RL algorithm for robot movement in CoppeliaSim for just one iteration, using a preconfigured scene.")
    test_scene_parser.add_argument("--model_ids", type=int, nargs='+', help="List with numerical IDs of the different models to be plotted. They must be located inside 'models' folder. Program will take the '_last' one", required=True)
    test_scene_parser.add_argument("--scene_to_load_folder", type=str, help="Folder name, located inside 'scene_configs', that contains the scene be loaded", required=True)
    test_scene_parser.add_argument("--robot_name", type=str, help="Name for the robot.", required=True)
    test_scene_parser.add_argument("--iters_per_model", type=int, help="Number of iterations for testing each model.",required=False, default=1)
    test_scene_parser.add_argument("--scene_path", type=str, help="Path to the CoppeliaSim scene file.", required=False)
    test_scene_parser.add_argument("--dis_parallel_mode", action="store_true", help="Disables the parallel training or testing.", required=False)
    test_scene_parser.add_argument("--no_gui", action="store_true", help="Disables Coppelia GUI, it will just show the terminal", required=False)
    test_scene_parser.add_argument("--params_file", type=str, help="Path to the configuration file.",required=False)
    test_scene_parser.add_argument("--timestamp", type=str, help="Timestamp provided externally (e.g., from GUI).", required=False)
    test_scene_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. \
                             1: just a progress bar and save warnings. 2: just a progress bar and save everything. 3: all logs shown and saved for debugging. Other: just terminal, logs are not saved", default=0, required=False)

    auto_training_parser = subparsers.add_parser("auto_training", help="Auto training of several models using different parameters pre-configured by using different configuration files.")
    auto_training_parser.add_argument("--session_name", type=str, help="Name for the session's folder.", required=True)
    auto_training_parser.add_argument("--robot_name", type=str, help="Name for the robot.", required=True)
    auto_training_parser.add_argument("--dis_parallel_mode", action="store_true", help="True if the user wants to disable the parallel execution and run the different trainings sequentially.", default=False)
    auto_training_parser.add_argument("--max_workers", type=int, help="Number of parallel processes if '--parallel_mode' flag is activated", default=3)
    auto_training_parser.add_argument("--timestamp", type=str, help="Timestamp provided externally (e.g., from GUI).", required=False)
    auto_training_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. \
                             1: just a progress bar and save warnings. 2: just a progress bar and save everything. 3: all logs shown and saved for debugging. Other: just terminal, logs are not saved", default=0, required=False)

    auto_testing_parser = subparsers.add_parser("auto_testing", help="Auto testing of several models, saving the results of the comparision.")
    auto_testing_parser.add_argument("--session_name", type=str, help="Name for the testing session.", required=True)
    auto_testing_parser.add_argument("--robot_name", type=str, help="Name for the robot.", required=True)
    auto_testing_parser.add_argument("--model_ids", type=int, nargs='+', help="List with numerical IDs of the different models to be tested.", required=True)
    auto_testing_parser.add_argument("--iterations", type=int, help="Number of iterations for the test.", default=200, required=True)
    auto_testing_parser.add_argument("--dis_parallel_mode", action="store_true", help="True if the user wants to disable the parallel execution and run the different trainings sequentially.", default=False)
    auto_testing_parser.add_argument("--max_workers", type=int, help="Number of parallel processes if '--parallel_mode' flag is activated", default=3)
    auto_testing_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. \
                             1: just a progress bar and save warnings. 2: just a progress bar and save everything. 3: all logs shown and saved for debugging. Other: just terminal, logs are not saved", default=0, required=False)

    sampling_at_parser = subparsers.add_parser("sat_training", help="Auto training of several models modifying just the fixed action time from an unique configuration file.")
    sampling_at_parser.add_argument("--session_name", type=str, help="Name for the session's folder.", required=True)
    sampling_at_parser.add_argument("--robot_name", type=str, help="Name for the robot.", required=True)
    sampling_at_parser.add_argument("--base_params_file", type=str, help="Path to the base parameters file to modify.", required=True)
    sampling_at_parser.add_argument("--dis_parallel_mode", action="store_true", help="True if the user wants to disable the parallel execution and run the different trainings sequentially.", default=False)
    sampling_at_parser.add_argument("--max_workers", type=int, help="Number of parallel processes if '--parallel_mode' flag is activated", default=3)
    sampling_at_parser.add_argument("--start_value", type=float, help="Starting value for fixed_actime.", default=0.06)
    sampling_at_parser.add_argument("--end_value", type=float, help="Ending value for fixed_actime.", default=2.1)
    sampling_at_parser.add_argument("--increment", type=float, help="Increment value for fixed_actime.", default=0.01)
    sampling_at_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. \
                             1: just a progress bar and save warnings. 2: just a progress bar and save everything. 3: all logs shown and saved for debugging. Other: just terminal, logs are not saved", default=0, required=False)

    save_parser = subparsers.add_parser("save", help="Save a trained model, along with all the date generated during its training/testing processes.")
    save_parser.add_argument("--model_name", type=str, help="Name of the model to be saved (it must be located under 'models' folder)", required=True)
    save_parser.add_argument("--new_name", type=str, help="New name for saving the model", required=True)
    save_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. \
                             1: just a progress bar and save warnings. 2: just a progress bar and save everything. 3: all logs shown and saved for debugging. Other: just terminal, logs are not saved", default=-1, required=False)

    tf_start_parser = subparsers.add_parser("tf_start", help="Starts the tensorboard to check the metrics generated during the training of a model.")
    tf_start_parser.add_argument("--model_name", type=str, help="Name of the model to be checked (it must be located under 'models' folder)", required=True)
    tf_start_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. \
                             1: just a progress bar and save warnings. 2: just a progress bar and save everything. 3: all logs shown and saved for debugging. Other: just terminal, logs are not saved", default=-1, required=False)

    plot_parser = subparsers.add_parser("plot", help="Creates a set of plots for getting the results of a trained model or for comparing some models.")
    plot_parser.add_argument("--robot_name", type=str, help="Name for the robot.", required=True)
    plot_parser.add_argument("--plot_types", type=str, nargs='+', help="List of types of plots that the user wants to create.", required=True)
    plot_parser.add_argument("--model_ids", type=int, nargs='+', help="List with numerical IDs of the different models to be plotted.", required=False)
    plot_parser.add_argument("--scene_to_load_folder", type=str, help="Folder name, located inside 'scene_configs', that contains the scene and trajectories to be loaded", required=False)
    plot_parser.add_argument("--save_plots", action="store_true", help="Saves the plots inside current folder instead of showing them.", required=False, default=False)
    plot_parser.add_argument("--lat_fixed_timestep", type=float, help="Fixed timestep for LAT plots (optional).", default=0, required=False)
    plot_parser.add_argument("--timestep_unit", type=str, help="Unit for timestep for LAT plots (optional).", default="s", required=False)
    plot_parser.add_argument("--csv_file_path", type=str, help="Path to a specific CSV file, e.g. LAT file (optional).", required=False)
    plot_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. \
                             1: just a progress bar and save warnings. 2: just a progress bar and save everything. 3: all logs shown and saved for debugging. Other: just terminal, logs are not saved", default=-1, required=False)

    retrain_parser = subparsers.add_parser("retrain", help="Retrain a pretrained RL algorithm for robot movement in CoppeliaSim.")
    retrain_parser.add_argument("--model_name", type=str, help="Name of the trained model is required (it must be located under 'models' folder)", required=True)
    retrain_parser.add_argument("--retrain_steps", type=int, help="Number of steps for the retraining. Default = 50.000",required=True, default=50000)
    retrain_parser.add_argument("--scene_path", type=str, help="Path to the CoppeliaSim scene file.", required=False)
    retrain_parser.add_argument("--dis_parallel_mode", action="store_true", help="Disables the parallel training or testing.", required=False)
    retrain_parser.add_argument("--no_gui", action="store_true", help="Disables Coppelia GUI, it will just show the terminal", required=False)
    retrain_parser.add_argument("--params_file", type=str, help="Path to the configuration file. It's not recommended to use a different one from the one used for the previous training",required=False)
    retrain_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. \
                             1: just a progress bar and save warnings. 2: just a progress bar and save everything. 3: all logs shown and saved for debugging. Other: just terminal, logs are not saved", default=0, required=False)

    args = parser.parse_args(argv)  # Parse CLI arguments (from sys.argv or passed manually)

    # Print UnCORE logo
    utils.print_uncore_logo()

    if args.command == "train":
        from rl_coppelia import train
        train.main(args)
    elif args.command == "test":
        from rl_coppelia import test
        test.main(args)
    elif args.command == "auto_training":
        from rl_coppelia import auto_training
        auto_training.main(args)
    elif args.command == "sat_training":
        from rl_coppelia import sat_training
        sat_training.main(args)
    elif args.command == "save":
        from rl_coppelia import save
        save.main(args)
    elif args.command == "tf_start":
        from rl_coppelia import tf_start
        tf_start.main(args)
    elif args.command == "auto_testing":
        from rl_coppelia import auto_testing
        auto_testing.main(args)
    elif args.command == "plot":
        from rl_coppelia import plot
        plot.main(args)
    elif args.command == "retrain":
        from rl_coppelia import retrain
        retrain.main(args)
    elif args.command == "test_scene":
        from rl_coppelia import test_scene
        test_scene.main(args)
    elif args.command == "create_robot":
        from rl_coppelia import create_robot
        create_robot.main()
    else:
        parser.print_help() # Show help if no command provided

if __name__ == "__main__":
    main()