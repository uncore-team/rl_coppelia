import argparse
from rl_coppelia import auto_testing, auto_training, plot, sat_training, train, test, save, tf_start

def main():
    parser = argparse.ArgumentParser(prog="rl_coppelia", description="Training and testing CLI")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train a RL algorithm for robot movement in CoppeliaSim")
    train_parser.add_argument("--robot_name", type=str, help="Name for the robot.", required=True)
    train_parser.add_argument("--scene_path", type=str, help="Path to the CoppeliaSim scene file.", required=False)
    train_parser.add_argument("--dis_parallel_mode", action="store_true", help="Disables the parallel training or testing.", required=False)
    train_parser.add_argument("--no_gui", action="store_true", help="Disables Coppelia GUI, it will just show the terminal", required=False)
    train_parser.add_argument("--params_file", type=str, help="Path to the configuration file.",required=False)
    train_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. 1: just a progress bar. 2: all logs for debugging", default=0, required=False)

    test_parser = subparsers.add_parser("test", help="Test a trained RL algorithm for robot movement in CoppeliaSim")
    test_parser.add_argument("--model_name", type=str, help="If the inference/testing mode is enabled, a trained model is required (it must be located under 'models' folder)", required=True)
    test_parser.add_argument("--robot_name", type=str, help="Name for the robot. Default will be burgerBot.", required=False)
    test_parser.add_argument("--scene_path", type=str, help="Path to the CoppeliaSim scene file.", required=False)
    test_parser.add_argument("--dis_parallel_mode", action="store_true", help="Disables the parallel training or testing.", required=False)
    test_parser.add_argument("--no_gui", action="store_true", help="Disables Coppelia GUI, it will just show the terminal", required=False)
    test_parser.add_argument("--params_file", type=str, help="Path to the configuration file.",required=False)
    test_parser.add_argument("--iterations", type=int, help="Number of iterations for the test. If set, it will override the parameter from the parameters' json file.",required=False)
    test_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. 1: just a progress bar. 2: all logs for debugging", default=0, required=False)

    auto_training_parser = subparsers.add_parser("auto_training", help="Auto training of several models using different parameters pre-configured by using different configuration files.")
    auto_training_parser.add_argument("--session_name", type=str, help="Name for the session's folder.", required=True)
    auto_training_parser.add_argument("--robot_name", type=str, help="Name for the robot.", required=True)
    auto_training_parser.add_argument("--dis_parallel_mode", action="store_true", help="True if the user wants to disable the parallel execution and run the different trainings sequentially.", default=False)
    auto_training_parser.add_argument("--max_workers", type=int, help="Number of parallel processes if '--parallel_mode' flag is activated", default=3)
    auto_training_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. 1: just a progress bar. 2: all logs for debugging", default=0, required=False)

    auto_testing_parser = subparsers.add_parser("auto_testing", help="Auto testing of several models, saving the results of the comparision.")
    auto_testing_parser.add_argument("--robot_name", type=str, help="Name for the robot.", required=True)
    auto_testing_parser.add_argument("--model_ids", type=int, nargs='+', help="List with numerical IDs of the different models to be tested.", required=True)
    auto_testing_parser.add_argument("--iterations", type=int, help="Number of iterations for the test.", default=50, required=True)
    auto_testing_parser.add_argument("--dis_parallel_mode", action="store_true", help="True if the user wants to disable the parallel execution and run the different trainings sequentially.", default=False)
    auto_testing_parser.add_argument("--max_workers", type=int, help="Number of parallel processes if '--parallel_mode' flag is activated", default=3)
    auto_testing_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. 1: just a progress bar. 2: all logs for debugging", default=0, required=False)

    sampling_at_parser = subparsers.add_parser("sat_training", help="Auto training of several models modifying just the fixed action time from an unique configuration file.")
    sampling_at_parser.add_argument("--session_name", type=str, help="Name for the session's folder.", required=True)
    sampling_at_parser.add_argument("--robot_name", type=str, help="Name for the robot.", required=True)
    sampling_at_parser.add_argument("--base_params_file", type=str, help="Path to the base parameters file to modify.", required=True)
    sampling_at_parser.add_argument("--dis_parallel_mode", action="store_true", help="True if the user wants to disable the parallel execution and run the different trainings sequentially.", default=False)
    sampling_at_parser.add_argument("--max_workers", type=int, help="Number of parallel processes if '--parallel_mode' flag is activated", default=3)
    sampling_at_parser.add_argument("--start_value", type=float, help="Starting value for fixed_actime.", default=0.06)
    sampling_at_parser.add_argument("--end_value", type=float, help="Ending value for fixed_actime.", default=2.1)
    sampling_at_parser.add_argument("--increment", type=float, help="Increment value for fixed_actime.", default=0.01)
    sampling_at_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. 1: just a progress bar. 2: all logs for debugging", default=0, required=False)

    save_parser = subparsers.add_parser("save", help="Save a trained model, along with all the date generated during its training/testing processes")
    save_parser.add_argument("--model_name", type=str, help="Name of the model to be saved (it must be located under 'models' folder)", required=True)
    save_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. 1: just a progress bar. 2: all logs for debugging", default=0, required=False)

    tf_start_parser = subparsers.add_parser("tf_start", help="Starts the tensorboard to check the metrics generated during the training of a model.")
    tf_start_parser.add_argument("--model_name", type=str, help="Name of the model to be checked (it must be located under 'models' folder)", required=True)
    tf_start_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. 1: just a progress bar. 2: all logs for debugging", default=0, required=False)

    plot_parser = subparsers.add_parser("plot", help="Creates a set of plots for getting the results of a trained model or for comparing some models.")
    plot_parser.add_argument("--robot_name", type=str, help="Name for the robot.", required=True)
    plot_parser.add_argument("--model_ids", type=int, nargs='+', help="List with numerical IDs of the different models to be plotted.", required=True)
    plot_parser.add_argument("--plot_types", type=str, nargs='+', help="List of types of plots that the user wants to create.", default=["spider", "convergence", "compare-rewards", "compare-episodes_length"], required=False)
    plot_parser.add_argument("--verbose", type=int, help="Enable debugging through info logs using the terminal. 0: no logs at all. 1: just a progress bar. 2: all logs for debugging", default=0, required=False)


    args = parser.parse_args()

    if args.command == "train":
        train.main(args)
    elif args.command == "test":
        test.main(args)
    elif args.command == "auto_training":
        auto_training.main(args)
    elif args.command == "sat_training":
        sat_training.main(args)
    elif args.command == "save":
        save.main(args)
    elif args.command == "tf_start":
        tf_start.main(args)
    elif args.command == "auto_testing":
        auto_testing.main(args)
    elif args.command == "plot":
        plot.main(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()