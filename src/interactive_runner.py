import json
import os
import subprocess

# Directory for configurations
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(base_path, "configs")
DEFAULT_PARAMS_FILE = os.path.join(CONFIG_DIR, "params_file.json")

# Available commands and their expected parameters
COMMANDS = {
    "train": ["robot_name", "scene_path", "dis_parallel_mode", "no_gui", "params_file", "verbose"],
    "test": ["robot_name", "scene_path", "test_params", "verbose"],
    "sav": ["save_path", "compression", "overwrite"],
    "tf_start": ["log_dir"],
    "sat_training": ["robot_name", "sat_params_file", "learning_rate"]
}

def load_json(file_path):
    """Loads a JSON file if it exists, otherwise returns an empty dictionary."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    print(f"Warning: {file_path} not found. Using empty configuration.")
    return {}

def save_json(file_path, data):
    """Saves data to a JSON file, ensuring the directory exists."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def get_user_input(prompt, default=None, input_type=str):
    """Prompts the user for input, returning the correct data type."""
    while True:
        user_input = input(f"{prompt} [{default}]: ") or default
        if input_type == bool:
            return user_input.lower() in ["yes", "y", "true", "1"]
        try:
            return input_type(user_input)
        except ValueError:
            print(f"Invalid input. Please enter a valid {input_type.__name__} value.")

def handle_params_file():
    """Handles params_file input, allowing user to select an existing file or create a new one."""
    use_existing = get_user_input("Do you want to provide an existing params_file? (y/n)", "y", bool)
    if use_existing:
        return get_user_input("Enter the path to the params file", "configs/params_file.json")

    # Load default parameters
    default_params = load_json(DEFAULT_PARAMS_FILE)
    if not default_params:
        print("Error: No default parameters found. Please check configs/params_file_default.json")
        return None

    # Prompt user for manual parameter input
    params = {}
    for category, params_dict in default_params.items():
        print(f"\nConfiguring {category}:")
        params[category] = {}
        for param, default in params_dict.items():
            input_type = type(default)
            params[category][param] = get_user_input(f"Enter {param}", str(default), input_type)

    # Save custom parameters to a new file
    params_path = os.path.join(CONFIG_DIR, "custom_params.json")
    save_json(params_path, params)

    print(f"Custom parameters saved to {params_path}")
    return params_path

def main():
    print("=== Interactive Training Manager ===")

    # Display available functionalities
    print("\nAvailable functionalities:")
    for i, cmd in enumerate(COMMANDS.keys(), 1):
        print(f"{i}. {cmd}")

    # Select a command
    while True:
        try:
            choice = int(input("\nSelect the functionality to execute (number): ")) - 1
            command = list(COMMANDS.keys())[choice]
            break
        except (IndexError, ValueError):
            print("Invalid selection. Try again.")

    print(f"\nSelected: {command}")

    # Load previous configuration if available
    config_file = os.path.join(CONFIG_DIR, f"{command}.json")
    config = load_json(config_file)

    # Request parameters from the user
    for param in COMMANDS[command]:
        if param == "params_file":
            config[param] = handle_params_file()
            if not config[param]:  # If no params file was selected or created, exit
                print("Error: No parameters file provided. Exiting.")
                return
        else:
            default_value = config.get(param, "")
            input_type = bool if param in ["dis_parallel_mode", "no_gui", "overwrite"] else str
            config[param] = get_user_input(f"Enter {param}", default_value, input_type)

    # Save configuration for future use
    save_json(config_file, config)
    print(f"Configuration saved in {config_file}")

    # Build the command
    cmd_list = ["python", f"{command}.py"]
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd_list.append(f"--{key}")
        else:
            cmd_list.extend([f"--{key}", str(value)])

    # Execute the command
    print(f"\nExecuting: {' '.join(cmd_list)}")
    subprocess.run(cmd_list)

if __name__ == "__main__":
    main()