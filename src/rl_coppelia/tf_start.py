import logging
import os
import re
import signal
import subprocess
import sys
import threading
import webbrowser
from common.rl_coppelia_manager import RLCoppeliaManager


# Start TensorBoard and capture its output
def run_tensorboard(curr_tf_logs_path):
    command = ["tensorboard", "--logdir", curr_tf_logs_path, "--port", "0"]  # Use port 0 to auto-select port
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Capture the output line by line
    for line in process.stderr:
        logging.debug(line.strip())
        if "http://localhost" in line:
            # Extract the port from the line
            port_match = re.search(r'http://localhost:(\d+)', line)
            if port_match:
                port = port_match.group(1)
                url = f'http://localhost:{port}'
                logging.info(f"TensorBoard is running on port {port}. Opening the browser.")
                webbrowser.open(url)
                break

    logging.info("Press 'q' and Enter to stop TensorBoard.")
    input()  # Wait for the user to press Enter to stop

    process.send_signal(signal.SIGINT)  # Gracefully stop TensorBoard
    
    # Wait for the process to finish (don't block forever)
    process.communicate()


def main(args):
    """
    Automates the process of launching TensorBoard for a given model and opening it in the browser.

    This function takes a model name of the form <robot_name>_model_<id>, extracts the robot name and 
    model ID, searches for the corresponding TensorBoard logs folder, and opens TensorBoard in the browser.

    Args:
        model_name (str): The model name in the format <robot_name>_model_<id> (e.g. "turtleBot_model_15").
    
    Raises:
        Exception: If the TensorBoard logs folder does not exist or there is an issue running TensorBoard.
    """
    rl_copp = RLCoppeliaManager(args)

    # Extract robot name and model id from the model_name
    try:
        parts = rl_copp.args.model_name.split('_')
        robot_name = parts[0]  # The first part is the robot name
        model_id = parts[2]  # The third part is the model ID
    except IndexError:
        logging.error(f"Error: Invalid model name format. Expected format is '<robot_name>_model_<id>', but got '{rl_copp.args.model_name}'")
        return
    
    # Define path
    tf_logs_path = rl_copp.paths["tf_logs"]
    curr_tf_logs_path = os.path.join(tf_logs_path, f'{robot_name}_tflogs_{model_id}')

    logging.info(f"Checking in this directory: {curr_tf_logs_path}")
    
    # Check if the folder exists
    if os.path.isdir(curr_tf_logs_path):
        logging.info(f"Found TensorBoard logs at: {curr_tf_logs_path}")
        
        # Run TensorBoard in a separate thread so that the program doesn't block
        tensorboard_thread = threading.Thread(target=run_tensorboard, args=(curr_tf_logs_path,))
        tensorboard_thread.start()
    else:
        logging.critical(f"Warning: Something failed while checking in this directory: {curr_tf_logs_path}. TensorBoard logs not found for model {model_id} of robot {robot_name}. Check your spelling!")
        sys.exit()


if __name__ == "__main__":
    main()