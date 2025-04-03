import logging
import os
from stable_baselines3 import SAC
import utils
from coppelia_envs import BurgerBotEnv, TurtleBotEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback



model_name = "/home/adrian/Documents/rl_coppelia/robots/turtleBot/models/turtleBot_model_2.zip"
# Cargar el modelo ya guardado
model = SAC.load(model_name)
print(f"RETRAIN MODEL {model_name}")

# Define comm_side
comm_side = "rl"

# Get root directory
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(base_path)

# Parse user arguments
args, warning_flags = utils.parse_args()

# Generate paths 
paths = utils.get_robot_paths(base_path, args.robot_name)

# Define log paths
train_log_path = paths["tf_logs"]
logs_path = paths["script_logs"]

# Get the next index for all the files that will be saved during the execution, so we can assign it as an ID to the execution
file_id = utils.get_file_index (train_log_path, args.robot_name)

# Initialization
utils.logging_config(logs_path, comm_side, args.robot_name, file_id, log_level=logging.INFO)

# Show possible warnings obtained during the parsing arguments function.
utils.show_warning_logs(warning_flags)

# Get params file path in case that we are in testing mode, as we need to use the same one that was used for training the model that we want to test.
if (args.testing_mode):
    args.params_file = utils.get_params_file(paths,args)

# Load the parameters defined by the user in params_file ('params_file.json' by default).
params_env, params_train, params_test = utils.load_params(args.params_file)

# Update the port that will be used for communications between agent and environment
comms_port = 49054
if args.parallel_training:
    comms_port = utils.find_next_free_port(comms_port)

# Start CoppeliaSim software, load scene and start the simulation
# Also, the code inside the scene will be automatically updated with the code of 'agent_coppelia_script.py'.
current_sim = utils.start_coppelia_and_simulation(base_path, args, params_env, comms_port)


base_env = TurtleBotEnv(params_env, comms_port=comms_port)

# Vectorize the environment
env = DummyVecEnv([lambda: base_env])

model.set_env(env)

# Continuar el entrenamiento
model.learn(total_timesteps=10000)  # Ajusta el número de timesteps según necesites


models_path = paths["models"]
callbacks_path = paths["callbacks"]
train_log_path = paths["tf_logs"]
training_metrics_path = paths["training_metrics"]
parameters_used_path = paths["parameters_used"]
train_log_file_path = os.path.join(train_log_path,f"{args.robot_name}_tflogs_{file_id}")

# TODO Check that for some reason, it starts sending and receiving resets and steps before next log ?? 
logging.info(f"Training mode. Final trained model will be saved in {models_path}")

# Callback function to save the model every x timesteps
to_save_callbacks_path, _ = utils.get_next_model_name(callbacks_path, args.robot_name, file_id, callback_mode=True)
checkpoint_callback = CheckpointCallback(save_freq=params_train['callback_frequency'], save_path=to_save_callbacks_path, name_prefix=args.robot_name)

# Start the training
model.learn(
    total_timesteps=params_train['total_timesteps'],
    callback=checkpoint_callback, 
    log_interval=1, # This is needed to assure that a tf.events file is generated even if thr training lasts few timesteps.
    tb_log_name=f"{args.robot_name}_tflogs"
    )

# Guardar el modelo actualizado si lo deseas
model.save("ruta/al/modelo_continuado.zip")