"""
Project: Robot Training and Testing RL Algorithms in CoppeliaSim
Author: Adrián Bañuls Arias
Version: 1.0
Date: 2025-03-25
License: GNU General Public License v3.0

Description:
    This script generates various types of plots to analyze the performance of trained reinforcement 
    learning models in a CoppeliaSim environment. It allows for visual comparison of multiple models 
    using metrics such as reward evolution, convergence points, episode efficiency, and LAT behavior.

Usage:
    rl_coppelia plot --robot_name <robot_name> --plot_types <plot_type_1> [<plot_type_2> ...]
                     [--model_ids <id_1> <id_2> ...] [--scene_to_load_folder <folder>] 
                     [--save_plots] [--lat_fixed_timestep <float>] [--timestep_unit <str>] 
                     [--lat_file_path <path>] [--verbose <level>]

Supported Plot Types:
        - spider: Generates a spider chart comparing multiple models across various metrics.
        - convergence-walltime: Plots reward convergence vs. wall time for each model.
        - convergence-steps: Plots reward convergence vs. steps for each model.
        - convergence-simtime: Plots reward convergence vs. simulation time for each model.
        - convergence-episodes: Plots reward convergence vs. episodes for each model.
        - convergence-all: Generates all convergence plots for each model.
        - compare-rewards: Compares rewards across multiple models with smoothing and variability bands.
        - compare-episodes_length: Compares episode lengths across multiple models.
        - compare-convergences: Compares convergence points across multiple models.
        - histogram_speeds: Plots histograms for linear and angular speeds for each model.
        - grouped_bar_speeds: Creates grouped bar charts for linear and angular speeds across models.
        - grouped_bar_targets: Creates grouped bar charts for target zone frequencies across models.
        - bar_target_zones: Plots bar charts for target zone distributions for each model.
        - plot_scene_trajs: Visualizes the scene and trajectories followed by the robot during testing.
        - plot_boxplots: Generates boxplots for various metrics across models.
        - lat_curves: Plots LAT-Agent and LAT-wall curves for each model.
        - speed_lat_curves: Plots curves comparing linear speed and LAT-Agent over time for a model.
        - dist_lat_curves: Plots curves comparing distance traveled and LAT-Agent over time for a model.
        - plot_from_csv: Loads externally saved CSV logs to generate reward comparison and metric boxplots across models.

Features:
    - Generates reward, episode length, and convergence plots for multiple models.
    - Creates spider charts comparing overall model performance.
    - Visualizes trajectories and scene layouts from testing sessions.
    - Plots LAT curves (Agent and Wall time) and speed/distance metrics per timestep.
    - Exports boxplots, bar charts, and histograms to files or displays them interactively.
    - Accepts CSV logs from external sources for comparison.
    - Automatically handles input data discovery based on naming patterns.
"""
from datetime import datetime
import matplotlib
from collections import defaultdict
import glob
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import os
import pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from common import utils
from common.rl_coppelia_manager import RLCoppeliaManager
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline


def plot_spider(rl_copp_obj, title='Models Comparison'):
    """
    Plots multiple spider charts on the same figure to compare different models.

    Args:
    - rl_copp_object (RLCoppeliaManager): Instance of RLCoppeliaManager class just for managing the args and the base path.
    - title (str): The title of the chart.
    """

    # Custom labels for each axis of the spider chart
    categories = [
        "Learning_Convergence",
        "Mean_Reward",
        "Episode_Efficiency",
        "Episode_Completion Rate",  
        "Innermost Target_Rate"
    ]

    # Get metrics from testing summary file
    testing_csv_path = os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "testing_metrics", "test_records.csv")
    test_column_names = [
        "Avg reward",
        "Avg time reach target",
        "Percentage terminated",
        "Target zone 3 (%)"
    ]
    test_data = utils.get_data_for_spider(testing_csv_path, rl_copp_obj.args, test_column_names)

    # Get metrics from training summary file
    training_csv_path = os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "training_metrics", "train_records.csv")
    # Add train results like: "Action time (s)", "Time to converge (h)", "train/actor_loss", "train/critic_loss", "rollout/ep_len_mean", "rollout/ep_rew_mean"

    train_column_names = [
        "Action time (s)",
        "Time to converge (h)",
    ]
    train_data = utils.get_data_for_spider(training_csv_path, rl_copp_obj.args, train_column_names)

    # Concatenate the train and test DataFrames along the columns axis
    df_train = pd.DataFrame(train_data).T  # Transpose so each row is a model
    df_test = pd.DataFrame(test_data).T  
    concatenated_df = pd.concat([df_train, df_test], axis=1)    # Merge both datasets by columns
    concatenated_df = concatenated_df.fillna(np.nan)   # Ensure missing values are handled

    data_list, names, labels = utils.process_spider_data(concatenated_df)   # Prepare data for plotting

    # Override the labels with the names saved in categories list
    for cat in range(len(categories)):
        try:
            labels[cat] = categories[cat]
        except: # If some tag is missing, just pass
            pass

    # Plot the spider graph
    num_vars = len(labels)  # Number of axes in the radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()    # Evenly spaced angles
    angles += angles[:1]    # Close the loop
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True)) # Create polar subplot
    ax.set_theta_offset(np.pi / 2)  # Rotate so the first axis is at the top
    ax.set_theta_direction(-1)      # Clockwise direction


    for data, name in zip(data_list, names):    # Plot each data set
        data = data + data[:1]  # Assure that we are closing the polygon
        ax.plot(angles, data, linewidth=2, linestyle='solid', label=name)
        ax.fill(angles, data, alpha=0.1)

    # Labels of the axis
    labels = [label.replace("_", "\n") for label in labels]  # Replace underscores with newlines for better readability
    ax.set_yticklabels([])  # Remove labels from radial axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=18)  # Configure labels

    # Fine-tune label positions manually
    for label in ax.get_xticklabels():
        if "Convergence" in label.get_text():
            label.set_y(label.get_position()[1] - 0.1)  
        elif "Reward" in label.get_text():
            label.set_y(label.get_position()[1] - 0.25) 
        elif "Efficiency" in label.get_text():
            label.set_y(label.get_position()[1] - 0.15)
        elif "Completion" in label.get_text():
            label.set_y(label.get_position()[1] - 0.15)
        elif "Target" in label.get_text():
            label.set_y(label.get_position()[1] - 0.3)
        elif "Trajectory" in label.get_text():
            label.set_y(label.get_position()[1] - 0.2)
        else:
            label.set_y(label.get_position()[1] - 0.1)  

    # Set the radial axis limits
    ax.set_ylim(0, 1.1) 
    
    ax.spines['polar'].set_visible(False)   # Hide outer circle
    ax.spines['polar'].set_bounds(0, 1)     # Limit the visible part of the spine

    # Add the leyend and title
    ax.legend(
        loc='upper right',
        bbox_to_anchor=(1.6, 1.3),
        fontsize = 14,
        ncol = 2
        ) 
    # ax.set_title(title, size=16, color='black', y=1.1)
    plt.tight_layout()

    # Save or show plot
    if rl_copp_obj.args.save_plots:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"spider_{timestamp}.png"
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_convergence (rl_copp_obj, model_index, x_axis, show_plots = True, title = "Reward Convergence Analysis"):
    """
    Plots a graph with the reward vs:
        - Wall time
        - Steps
        - Episodes 
        - Simulation time, 
    and shows the point at which the reward stabilizes (converges) based on a first-order fit.

    Args:
        rl_copp_object (RLCoppeliaManager): Instance of RLCoppeliaManager class just for managing the args and the base path.
        model_index (int): Index of the model to analyze.
        x_axis (str): Name of the x_axis.
        show_plots (bool): If True, it will show the plots. If False, it will just return the convergence point.
        title (str, Optional): The title of the chart. By default, it is "Reward Convergence Analysis".
    Returns:
        convergence_point (float): The point at which the reward converges.
    """
    # CSV File path to get data from
    file_pattern = f"{rl_copp_obj.args.robot_name}_model_{rl_copp_obj.args.model_ids[model_index]}_*.csv"
    files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "training_metrics", file_pattern))
    
    # Calculate the convergence time
    convergence_point, reward_fit, x_axis_values, reward, reward_at_convergence = utils.get_convergence_point (files[0], x_axis, convergence_threshold=0.001)
    logging.info(f"Convergence point: {round(convergence_point,2)} - Reward: {round(reward_at_convergence,2)}")

    # Get the training csv path for later getting the action times from there
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path,"train_records.csv")    # Name of the train records csv to search the algorithm used
    model_name = rl_copp_obj.args.robot_name + "_model_" + str(rl_copp_obj.args.model_ids[model_index])
    timestep = (utils.get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)"))


    # Plot results
    if show_plots:
        plt.figure(figsize=(8, 5))
        # plt.plot(x_axis_values, reward, label='Original Data', marker='o', linestyle='')
        # plt.plot(x_axis_values, reward_fit, label='Exponential Fit', linestyle='--')
        if x_axis == "WallTime":
            plt.plot(x_axis_values, reward, label='Original Data', linestyle='-')
            plt.plot(x_axis_values, reward_fit, label='Exponential Fit', linestyle='-')
            plt.xlabel('Wall time (hours)')
            plt.axvline(convergence_point, color='r', linestyle=':', label=f'Convergence Wall Time: {convergence_point:.2f}h')
            title = title + ' vs Wall Time'
        elif x_axis == "Steps":
            plt.plot(x_axis_values, reward, label='Original Data', linestyle='-')
            plt.plot(x_axis_values, reward_fit, label='Exponential Fit', linestyle='-')
            plt.xlabel('Steps')
            plt.axvline(convergence_point, color='r', linestyle=':', label=f'Convergence Steps: {convergence_point:.2f}')
            title = title +  ' vs Steps'
        elif x_axis == "SimTime":
            plt.plot(x_axis_values, reward, label='Original Data', linestyle='-')
            plt.plot(x_axis_values, reward_fit, label='Exponential Fit', linestyle='-')
            plt.xlabel('Simulation time (hours)')
            plt.axvline(convergence_point, color='r', linestyle=':', label=f'Convergence Sim Time: {convergence_point:.2f}h')
            title = title + ' vs Sim Time'
        elif x_axis == "Episodes":
            plt.plot(x_axis_values, reward, label='Original Data', linestyle='-')
            plt.plot(x_axis_values, reward_fit, label='Exponential Fit', linestyle='-')
            plt.xlabel('Episodes')
            plt.axvline(convergence_point, color='r', linestyle=':', label=f'Convergence Episodes: {convergence_point:.2f}')
            title = title +  ' vs Episodes'
        plt.ylabel('Reward')
        plt.legend()
        plt.title(title + ": Model " + str(timestep) + "s")
        plt.grid()
        if rl_copp_obj.args.save_plots:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"convergence_{rl_copp_obj.args.model_ids[model_index]}_{x_axis}_{timestamp}.png"
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    return convergence_point


def plot_metrics_comparison (rl_copp_obj, metric, max_steps = 200000, smooth_flag=True, smooth_level = 150, band_flag = False, title = "Comparison"):
    """
    Plot the same metric of multiple models for comparing them (with mean curve and variability). 
    X axis will be the number of steps.

    Args:
        rl_copp_object (RLCoppeliaManager): Instance of RLCoppeliaManager class just for managing the args and the base path.
        metric (str): The metric to be plotted ("rewards" or "episodes_length").
        ax_steps (int, optional): Maximum number of steps to consider for the plot. If None, it will use the maximum steps from the data.
        smooth_flag (bool): If True, it will apply a moving average smoothing to the data.
        smooth_level (int): The window size for the moving average smoothing.
        band_flag (bool): If True, it will plot a variability band around the mean curve.
        title (str): The title of the chart.
    """
    # Initialize variable for maximum global step across all models
    max_global_step = 0

    # Get the training csv path for later getting the action times from there
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path, "train_records.csv")  # Name of the train records csv to search the algorithm used
    timestep_to_data = {}

    # CSV File path to get data from
    for model_index, model_id in enumerate(rl_copp_obj.args.model_ids):
        model_name = f"{rl_copp_obj.args.robot_name}_model_{model_id}"
        file_pattern = f"{model_name}_*.csv"
        files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "training_metrics", file_pattern))
        logging.info(f"Files detected {files}, model index {model_index}, model ID {model_id}")

        if not files:
            logging.warning(f"No training CSV found for model {model_name}. Skipping.")
            continue
        
        # Read the CSV file
        try:
            df = pd.read_csv(files[0])
        except Exception as e:
            logging.error(f"File not found for model {model_name}. Skipping this model. Error: {e}")   
            # del model_id
            continue

        # Get timestep of the selected model
        timestep = utils.get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)")
        
        # Extract steps and rewards
        if max_steps is not None:   # Limit max steps if specified
            df = df[df['Step'] <= max_steps]
        steps = df['Step'].values
        max_global_step = max(max_global_step, steps.max()) # Update the maximum global step
        if metric == "rewards":
            data = df['rollout/ep_rew_mean'].values
            y_label = "Reward"
        elif metric == "episodes_length":
            data = df['rollout/ep_len_mean'].values
            y_label = "Episodes length (steps)"
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        # Smooth the data
        if smooth_flag:
            data = utils.moving_average(data, window_size=smooth_level)
            steps = steps[:len(data)]

        # Group data by timestep
        # If the timestep is not already in the dictionary, create a new entry
        # Otherwise, append the (steps, data) tuple to the existing list for that timestep
        if timestep not in timestep_to_data:
            timestep_to_data[timestep] = []
        timestep_to_data[timestep].append((steps, data))
            
    # Create a color map for the models
    color_map = utils.get_color_map(len(timestep_to_data)) 

    # Plot the mean curve and variability band for each timestep group
    plt.figure(figsize=(11.2, 7.2))
    for model_index, (timestep, data_list) in enumerate(timestep_to_data.items()):
        # Align data by step and calculate mean and std
        all_steps = [d[0] for d in data_list]
        all_data = [d[1] for d in data_list]

        # Define a common set of steps (e.g., 1000 evenly spaced points)
        common_steps = np.linspace(
            min([steps[0] for steps in all_steps]), 
            max([steps[-1] for steps in all_steps]), 
            1000)

        # Interpolate all curves to the common set of steps
        interpolated_data = []
        for steps, data in zip(all_steps, all_data):
            interpolator = interp1d(steps, data, kind='linear', bounds_error=False, fill_value="extrapolate")
            interpolated_data.append(interpolator(common_steps))

        # Convert to NumPy arrays
        interpolated_data = np.array(interpolated_data)

        # Calculate mean and standard deviation
        mean_data = np.mean(interpolated_data, axis=0)
        std_data = np.std(interpolated_data, axis=0)

        # Plot mean curve
        color = color_map[model_index]  # Assign a unique color for each model
        common_steps = common_steps/1000
        plt.plot(common_steps, mean_data, label=f"Model {timestep}s", color=color, linewidth=2.5)

        # Plot variability band if enabled
        if band_flag:
            # Fill the area between mean - std and mean + std
            plt.fill_between(
                common_steps,
                mean_data - std_data,
                mean_data + std_data,
                color=color,
                alpha=0.3,
                edgecolor="black",
                linewidth=0.5
            )

    # Plot configuration
    plt.xlabel(r'Steps $\times$ 10$^3$', fontsize=22, labelpad=5)
    plt.ylabel(y_label, fontsize=22, labelpad=-4)
    max_step_k = max_global_step / 1000.0
    max_tick = int(np.ceil(max_step_k / 50.0) * 50)
    xticks = np.arange(0, max_tick + 1, 50)
    plt.xticks(xticks, fontsize=16)
    plt.xlim(0, max_tick)
    plt.tick_params(axis='both', which='major', labelsize=18, pad = 10)
    plt.legend(fontsize=18, ncol = utils.get_legend_columns(len(timestep_to_data), items_per_column = 8))
    plt.grid(True)
    plt.tight_layout()

    # Save plots if needed
    if rl_copp_obj.args.save_plots:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"metrics_comparison_{metric}_{timestamp}.png"
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_convergence_comparison (rl_copp_obj, title = "Convergence Comparison "):
    """
    Plot the convergence point of multiple models for comparing them.

    Args:
        rl_copp_obj (RLCoppeliaManager): Instance used to access arguments and file paths.
        title (str): Base title for each plot.
    """

    # Define the x-axis options
    x_axis = ["WallTime", "Steps", "SimTime", "Episodes"]

    # Get the training csv path for later getting the action times from there
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path,"train_records.csv")
    timestep = []
    n_models = len(rl_copp_obj.args.model_ids)

    # Generate a color map for the models
    color_map = utils.get_color_map(n_models)
    
    # For each category
    for conv_type in x_axis:
        plt.figure(figsize=(10, 6))

        max_convergence_point = 0  # Track the maximum convergence point for this category

        for model_index, model_id in enumerate(rl_copp_obj.args.model_ids):
            # Get timestep of the selected model
            model_name = f"{rl_copp_obj.args.robot_name}_model_{model_id}"
            timestep.append(utils.get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)"))
            
            # CSV File path to get data from
            file_pattern = f"{model_name}_*.csv"
            files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "training_metrics", file_pattern))

            if not files:
                continue

            # Get convergence point and reward data
            convergence_point, _reward_fit, x_axis_values, reward, _reward_at_convergence = utils.get_convergence_point(
                files[0], 
                conv_type, 
                convergence_threshold=0.01
            )

            # Update the maximum convergence point
            max_convergence_point = max(max_convergence_point, convergence_point)

            # Assign a unique color for each model
            color = color_map[model_index]
            
            # Smooth reward data
            smoothed_rewards = utils.moving_average(reward, window_size=100)
            smoothed_x = x_axis_values[:len(smoothed_rewards)]  # recortar x para que coincida

            # Convert to k steps in "Steps" category
            if conv_type == "Steps":
                smoothed_x = smoothed_x / 1000.0
                conv_plot_x = convergence_point / 1000.0
            else:
                conv_plot_x = convergence_point

            # Plot curve and vertical line
            plt.plot(smoothed_x, smoothed_rewards, color=color, linewidth=3)
            plt.axvline(
                conv_plot_x,
                color=color, 
                linestyle='--', 
                label=f'{timestep[model_index]}s', 
                # label=f'Timestep {timestep[model_index]}s - {line_label}', 
                linewidth=2.5
            )
            
        # Plot configuration
        x_labels = {
            "WallTime": "Wall time (hours)",
            "Steps": r"Steps $\times$ 10$^3$",
            "SimTime": "Simulation time (hours)",
            "Episodes": "Episodes"
        }
        plt.xlabel(x_labels[conv_type], fontsize=26, labelpad=12)
        plt.ylabel("Reward", fontsize=26, labelpad=10)
        plt.tick_params(axis='both', which='major', labelsize=22)
        plt.yticks(np.arange(-1, 1, 0.25))
        if conv_type == "Steps":
            plt.xlim(0, max_convergence_point * 1.2 / 1000)
        else:
            plt.xlim(0, max_convergence_point * 1.2)

        # Title and legend
        # plt.title(f"{title} ({conv_type})", fontsize=24)
        plt.legend(loc='lower right', fontsize=18, ncol=utils.get_legend_columns(n_models, items_per_column=4))

        # Layout and save/show
        plt.grid()
        plt.tight_layout()
        if rl_copp_obj.args.save_plots:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"convergence_{conv_type}_{timestamp}.png"
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()


def plot_convergence_comparison_cloud (rl_copp_obj, title = "Convergence Comparison "):
    """
    Plot a scatter plot of convergence points for multiple models vs. their timesteps.

    Args:
        rl_copp_obj (RLCoppeliaManager): Instance used to access arguments and file paths.
        title (str): Base title for each plot.
    """

    # Define the x-axis options
    x_axis = "Steps"

    # Get the training csv path for later getting the action times from there
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path,"train_records.csv")
    timestep = []
    n_models = len(rl_copp_obj.args.model_ids)
    convergence_point_list = []

    # Generate a color map for the models
    color_map = utils.get_color_map(n_models)
    
    plt.figure(figsize=(10, 6))

    max_convergence_point = 0  # Track the maximum convergence point for this category

    for model_index, model_id in enumerate(rl_copp_obj.args.model_ids):
        # Get timestep of the selected model
        model_name = f"{rl_copp_obj.args.robot_name}_model_{model_id}"
        timestep.append(float(utils.get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)")))
        
        # CSV File path to get data from
        file_pattern = f"{model_name}_*.csv"
        files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "training_metrics", file_pattern))

        if not files:
            continue

        # Get convergence point and reward data
        convergence_point, *_ = utils.get_convergence_point(files[0], x_axis, convergence_threshold=0.01)

        # Add convergence point to the list for cloud plot
        convergence_point_list.append(convergence_point)

    # Update the maximum convergence point
    max_convergence_point = max(convergence_point_list)

    # Convert convergence points to thousands for visualization
    convergence_point_list = np.array(convergence_point_list) / 1000.0

    # Plot curve and vertical line
    plt.scatter(timestep, convergence_point_list, c='blue', s=80, edgecolors='black')

    # for t, y in zip(timestep, convergence_point_list):
    #     plt.text(t, y + 1, f"{t:.2f}s", fontsize=12, ha='center')

    plt.xlabel("Timestep (s)", fontsize=18, labelpad=8)
    plt.ylabel("Convergence point (Steps × 10³)", fontsize=18, labelpad=5)

    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.xticks(sorted(set(timestep)), [f"{t:.2f}" for t in sorted(set(timestep))], rotation=45)
    plt.yticks(np.arange(0, max(convergence_point_list) + 5, 50), fontsize=16)
    # plt.xlim(0, max_convergence_point * 1.2 / 1000)

    # Title and legend
    # plt.title(f"{title} ({conv_type})", fontsize=24)
    plt.legend(loc='lower right', fontsize=18, ncol=utils.get_legend_columns(n_models, items_per_column=4))

    # Layout and save/show
    plt.grid()
    plt.tight_layout()
    if rl_copp_obj.args.save_plots:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"convergence_cloud_{x_axis}_{timestamp}.png"
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()



def plot_grouped_bar_chart(rl_copp_obj, mode, num_intervals=10, title=" Distribution by Intervals"):
    """
    Creates a grouped bar chart showing the distribution of values across multiple models.

    Args:
        rl_copp_obj (RLCoppeliaManager): Manager containing argument paths and model info.
        mode (str): Type of data to visualize ("speeds" or "target_zones").
        num_intervals (int): Number of intervals for histogram (only applies to "speeds").
        title (str): Title suffix for the chart.
    """

    # Initialize data keys and ranges depending on the mode
    if mode == "speeds":
        data_keys = ['Angular speed', 'Linear speed']
        data_keys_units = ["(rad/s)", "(m/s)"]
        min_value = [-0.5, 0.1]
        max_value = [0.5, 0.5]
    elif mode == "target_zones":
        data_keys = ['Target zone']
        data_keys_units = [""]
        categories = [1, 2, 3]
    else:
        logging.error(f"Unsupported mode: {mode}")
        raise ValueError(f"Unsupported mode: {mode}")

    # Get the training csv path for later getting the action times from there
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path,"train_records.csv")  
    n_models = len(rl_copp_obj.args.model_ids)

    color_map = utils.get_color_map(n_models)

    # Plot one chart per variable (e.g., Angular speed, Linear speed)
    for id_data in range(len(data_keys)):
        timestep = []

        if mode== "speeds": # continuos intervals case
            # Compute evenly spaced intervals
            interval_size = (max_value[id_data] - min_value[id_data]) / num_intervals
            intervals = [(min_value[id_data] + i * interval_size, min_value[id_data] + (i + 1) * interval_size) 
                        for i in range(num_intervals)]
            
            # Prepare data structure for frequencies
            interval_labels = [f"[{a:.2f}, {b:.2f}]" for a, b in intervals]
            num_groups = len(intervals)

        elif mode == "target_zones":    # discrete intervals case
            interval_labels = [str(cat) for cat in categories]
            num_groups = len(categories)
    
        # Matrix to store frequencies: rows=models, columns=intervals
        frequencies = np.zeros((n_models, num_groups))

        # Load data and calculate frequencies for each model and interval
        for i, model_idx in enumerate(rl_copp_obj.args.model_ids):

            # File selection based on mode
            if mode == "speeds":    # turtleBot_model_308_last_speeds_2025-05-10_12-00-32.csv
                file_pattern = f"{rl_copp_obj.args.robot_name}_model_{model_idx}_*_otherdata_*.csv"
                
            elif mode == "target_zones":    # turtleBot_model_308_last_test_2025-05-10_12-00-32.csv
                file_pattern = f"{rl_copp_obj.args.robot_name}_model_{model_idx}_*_test_*.csv"

            # Search for the files with that pattern inside testing_metrics directory
            subfolder_pattern = f"{rl_copp_obj.args.robot_name}_model_{model_idx}_*_testing"
            files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "testing_metrics", subfolder_pattern, file_pattern))

            
            # If there are no files
            if not files:
                logging.error(f"Error: no files found for model {model_idx}")
                continue

            # Read csv file
            df = pd.read_csv(files[0])
            data = (df[data_keys[id_data]])    
           
            # Calculate the frequency for each case
            if mode== "speeds":
                for j, (interval_start, interval_end) in enumerate(intervals):
                    mask = (data >= interval_start) & (data < interval_end)
                    frequencies[i, j] = np.sum(mask)
            elif mode == "target_zones":
                for j, category in enumerate(categories):
                    mask = (data == category)
                    frequencies[i, j] = np.sum(mask)
            
            # Normalize to percentage if desired
            frequencies[i, :] = frequencies[i, :] / len(data) * 100

            # Get timestep of the selected model
            model_name = rl_copp_obj.args.robot_name + "_model_" + str(model_idx)
            timestep.append(utils.get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)"))
         
        # Plot creation
        plt.figure(figsize=(12, 7))
        
        # Set width of bars and positions
        bar_width = 0.8 / len(timestep)
        r = np.arange(num_groups)
        
        # Plot bars for each model
        for i in range(len(timestep)):
            position = r + i * bar_width - (len(timestep) - 1) * bar_width / 2
            plt.bar(position, frequencies[i, :], width=bar_width, color = color_map[i],
                    label=f"Model {timestep[i]}s")
        
        # Plot configuration
        plt.xlabel(f"{data_keys[id_data]} {data_keys_units[id_data]}", fontsize=22, labelpad=15)
        plt.ylabel('Percentage of Samples (%)', fontsize=22, labelpad=15)
        # plt.title(data_keys[id_data] + title, fontsize=16, pad=20)
        plt.xticks(r, interval_labels, rotation=30 if mode == "speeds" else 0, ha='right', fontsize=20)  # rotate labels for speed (many intervals)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=17, ncol = utils.get_legend_columns(len(timestep), items_per_column=5)) # maybe you will have to add this: bbox_to_anchor=(1.05, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.savefig("fig_fixed.pdf", dpi=300, bbox_inches="tight")

        # Save or show plot
        if rl_copp_obj.args.save_plots:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"grouped_bar_chart_{id_data}_{timestamp}.png"
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
        

def plot_scene_trajs(rl_copp_obj, folder_path):
    """
    Plots a 2D scene representation along with robot trajectories.

    The function draws the robot, obstacles and target rings from a CSV scene description,
    and overlays the recorded robot trajectories from multiple models. It marks whether
    the robot successfully reached a target or not based on final position distance.

    Args:
        rl_copp_obj (RLCoppeliaManager): Object managing paths and CLI args.
        folder_path (str): Directory containing the scene and trajectory files.

    Raises:
        ValueError: If the number of scene CSV files found is not 1.
    """
    # Search scene files
    scene_files = glob.glob(os.path.join(folder_path, "scene_*.csv"))
    if len(scene_files) != 1:
        raise ValueError(f"Expected one scene CSV, found {len(scene_files)}")
    scene_path = scene_files[0]
    df_scene = pd.read_csv(scene_path)

    # Plot creation
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel("x (m)", fontsize=18, labelpad=10)
    ax.set_ylabel("y (m)", fontsize=18, labelpad=2)
    ax.set_xlim(2.5, -2.5)  # inverted axis
    ax.set_ylim(2.5, -2.5)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='major', labelsize=16)  
    # ax.set_title("CoppeliaSim Scene Representation", fontsize=16, pad=20)

    # Draw 0.5 m grid
    for i in np.arange(-2.5, 3, 0.5):
        ax.axhline(i, color='lightgray', linewidth=0.5, zorder=0)
        ax.axvline(i, color='lightgray', linewidth=0.5, zorder=0)

    # Draw all the elements of the scene
    target_counter = 0
    for _, row in df_scene.iterrows():
        x, y = row['x'], row['y']
        if row['type'] == 'robot':
            # Robot body
            circle = plt.Circle((x, y), 0.35 / 2, color='black', label='Robot', zorder=4)
            ax.add_patch(circle)

            # Orientation marker (triangle)
            if 'theta' in row:
                theta = row['theta']
                # Triangle dimensions
                front_length = 0.15
                side_offset = 0.08
                # Front point
                front = (x + front_length * np.cos(theta), y + front_length * np.sin(theta))
                # Side points
                left = (x + side_offset * np.cos(theta + 2.5), y + side_offset * np.sin(theta + 2.5))
                right = (x + side_offset * np.cos(theta - 2.5), y + side_offset * np.sin(theta - 2.5))
                triangle = plt.Polygon([front, left, right], color='white', zorder=4)
                ax.add_patch(triangle)

        elif row['type'] == 'obstacle':
            # Columns as osbtacles
            circle = plt.Circle((x, y), 0.25 / 2, color='gray', label='Obstacle')
            ax.add_patch(circle)

        elif row['type'] == 'target':
            # Plot the target rings
            target_rings = [(0.5 / 2, 'blue'), (0.25 / 2, 'red'), (0.03 / 2, 'yellow')]
            for radius, color in target_rings:
                circle = plt.Circle((x, y), radius, color=color, fill=True, alpha=0.6)
                ax.add_patch(circle)
            target_label = chr(ord('A') + target_counter)
            # For drawing the identifying letter in one position or another depending on the target location
            y_offset = -0.3 if y < 0 else 0.48
            ax.text(x, y + y_offset, f"{target_label}", fontsize=22, fontweight='bold', color='black', zorder=10, ha='center')
            target_counter += 1
    
    # Load and group trajectory files
    traj_files = glob.glob(os.path.join(folder_path, "trajs", "trajectory_*.csv"))
    model_trajs = defaultdict(list)
    for file in traj_files:
        parts = file.split('_')
        model_id = parts[-1].split('.')[0]
        model_trajs[model_id].append(os.path.join(folder_path, file))

    colors = plt.cm.get_cmap("tab10")
    train_records_csv_name = os.path.join(rl_copp_obj.paths["training_metrics"], "train_records.csv")
    model_plot_data = []

    # Process trajectories and gather plot info
    for i, (model_id, paths) in enumerate(model_trajs.items()):
        color = colors((i + 1) % 10)
        model_name = rl_copp_obj.args.robot_name + "_model_" + str(model_id)
        timestep = float(utils.get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)"))

        for path in paths:
            df = pd.read_csv(path)
            model_plot_data.append({
                "timestep": timestep,
                "color": color,
                "x": df["x"],
                "y": df["y"],
                "label": f"Model {timestep}s"
            })

    # Sort models by timestep before plotting
    model_plot_data.sort(key=lambda d: d["timestep"])

    # Get all target positions
    target_rows = df_scene[df_scene['type'] == 'target']
    targets = [(row['x'], row['y']) for _, row in target_rows.iterrows()]

    # Plot all trajectories and final positions
    for data in model_plot_data:
        ax.plot(data["x"], data["y"], color=data["color"],
                label=data["label"], linewidth=2, zorder=3)

        # Final position of the robot
        final_x = data["x"].iloc[-1]
        final_y = data["y"].iloc[-1]

        # Get distance to nearest target
        if len(targets) > 0:
            distances = [np.hypot(final_x - tx, final_y - ty) for tx, ty in targets]
            min_distance = min(distances)            
            logging.debug(f"Distance to closest target: {min_distance:.2f} m")

            # If distance is greater than 0.45 m, plot a cross to indicate a collision
            # Actually collision happens when the robot is nearer than 45cm to the target, but
            # here we are measuring the distance between the central point of the robot and the target, 
            # so we need to increase the distance a bit
            if min_distance > 0.45:
                ax.plot(final_x, final_y, marker='x', color=data["color"],
                        markersize=12, markeredgewidth=2, zorder=4)
                ax.plot(final_x, final_y, marker='o', color="black",
                        markersize=13, markeredgewidth=1.5, markerfacecolor='none', zorder=4)
            # If not, draw a little circle to indicate the final position
            else:
                ax.plot(final_x, final_y, marker='o', color=data["color"],
                        markersize=4, markeredgewidth=2, zorder=4)

    # Plot configuration
    # Removed duplicated labels from legend
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), 
              unique.keys(), 
              loc='upper right', 
            #   bbox_to_anchor=(1.46, 1.03), 
              fontsize = 14,
              ncol = 2)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("fig_fixed.pdf", dpi=300, bbox_inches="tight")

    # Save or show plot
    if rl_copp_obj.args.save_plots:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"plot_scene_trajs_{timestamp}.png"
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_scene_trajs_streaming(rl_copp_obj, folder_path, poll_interval=1.0, stop_after_idle=None):
    """
    Live plot of a 2D scene with trajectories discovered over time.

    The scene (robot, obstacles, targets) is drawn once. Then the function keeps
    the figure open and polls the `trajs/` subfolder for new `trajectory_*.csv`
    files. Each new trajectory is plotted immediately alongside the existing ones.

    Legend ordering and colors:
      - "Robot" and "Obstacle" always appear first in the legend (in that order).
      - Model colors follow the original rule: cmap=tab10, color index is (i+1) % 10
        where i is the (0-based) appearance order of the model id.

    Args:
        rl_copp_obj: RLCoppeliaManager-like object (provides args and paths).
        folder_path (str): Directory with the scene CSV and `trajs/` subfolder.
        poll_interval (float): Seconds between scans.
        stop_after_idle (float | None): If set, stop after this idle time (s)
            without new files. If None, run until interrupted.

    Notes:
        - Uses Matplotlib interactive mode (plt.ion()).
        - Skips CSVs that fail to parse (likely still being written) and retries.
        - Deduplicates legend labels and sorts model entries by timestep when known.
    """
    import os, glob, time, logging
    from collections import defaultdict
    from datetime import datetime

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # --- Scene CSV ---
    scene_files = glob.glob(os.path.join(folder_path, "scene_*.csv"))
    if len(scene_files) != 1:
        raise ValueError(f"Expected one scene CSV, found {len(scene_files)}")
    scene_path = scene_files[0]
    df_scene = pd.read_csv(scene_path)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel("x (m)", fontsize=18, labelpad=10)
    ax.set_ylabel("y (m)", fontsize=18, labelpad=2)
    ax.set_xlim(2.5, -2.5)  # inverted axis (keep your convention)
    ax.set_ylim(2.5, -2.5)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='major', labelsize=16)

    # 0.5 m grid
    for i in np.arange(-2.5, 3, 0.5):
        ax.axhline(i, color='lightgray', linewidth=0.5, zorder=0)
        ax.axvline(i, color='lightgray', linewidth=0.5, zorder=0)

    # --- Draw static scene; keep handles for legend priority ---
    robot_patch_handle = None
    obstacle_patch_handle = None

    target_counter = 0
    for _, row in df_scene.iterrows():
        x, y = row['x'], row['y']
        if row['type'] == 'robot':
            circle = plt.Circle((x, y), 0.35 / 2, color='black', label='Robot', zorder=4)
            ax.add_patch(circle)
            robot_patch_handle = robot_patch_handle or circle

            if 'theta' in row:
                theta = row['theta']
                front_length = 0.15
                side_offset = 0.08
                front = (x + front_length * np.cos(theta), y + front_length * np.sin(theta))
                left = (x + side_offset * np.cos(theta + 2.5), y + side_offset * np.sin(theta + 2.5))
                right = (x + side_offset * np.cos(theta - 2.5), y + side_offset * np.sin(theta - 2.5))
                triangle = plt.Polygon([front, left, right], color='white', zorder=4)
                ax.add_patch(triangle)

        elif row['type'] == 'obstacle':
            circle = plt.Circle((x, y), 0.25 / 2, color='gray', label='Obstacle')
            ax.add_patch(circle)
            obstacle_patch_handle = obstacle_patch_handle or circle

        elif row['type'] == 'target':
            for radius, color in [(0.5/2, 'blue'), (0.25/2, 'red'), (0.03/2, 'yellow')]:
                ax.add_patch(plt.Circle((x, y), radius, color=color, fill=True, alpha=0.6))
            target_label = chr(ord('A') + target_counter)
            y_offset = -0.3 if y < 0 else 0.48
            ax.text(x, y + y_offset, f"{target_label}", fontsize=22, fontweight='bold',
                    color='black', zorder=10, ha='center')
            target_counter += 1

    # Targets list for success/fail marker
    targets = [(row['x'], row['y']) for _, row in df_scene[df_scene['type'] == 'target'].iterrows()]

    # --- Colors and bookkeeping ---
    cmap = plt.cm.get_cmap("tab10")
    train_records_csv = os.path.join(rl_copp_obj.paths["training_metrics"], "train_records.csv")

    seen_files = set()
    model_color_map = {}          # model_id -> RGBA (from tab10)
    model_appearance_index = {}   # model_id -> i (0-based)
    next_model_idx = 0            # increments per new model id

    model_label_to_handle = {}    # latest Line2D handle per label
    model_label_to_ts = {}        # label -> timestep (float) for legend sort

    def get_model_label_and_ts(model_id: str):
        """Return ('Model Xs', X) when possible; else ('Model <id>', inf)."""
        model_name = rl_copp_obj.args.robot_name + "_model_" + str(model_id)
        try:
            ts = float(utils.get_data_from_training_csv(
                model_name, train_records_csv, column_header="Action time (s)"
            ))
            return f"Model {ts:g}s", ts
        except Exception:
            return f"Model {model_id}", float('inf')

    def rebuild_legend():
        """Legend with Robot/Obstacle first, then models sorted by timestep."""
        handles = []
        labels = []

        # Priority entries: Robot, Obstacle (if present)
        if robot_patch_handle is not None:
            handles.append(robot_patch_handle)
            labels.append('Robot')
        if obstacle_patch_handle is not None:
            handles.append(obstacle_patch_handle)
            labels.append('Obstacle')

        # Then model entries, sorted by timestep (unknowns at the end)
        if model_label_to_handle:
            ordered_models = sorted(model_label_to_handle.items(),
                                    key=lambda kv: model_label_to_ts.get(kv[0], float('inf')))
            for lab, h in ordered_models:
                handles.append(h)
                labels.append(lab)

        ax.legend(handles, labels, loc='upper right', fontsize=14, ncol=2)

    # --- Initial draw ---
    plt.grid(True)
    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    # --- Polling loop ---
    traj_dir = os.path.join(folder_path, "trajs")
    last_new_time = time.time()

    try:
        while True:
            current = glob.glob(os.path.join(traj_dir, "trajectory_*.csv"))
            new_files = [f for f in current if f not in seen_files]
            any_new = False

            for fpath in sorted(new_files):
                base = os.path.basename(fpath)
                try:
                    model_id = base.split('_')[-1].split('.')[0]
                except Exception:
                    model_id = "unknown"

                # Color assignment as in your original: tab10 with (i+1) % 10
                if model_id not in model_color_map:
                    i = next_model_idx
                    model_color_map[model_id] = cmap((i + 1) % 10)
                    model_appearance_index[model_id] = i
                    next_model_idx += 1

                # Load CSV (skip if temporary/locked)
                try:
                    df = pd.read_csv(fpath)
                    if not {"x", "y"}.issubset(df.columns):
                        logging.warning(f"[stream] Missing x/y in {fpath}, skipping.")
                        continue
                except Exception:
                    continue  # try again next poll

                # Label + timestep (for legend sort)
                label, ts = get_model_label_and_ts(model_id)

                # Plot trajectory
                color = model_color_map[model_id]
                (line_handle,) = ax.plot(df["x"], df["y"], color=color, label=label,
                                         linewidth=2, zorder=3)

                # Store latest handle/ts per label for legend
                model_label_to_handle[label] = line_handle
                model_label_to_ts[label] = ts

                # Final position marker: success/fail against nearest target
                try:
                    fx, fy = df["x"].iloc[-1], df["y"].iloc[-1]
                    if targets:
                        min_d = min(np.hypot(fx - tx, fy - ty) for tx, ty in targets)
                        if min_d > 0.45:
                            ax.plot(fx, fy, marker='x', color=color,
                                    markersize=12, markeredgewidth=2, zorder=4)
                            ax.plot(fx, fy, marker='o', color="black",
                                    markersize=13, markeredgewidth=1.5,
                                    markerfacecolor='none', zorder=4)
                        else:
                            ax.plot(fx, fy, marker='o', color=color,
                                    markersize=4, markeredgewidth=2, zorder=4)
                except Exception:
                    pass

                seen_files.add(fpath)
                any_new = True

            if any_new:
                rebuild_legend()
                fig.canvas.draw()
                fig.canvas.flush_events()
                last_new_time = time.time()

            if stop_after_idle is not None and (time.time() - last_new_time) >= stop_after_idle:
                break

            plt.pause(poll_interval)

    except KeyboardInterrupt:
        pass

    if getattr(rl_copp_obj.args, "save_plots", False):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"plot_scene_trajs_streaming_{timestamp}.png"
        fig.savefig(filename, dpi=150)
        plt.close(fig)



def compare_models_boxplots(rl_copp_obj, model_ids):
    """
    Compare multiple trained models across several testing metrics using boxplots and bar charts.

    This function loads the test results of different models and creates visual comparisons 
    for metrics such as reward, episode time, target zone distribution, crash rate, and speed profiles.

    Args:
        rl_copp_obj (RLCoppeliaManager): Manager instance containing paths and CLI arguments.
        model_ids (list): List of model IDs to compare.
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import logging
    import os
    import glob

    # List of metrics to be plotted
    metrics = [
        "Time (s)", 
        "Reward", 
        "Target zone", 
        "Crashes", 
        "Linear speed", 
        "Angular speed", 
        "Distance traveled (m)"
    ]
    combined_data = []
    model_action_times = []
    timestep_values = []

    # Get training CSV path to later extract timesteps
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path, "train_records.csv")

    # Load data from all models
    for model_id in model_ids:
        subfolder_pattern = f"{rl_copp_obj.args.robot_name}_model_{model_id}_*_testing"
        
        # Load main testing data
        file_pattern = f"{rl_copp_obj.args.robot_name}_model_{model_id}_*_test_*.csv"
        files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "testing_metrics", subfolder_pattern, file_pattern))
        if not files:
            logging.error(f"[!] File not found for model {model_id}")
            continue

        # Retrieve action time from training CSV
        model_name = f"{rl_copp_obj.args.robot_name}_model_{model_id}"
        timestep = float(utils.get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)"))
        model_action_times.append(f"{timestep}s")
        timestep_values.append(timestep)

        for file in files:
            df = pd.read_csv(file)
            df["Model"] = f"{timestep}s"
            df["Timestep"] = timestep
            combined_data.append(df)

        # Load additional metrics (e.g., speeds)
        other_pattern = f"{rl_copp_obj.args.robot_name}_model_{model_id}_*_otherdata_*.csv"
        files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "testing_metrics", subfolder_pattern, other_pattern))
        if not files:
            logging.error(f"[!] Other data file not found for model {model_id}")
            continue

        for file in files:
            df = pd.read_csv(file)
            df["Model"] = f"{timestep}s"
            df["Timestep"] = timestep
            df["Angular speed"] = df["Angular speed"].abs()  # Use absolute value for angular speed
            combined_data.append(df)

    if not combined_data:
        logging.error("[!] No data loaded.")
        return

    full_df = pd.concat(combined_data, ignore_index=True)

    for metric in metrics:
        if metric not in full_df.columns:
            logging.warning(f"[!] Column '{metric}' not found in the dataset.")
            continue

        # Plot each metric type
        if metric == "Reward":
            print("na")
            # fig, ax = utils.plot_metric_boxplot_by_timestep(full_df, metric, ylabel="Reward")

        # elif metric == "Time (s)":
        #     fig, ax = utils.plot_metric_boxplot_by_timestep(full_df, metric, ylabel="Average episode duration (s)")

        # elif metric == "Linear speed":
        #     fig, ax = utils.plot_metric_boxplot_by_timestep(full_df, metric, ylabel="Linear speed (m/s)")

        # elif metric == "Angular speed":
        #     fig, ax = utils.plot_metric_boxplot_by_timestep(full_df, metric, ylabel="Angular speed (rad/s)")

        # elif metric == "Distance traveled (m)":
        #     fig, ax = utils.plot_metric_boxplot_by_timestep(full_df, metric, ylabel="Distance traveled (m)")

        elif metric == "Target zone":
            # Filter valid target zone values
            df_target = full_df[full_df[metric].notna() & (full_df[metric] != 0)]
            model_order = [str(mid) for mid in model_action_times]

            # Compute percentage of each zone reached
            zone_counts = df_target.groupby(["Model", "Target zone"]).size().reset_index(name='count')
            totals = full_df[full_df[metric].notna()].groupby("Model").size().reset_index(name='total')
            zone_percents = pd.merge(zone_counts, totals, on="Model")
            zone_percents["Zone percentage (%)"] = 100 * zone_percents["count"] / zone_percents["total"]

            # Get color map
            color_map = utils.get_color_map(len(model_ids)) 

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                ax=ax,
                data=zone_percents,
                x="Target zone",
                y="Zone percentage (%)",
                hue="Model",
                hue_order=model_order,
                palette=color_map[:len(model_order)]
            )
            ax.set_xlabel("Target zone", fontsize=20)
            ax.set_ylabel("Probability (%)", fontsize=20, labelpad=0)
            ax.tick_params(axis='both', which='major', labelsize=16)
            
            ax.legend(title="Timestep", ncol=2, fontsize=14, title_fontsize=18, loc='upper left', bbox_to_anchor=(1, 1))
            ax.grid(True)
            fig.tight_layout()
            # plt.savefig("fig_type3font.svg", dpi=300, bbox_inches="tight")
            
        elif metric == "Crashes":
            # Normalize boolean values to True/False
            full_df[metric] = full_df[metric].astype(str).str.strip().str.lower().map({
                "true": True, "1": True, "yes": True,
                "false": False, "0": False, "no": False
            })

            crash_pct = (
                full_df.groupby("Timestep")[metric]
                .mean()
                .mul(100)
                .rename("Collision Rate")
                .reset_index()
            )

            # Orden numérico de los timesteps
            ts_order = sorted(crash_pct["Timestep"].unique())

        
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(ax = ax, data=crash_pct, x="Timestep", y="Collision Rate",order=ts_order)
            ax.set_xlabel("Timestep (s)", fontsize=20, labelpad=10)
            ax.set_xticklabels([f"{v:.2f}" for v in ts_order], rotation=90)
            ax.set_ylabel("Episodes with collision (%)", fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.grid(True)
            fig.tight_layout()

        # Show/save the plot
        if rl_copp_obj.args.save_plots:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"boxplot_{metric}_{timestamp}.png"
            fig.savefig(filename)
            plt.close()
        else:
            plt.show()

    
def plot_lat_curves(rl_copp_obj, model_index):
    """
    Plots LAT (Latency) curves for both simulation and real wall-clock time.

    Args:
        rl_copp_obj (RLCoppeliaManager): Object containing the base path and parsed arguments.
        model_index (int): Index of the model to process within the model_ids list.

    Returns:
        None
    """
    # Get the training csv path for later getting the action times from there
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path,"train_records.csv")    # Name of the train records csv to search the algorithm used

    # Get action time 
    model_id = rl_copp_obj.args.model_ids[model_index]
    model_name = f"{rl_copp_obj.args.robot_name}_model_{model_id}"
    robot_name = rl_copp_obj.args.robot_name

    # Locate LAT CSV files
    if rl_copp_obj.args.lat_file_path is None:
        file_pattern = f"{model_name}_*_otherdata_*.csv"
        subfolder_pattern = f"{model_name}_*_testing"
        files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", robot_name, "testing_metrics", subfolder_pattern, file_pattern))
    else:
        files = rl_copp_obj.args.lat_file_path

    for lat_file in files:
        # Read CSV
        logging.info(f"Reading file: {lat_file}")
        df = pd.read_csv(lat_file)

        # Remove the first row as LAT is 0
        df = df.iloc[1:].reset_index(drop=True)
        
        # Just show the first 100 episodes as it is enough for the visualization
        if 'Episode number' in df.columns:
            df = df[df['Episode number'] <= 100]
        
        if rl_copp_obj.args.lat_fixed_timestep == 0:
            try:
                timestep = (utils.get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)"))
            except Exception as e:
                logging.error(f"You probably forgot to specify the timestep by --lat_fixed_timestep argument, please check it. Error message: {e}")

            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df["LAT-Sim (s)"], label="LAT-Agent (s)", color='tab:blue', linewidth=2, zorder = 3)
            plt.plot(df.index, df["LAT-Wall (s)"], label="LAT-Wall (s)", color='tab:orange', linewidth=2, zorder = 2)
            
        else:    # This case is used for some special cases in which we don't have the model data in the train_records file
            timestep = rl_copp_obj.args.lat_fixed_timestep
            timestep_unit = rl_copp_obj.args.timestep_unit
            if timestep_unit == "s":
                factor =  1
            elif timestep_unit == "ms":
                factor =  1000
            elif timestep_unit == "us":
                factor =  1000000

            # Detect the correct LAT column name (could be LAT-Wall or any column containing 'LAT')
            lat_col = None
            for col in df.columns:
                if "LAT" in col:
                    lat_col = col
                    break
            if lat_col is not None:
                df[lat_col] = df[lat_col] / factor  # Convert LAT unit
            else:
                logging.error("LAT column not found in the CSV file.")
                raise ValueError("LAT column not found in the CSV file.")

            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df[lat_col], label="LAT (s)", linewidth=1.5, zorder = 2)    
            
        # Draw a horizontal line for the timestep value
        plt.axhline(float(timestep), color="#245000", linestyle='-.', linewidth=2, label=f"Timestep = {timestep}s",xmin=0)

        # Plot configuration
        plt.xlabel("Steps", fontsize = 22, labelpad=12)
        plt.ylabel("LAT (s)", fontsize = 22, labelpad=12)
        plt.tick_params(axis='both', which='major', labelsize=18)
        # plt.title(f"LAT-Sim and LAT-Wall vs. Steps - Model {timestep}s")
        plt.legend(fontsize=18) # bbox_to_anchor=(1, 0.5)
        plt.grid(True)
        plt.tight_layout()

        # Show/save the plot
        if rl_copp_obj.args.save_plots:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"lat_curves_{timestep}_{timestamp}.png"
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()


def plot_reward_comparison_from_csv(rl_copp_obj, x_axis_name="Steps"):
    """
    Plots a comparison of rewards vs steps for multiple models from their respective CSV files.

    This function has been created for a custom case, in which the name of the csv files follows this
    patter: '<model>ms.csv', and <model> corresponds to the timestep. CSV files are located inside <robot_name>/Training/.

    Args:
        rl_copp_obj (RLCoppeliaManager): Instance of RLCoppeliaManager class to access paths and arguments.
        x_axis_name (str): Name of the x axis to plot. Options are "Steps", "Time (s)", "Episodes". Default is "Steps".
    """
    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Iterate over each model ID
    for model in rl_copp_obj.args.model_ids:
        # Construct the CSV file path
        csv_path = os.path.join(
            rl_copp_obj.base_path,
            "robots",
            rl_copp_obj.args.robot_name,
            "Training",
            f"{model}.csv"
        )

        # Check if the file exists
        if not os.path.isfile(csv_path):
            csv_path = csv_path.replace(".csv", "ms.csv")
            if not os.path.isfile(csv_path):
                logging.error(f"[Error] File not found: {csv_path}")
                continue

        # Load the CSV into a DataFrame
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logging.error(f"[Error] Failed to read CSV for model {model}: {e}")
            continue

        # Check required columns
        required_columns = ["Reward", x_axis_name]
        if not all(col in df.columns for col in required_columns):
            logging.error(f"[Error] CSV for model {model} must contain the following columns: {required_columns}")
            continue

        # Plot Reward vs Steps for the current model
        plt.plot(
            df[x_axis_name],
            df["Reward"],
            label=f"Model {model}ms",
            linestyle='-'
        )

    # Add labels, legend, and grid
    plt.xlabel(x_axis_name, fontsize=18, labelpad=10)
    plt.ylabel("Reward", fontsize=18, labelpad=6)
    # plt.title("Reward Comparison Across Models", fontsize=16)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    # Show/save the plot
    if rl_copp_obj.args.save_plots:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"reward_comparison_from_csv_{x_axis_name}_{timestamp}.png"
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_boxplots_from_csv(rl_copp_obj):
    """
    Plots boxplots for comparing Reward and Time (s) across multiple models. 

    This function has been created for a custom case, in which the name of the csv files follows this
    patter: '<model>ms.csv', and <model> corresponds to the timestep. CSV files are located inside <robot_name>/Explotation/.


    Args:
        rl_copp_obj (RLCoppeliaManager): Instance of RLCoppeliaManager class to access paths and arguments.
    """
    # Initialize a list to store data for all models
    combined_data = []

    # Iterate over each model ID
    for model in rl_copp_obj.args.model_ids:
        # Construct the CSV file path
        csv_path = os.path.join(
            rl_copp_obj.base_path,
            "robots",
            rl_copp_obj.args.robot_name,
            "Explotation",
            f"{model}.csv"
        )

        # Check if the file exists
        if not os.path.isfile(csv_path):
            csv_path = csv_path.replace(".csv", "ms.csv")
            if not os.path.isfile(csv_path):
                logging.error(f"[Error] File not found: {csv_path}")
                continue

        # Load the CSV into a DataFrame
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logging.error(f"[Error] Failed to read CSV for model {model}: {e}")
            continue

        # Check required columns
        required_columns = ["Reward", "Time (s)"]
        if not all(col in df.columns for col in required_columns):
            logging.error(f"[Error] CSV for model {model} must contain the following columns: {required_columns}")
            continue

        # Add a column to identify the model
        df["Model"] = model

        # Append the DataFrame to the combined data list
        combined_data.append(df)

    # Combine all data into a single DataFrame
    if not combined_data:
        logging.error("[Error] No valid data found for the specified models.")
        return

    combined_df = pd.concat(combined_data, ignore_index=True)


    def plot_metric(metric_name, ylabel):
        timesteps = sorted(combined_df["Model"].unique())
        data_per_timestep = [combined_df[combined_df["Model"] == t][metric_name].values for t in timesteps]

        plt.figure(figsize=(8, 6))
        plt.boxplot(data_per_timestep, positions=timesteps, widths=4, patch_artist=True)

        medians = [np.median(vals) for vals in data_per_timestep]

        # Interpolate spline
        if len(timesteps) > 2:
            x_smooth = np.linspace(min(timesteps), max(timesteps), 300)
            spline = make_interp_spline(timesteps, medians, k=2)
            y_smooth = spline(x_smooth)
            plt.plot(x_smooth, y_smooth, color='blue', linestyle='--', label='Spline')

            # Find best point
            opt_idx = np.argmax(y_smooth)
            opt_x = x_smooth[opt_idx]
            opt_y = y_smooth[opt_idx]

            plt.plot(opt_x, opt_y, marker='x', color='green', markersize=10, markeredgewidth=3, label=f'Optimal: {opt_x:.1f} ms')

        # Median points
        plt.scatter(timesteps, medians, marker='x', color='blue', s=60, zorder=5)

        # Labels
        plt.xlabel("Timestep (ms)", fontsize=18, labelpad=12)
        plt.ylabel(ylabel, fontsize=18, labelpad=12)
        plt.xticks(timesteps, [str(int(t)) for t in timesteps], fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.legend(fontsize = 16, loc='upper right')
        plt.tight_layout()

        # Show/save the plot
        if rl_copp_obj.args.save_plots:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"boxplot_from_csv_{ylabel}_{timestamp}.png"
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    # ---------- Plots ----------
    plot_metric("Time (s)", "Balancing time (ms)")
    plot_metric("Reward", "Reward")
    


def plot_boxplots_from_csv_dual_axis(rl_copp_obj):
    """
    Plots superimposed boxplots of Reward and Balancing Time for multiple models on a shared X-axis.

    This function reads CSV files corresponding to different model timesteps, each containing
    the metrics 'Reward' and 'Time (s)'. It then draws:
    Two Y-axes are used: left for Reward, right for Balancing Time.

    Args:
        rl_copp_obj (RLCoppeliaManager): An instance of the RLCoppeliaManager class.

    Returns:
        None. The function displays the plot or saves it to disk depending on `save_plots`.
    """

    combined_data = []

    for model in rl_copp_obj.args.model_ids:
        csv_path = os.path.join(
            rl_copp_obj.base_path,
            "robots",
            rl_copp_obj.args.robot_name,
            "Explotation",
            f"{model}.csv"
        )

        if not os.path.isfile(csv_path):
            csv_path = csv_path.replace(".csv", "ms.csv")
            if not os.path.isfile(csv_path):
                logging.error(f"[Error] File not found: {csv_path}")
                continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logging.error(f"[Error] Failed to read CSV for model {model}: {e}")
            continue

        if not all(col in df.columns for col in ["Reward", "Time (s)"]):
            logging.error(f"[Error] CSV for model {model} must contain 'Reward' and 'Time (s)'")
            continue

        df["Model"] = model
        combined_data.append(df)

    if not combined_data:
        logging.error("[Error] No valid data found.")
        return

    combined_df = pd.concat(combined_data, ignore_index=True)
    timesteps = sorted(combined_df["Model"].unique())
    positions = np.arange(len(timesteps)) * 10

    rewards_data = [combined_df[combined_df["Model"] == t]["Reward"].values for t in timesteps]
    times_data = [combined_df[combined_df["Model"] == t]["Time (s)"].values for t in timesteps]
    rewards_medians = [np.median(r) for r in rewards_data]
    times_medians = [np.median(t) for t in times_data]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Boxplots 
    ax1.boxplot(rewards_data, positions=positions, widths=4, patch_artist=True,
                boxprops=dict(facecolor="skyblue", alpha=0.6),
                medianprops=dict(color="blue"))

    ax2 = ax1.twinx()
    ax2.boxplot(times_data, positions=positions, widths=4, patch_artist=True,
                boxprops=dict(facecolor="lightgreen", alpha=0.5),
                medianprops=dict(color="green"))

    # Spline for reward
    if len(positions) > 2:
        spline_r = make_interp_spline(positions, rewards_medians, k=2)
        x_smooth = np.linspace(min(positions), max(positions), 300)
        y_smooth_r = spline_r(x_smooth)
        ax1.plot(x_smooth, y_smooth_r, linestyle='--', color='darkblue', label='Reward spline')
        ax1.scatter(positions, rewards_medians, color='darkblue', marker='x', s=60)

        opt_idx_r = np.argmax(y_smooth_r)
        opt_x_r = x_smooth[opt_idx_r]
        opt_y_r = y_smooth_r[opt_idx_r]
        opt_ts_r = np.interp(opt_x_r, positions, timesteps)
        ax1.plot(opt_x_r, opt_y_r, marker='o', color='darkblue',markersize=10, markeredgewidth=3,
                 label=f'Optimal: {opt_ts_r:.2f} ms')

    # Spline for balancing time
    if len(positions) > 2:
        spline_t = make_interp_spline(positions, times_medians, k=2)
        y_smooth_t = spline_t(x_smooth)
        ax2.plot(x_smooth, y_smooth_t, linestyle='--', color='darkgreen', label='Balancing time spline')
        ax2.scatter(positions, times_medians, color='darkgreen', marker='x', s=60)

        opt_idx_t = np.argmax(y_smooth_t)
        opt_x_t = x_smooth[opt_idx_t]
        opt_y_t = y_smooth_t[opt_idx_t]
        opt_ts_t = np.interp(opt_x_t, positions, timesteps)
        ax2.plot(opt_x_t, opt_y_t, marker='o', color='darkgreen',markersize=10, markeredgewidth=3,
                 label=f'Optimal: {opt_ts_t:.2f} ms')

    # Plot format
    ax1.set_xlabel("Timestep (ms)", fontsize=20, labelpad=12)
    ax1.set_ylabel("Reward", fontsize=20, color='darkblue', labelpad=2)
    ax2.set_ylabel("Balancing time (s)", fontsize=20, color='darkgreen', labelpad=14)
    ax1.tick_params(axis='y', labelsize=18, labelcolor='darkblue')
    ax2.tick_params(axis='y', labelsize=18, labelcolor='darkgreen')
    ax1.set_xticks(positions)
    ax1.set_xticklabels([str(int(t)) for t in timesteps], fontsize=18)

    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=14, bbox_to_anchor=(0.87, 0.99))
    plt.tight_layout()

    plt.savefig("fig_zoom.pdf", dpi=300, bbox_inches="tight")

    # Save plot
    if rl_copp_obj.args.save_plots:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"dual_boxplot_superpuesto_{timestamp}.png"
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()





def plot_convergence_points_comparison(rl_copp_obj, convergence_points_by_model):
    """
    Plots four line charts (one for each convergence metric: WallTime, Steps, SimTime, Episodes),
    each showing the convergence value for all models.

    Args:
        rl_copp_obj: RLCoppeliaManager instance.
        convergence_points_by_model (dict): Dict of {model_id: [walltime, steps, simtime, episodes]}.
    """
    if not convergence_points_by_model:
        logging.warning("No convergence values to plot.")
        return

    metrics = ["WallTime", "Steps", "SimTime", "Episodes"]
    model_ids = list(convergence_points_by_model.keys())
    values = np.array([convergence_points_by_model[mid] for mid in model_ids])

    # Get action times for x-tick labels
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path, "train_records.csv")
    action_times = []
    for mid in model_ids:
        model_name = rl_copp_obj.args.robot_name + "_model_" + str(mid)
        timestep = utils.get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)")
        action_times.append(float(timestep))

    # Sort by action_times for better visualization
    sorted_indices = np.argsort(action_times)
    action_times = np.array(action_times)[sorted_indices]
    values = values[sorted_indices]
    model_ids = np.array(model_ids)[sorted_indices]

    for i, metric in enumerate(metrics):
        plt.figure(figsize=(8, 6))
        plt.plot(action_times, values[:, i], marker='o', linewidth=2)
        plt.xlabel("Timestep (s)", fontsize=16)
        plt.ylabel(f"Convergence {metric}", fontsize=16)
        # plt.title(f"Convergence {metric} vs Timestep", fontsize=18)
        plt.xticks(action_times, [f"{t}" for t in action_times], fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        if rl_copp_obj.args.save_plots:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"convergence_comparison_{i}_{timestamp}.png"
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()


def plot_lat(rl_copp_obj, model_index, mode="speed"):
    """
    Plots the linear speed and LAT-Agent from the otherdata CSV file for a specific model index.
    The plot includes vertical lines indicating the start of each episode based on the test data.
    Args:
        rl_copp_obj (RLCoppeliaManager): Instance of RLCoppeliaManager class to access paths and arguments.
        model_index (int): Index of the model to plot data for.
        mode (str): Mode of the first subplot, either "speed" for linear speed or "distance" for distance traveled. Default is "speed".

    Returns:
        None: Displays the plot with linear speed and LAT-Agent over steps, with episode markers.
    """
    mask = None
    episode_numbers = None
    
    # Get the training csv path for later getting the action times from there
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path,"train_records.csv")    # Name of the train records csv to search the algorithm used
    
    # Get action time 
    model_name = rl_copp_obj.args.robot_name + "_model_" + str(rl_copp_obj.args.model_ids[model_index])
    timestep = (utils.get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)"))

    # CSV File path to get data from
    # Capture the desired files through a pattern
    file_pattern_otherdata = f"{rl_copp_obj.args.robot_name}_model_{rl_copp_obj.args.model_ids[model_index]}_*_otherdata_*.csv"
    file_pattern_testdata = f"{rl_copp_obj.args.robot_name}_model_{rl_copp_obj.args.model_ids[model_index]}_*_test_*.csv"
    subfolder_pattern = f"{rl_copp_obj.args.robot_name}_model_{rl_copp_obj.args.model_ids[model_index]}_*_testing"
    files_otherdata = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "testing_metrics", subfolder_pattern, file_pattern_otherdata))
    files_testdata = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "testing_metrics", subfolder_pattern, file_pattern_testdata))

    # User selects the experiment to plot if there are multiple files
    path_otherdata, path_test = utils.map_files_by_timestamp(files_otherdata, files_testdata)

    # Read CSV
    df_otherdata = pd.read_csv(path_otherdata)
    df_testdata = pd.read_csv(path_test)

    # Remove whitespace and ensure column names are clean
    df_otherdata.columns = df_otherdata.columns.str.strip()

    # Remove the first row as LAT is 0
    df_otherdata = df_otherdata.iloc[1:].reset_index(drop=True)

    steps = df_otherdata.index  # Each row represents a step
    timestep = float(timestep)

    # Create mask depending on the timestep
    if timestep >= 0.45:
        mask = (df_testdata['Target zone'] == 3) & (df_testdata['TimeSteps count'] > 1)
    elif timestep >=float(0.15) and timestep <=0.3:
        mask = (df_testdata['Target zone'] == 2) & (df_testdata['TimeSteps count'] > 1)
    
    if mask is not None:
        filtered = df_testdata[mask]
        episode_numbers = (filtered.index + 1).tolist()
        logging.info(f"Episodes that reach the next target ring: {episode_numbers}")

    # Create two subfigures in a single figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Subfigure 1: Linear speed or Distance traveled, depending on the mode selected.
    if mode == "speed":
        ax1.plot(steps, df_otherdata['Linear speed'], color="#004575", label='Linear speed (m/s)', linewidth=1.5)  # High-contrast blue
        ax1.set_ylabel('Linear speed (m/s)', color="#004575", fontsize=24, labelpad=22)
    elif mode == "distance":
        ax1.plot(steps, df_otherdata['Linear speed']*df_otherdata['LAT-Sim (s)'], color="#004575", label='Distance (m)', linewidth=1.5)  # High-contrast blue
    ax1.set_ylabel('Distance (m)', color="#004575", fontsize=24, labelpad=22)

    ax1.tick_params(axis='y', labelcolor="#004575")
    ax1.tick_params(axis='both', which='major', labelsize=20, pad=10)
    ax1.grid(True)
    ax1.legend(loc='upper right', fontsize=18)
    ymin, ymax = ax1.get_ylim() # Amplify y-axis limits by 20% for better visualization
    ax1.set_ylim(ymin - 0.2 * abs(ymax - ymin), ymax + 0.2 * abs(ymax - ymin))

    # Subfigure 2: LAT-Agent
    ax2.plot(steps, df_otherdata['LAT-Sim (s)'], color="#8d4400", label='LAT-Agent (s)', alpha=0.8, linewidth=2)  # High-contrast red
    ax2.set_ylabel('LAT-Agent (s)', color='#8d4400', fontsize=24, labelpad=10)
    ax2.tick_params(axis='y', labelcolor="#8d4400")
    ax2.set_xlabel('Steps', fontsize=24, labelpad=10)
    ax2.tick_params(axis='both', which='major', labelsize=20, pad=10)

    # Draw a horizontal line for the timestep value
    ax2.axhline(float(timestep), color="#245000", linestyle='-.', linewidth=2, label=f"Timestep = {timestep}s")
    ax2.grid(True)
    ax2.legend(loc='upper right', fontsize=18)
    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim(ymin - 0.2 * abs(ymax - ymin), ymax + 0.2 * abs(ymax - ymin))
    
    # Plot vertical lines at the start of each episode in which the robot reaches the next unreachable target ring
    if episode_numbers is not None:
        for ep in episode_numbers:
            # Find the rows where 'Episode number' matches the current episode
            mask = (df_otherdata['Episode number'] == ep)
            if not mask.any():
                continue
            last_idx = mask[mask].index[-1] # Get the last index where the episode finishes
            # If the last index is greater than the length of df_otherdata, use the last index
            # Otherwise, use the last index as it is. This ensures that we don't try to access an index that is out of bounds
            step_to_mark = last_idx if last_idx < len(df_otherdata) else last_idx
            ax1.axvline(step_to_mark, color='red', linestyle='--', linewidth=1.5)
            ax2.axvline(step_to_mark, color='red', linestyle='--', linewidth=1.5)

    # Title and layout
    # plt.suptitle('Linear Speed and LAT-Sim over Steps', fontsize=16)
    plt.tight_layout()

    # Ask user to set limits for x-y axes. This is useful when we want to zoom in a specific area of the plot
    utils.ask_and_set_limits_for_axes(ax1, "Linear speed (m/s)")
    utils.ask_and_set_limits_for_axes(ax2, "LAT-Agent (s)")

    # Show/save the plot
    if rl_copp_obj.args.save_plots:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"speed_and_lat_{rl_copp_obj.args.model_ids[model_index]}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def main(args):
    """
    Main function for generating various plots to analyze and compare the performance of trained models.

    This function processes user-specified plot types and generates corresponding visualizations 
    for analyzing the performance of reinforcement learning models trained in CoppeliaSim. 
    It supports multiple plot types, including convergence analysis, reward comparisons, 
    trajectory visualizations, and more.

    Args:
        args (Namespace): Parsed command-line arguments containing the following attributes:
            - robot_name (str): Name of the robot being analyzed.
            - model_ids (list): List of model IDs to analyze or compare.
            - plot_types (list): List of plot types to generate (e.g., "spider", "convergence-walltime").
            - scene_to_load_folder (str, optional): Path to the folder containing scene configurations 
              (required for trajectory plots).
            - save_plots (bool, optional): If False, the graphs will be shown. If True, graphs will be automatically saved.
            - verbose (int, optional): Verbosity level for logging.
    Raises:
        SystemExit: If a required argument (e.g., scene_to_load_folder for trajectory plots) is missing.
    Returns:
        None
    """

    rl_copp = RLCoppeliaManager(args)

    if rl_copp.args.save_plots:
        matplotlib.use('Agg')  # Use a non-GUI backend suitable for script-based PNG saving

    plot_type_correct = False

    if "spider" in args.plot_types:
        plot_type_correct = True
        if len(args.model_ids) <= 1:    # A spider graph doesn't make sense if there are less than 2 models. In fact it doesn't make sense to compare less than 3 models (#TODO)
            logging.error(f"Please, introduce more than one model ID for creating a spider graph. Models specified: {args.model_ids}")
        
        else:
            logging.info(f"Plotting spider graph for comparing the models {args.model_ids}")
            plot_spider(rl_copp)

    if "convergence-walltime" in args.plot_types:
        plot_type_correct = True
        for model in range(len(args.model_ids)):
            logging.info(f"Plotting convergence-vs-wall-time graph for model {args.model_ids[model]}")
            plot_convergence(rl_copp, model, "WallTime")
    
    if "convergence-steps" in args.plot_types:
        plot_type_correct = True
        for model in range(len(args.model_ids)):
            logging.info(f"Plotting convergence-vs-steps graph for model {args.model_ids[model]}")
            plot_convergence(rl_copp, model, "Steps")

    if "convergence-simtime" in args.plot_types:
        plot_type_correct = True
        for model in range(len(args.model_ids)):
            logging.info(f"Plotting convergence-vs-simtime graph for model {args.model_ids[model]}")
            plot_convergence(rl_copp, model, "SimTime")

    if "convergence-episodes" in args.plot_types:
        plot_type_correct = True
        for model in range(len(args.model_ids)):
            logging.info(f"Plotting convergence-vs-episodes graph for model {args.model_ids[model]}")
            plot_convergence(rl_copp, model, "Episodes")

    if "convergence-all" in args.plot_types:
        plot_type_correct = True
        # Global dictionary to hold convergence points for all models
        convergence_points_by_model = {}

        for model in range(len(args.model_ids)):
            logging.info(f"Plotting all the convergence graphs for model {args.model_ids[model]}")
            convergence_modes = ["WallTime", "Steps", "SimTime", "Episodes"]
            convergence_points = []

            # Collect convergence values per mode
            for conv_mode in convergence_modes:
                conv_x = plot_convergence(rl_copp, model, conv_mode, show_plots = False)
                convergence_points.append(conv_x)
            
            # Save in the dictionary
            convergence_points_by_model[args.model_ids[model]] = convergence_points

        plot_convergence_points_comparison(rl_copp, convergence_points_by_model)

    if "compare-rewards" in args.plot_types:
        plot_type_correct = True
        if len(args.model_ids) <= 1:    
            logging.error(f"Please, introduce more than one model ID for creating a rewards-comparison graph. Models specified: {args.model_ids}")
        else:
            logging.info(f"Plotting rewards-comparison graph for comparing the models {args.model_ids}")
            plot_metrics_comparison(rl_copp, "rewards")
        
    if "compare-episodes_length" in args.plot_types:
        plot_type_correct = True
        if len(args.model_ids) <= 1:    
            logging.error(f"Please, introduce more than one model ID for creating a episodes-length-comparison graph. Models specified: {args.model_ids}")
        else:
            logging.info(f"Plotting episodes-length-comparison graph for comparing the models {args.model_ids}")
            plot_metrics_comparison(rl_copp, "episodes_length")

    if "compare-convergences" in args.plot_types:
        plot_type_correct = True
        if len(args.model_ids) <= 1:    
            logging.error(f"Please, introduce more than one model ID for creating a convergences-comparison graph. Models specified: {args.model_ids}")
        else:
            logging.info(f"Plotting convergences-comparison graph for comparing the models {args.model_ids}")
            plot_convergence_comparison(rl_copp)
    
    if "convergence_cloud" in args.plot_types:
        plot_type_correct = True
        if len(args.model_ids) <= 1:    
            logging.error(f"Please, introduce more than one model ID for creating a convergence-cloud graph. Models specified: {args.model_ids}")
        else:
            logging.info(f"Plotting convergence cloud graph for comparing the models {args.model_ids}")
            plot_convergence_comparison_cloud(rl_copp)

    if "grouped_bar_speeds" in args.plot_types:
        plot_type_correct = True
        logging.info(f"Plotting grouped bar chart for angular and linear speeds for models {args.model_ids}")
        plot_grouped_bar_chart(rl_copp, mode="speeds")
    
    if "grouped_bar_targets" in args.plot_types:
        plot_type_correct = True
        logging.info(f"Plotting grouped bar chart representing the frequency of each target zone for models {args.model_ids}")
        plot_grouped_bar_chart(rl_copp, mode="target_zones")

    if "plot_scene_trajs" in args.plot_types:
        plot_type_correct = True
        logging.info(f"Plotting a scene image with the trajectories followed by next models: {args.model_ids}")
        if (args.scene_to_load_folder) is None:
            logging.error("Scene config path was not provided, program will exit as it cannot continue.")
            sys.exit()
        scene_folder = os.path.join(rl_copp.paths["scene_configs"], args.scene_to_load_folder)
        logging.info(f"Scene config folder to be loaded: {scene_folder}")
        plot_scene_trajs(rl_copp, scene_folder)

    if "plot_scene_trajs_streaming" in args.plot_types:
        plot_type_correct = True
        logging.info(f"Plotting a scene image with the trajectories followed by next models: {args.model_ids}")
        if (args.scene_to_load_folder) is None:
            logging.error("Scene config path was not provided, program will exit as it cannot continue.")
            sys.exit()
        scene_folder = os.path.join(rl_copp.paths["scene_configs"], args.scene_to_load_folder)
        logging.info(f"Scene config folder to be loaded: {scene_folder}")
        plot_scene_trajs_streaming(rl_copp, scene_folder)
    
    if "plot_boxplots" in args.plot_types:
        plot_type_correct = True
        logging.info(f"Plotting boxplots for models {args.model_ids}")
        compare_models_boxplots(rl_copp, args.model_ids)

    if "lat_curves" in args.plot_types:
        plot_type_correct = True
        for model in range(len(args.model_ids)):
            logging.info(f"Plotting curves for LAT-sim and LAT-wall for model {args.model_ids[model]}")
            plot_lat_curves(rl_copp, model)

    if "speed_lat_curves" in args.plot_types:
        plot_type_correct = True
        for model in range(len(args.model_ids)):
            logging.info(f"Plotting curves for speed and LAT-sim for model {args.model_ids[model]}")
            plot_lat(rl_copp, model, mode="speed")

    if "dist_lat_curves" in args.plot_types:
        plot_type_correct = True
        for model in range(len(args.model_ids)):
            logging.info(f"Plotting curves for distance and LAT-sim for model {args.model_ids[model]}")
            plot_lat(rl_copp, model, mode="distance")

    if "plot_from_csv" in args.plot_types:
        plot_type_correct = True
        if len(args.model_ids)>1:
            logging.info(f"Plotting a reward comparison graph for models {args.model_ids} for robot {args.robot_name}")
            plot_reward_comparison_from_csv(rl_copp, "Episodes")
            plot_boxplots_from_csv(rl_copp)
            plot_boxplots_from_csv_dual_axis(rl_copp)
    
    if not plot_type_correct:
        logging.error(f"Please check plot types: {args.plot_types}")