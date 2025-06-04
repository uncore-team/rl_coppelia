"""
Project: Robot Training and Testing RL Algorithms in CoppeliaSim
Author: Adrián Bañuls Arias
Version: 1.0
Date: 2025-03-25
License: GNU General Public License v3.0

Description: 
    Plot different kind of graphs for checking the performance of a trained model or for comparing
    several ones using metrics like reward of episode length.
    
Usage:
    rl_coppelia plot --robot_name <robot_name> --model_ids <model_ids> 
                                [--plot_types <str>] [--verbose <num>]

Features:
    - Automatically creates required directories if they do not exist.
    - Runs a testing session either sequentially or in parallel with a delay between submissions.
    - Saves a summary of testing results in a CSV file, along with some plots comparing the obtained metrics.
"""
import matplotlib
from collections import defaultdict
import glob
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

import pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from common import utils
from common.rl_coppelia_manager import RLCoppeliaManager
from pandas.api.types import CategoricalDtype
from scipy.interpolate import interp1d
from matplotlib.patches import Ellipse
from scipy.stats import shapiro,gaussian_kde
from sklearn.covariance import MinCovDet
from scipy.interpolate import make_interp_spline


def plot_spider(rl_copp_obj, title='Models Comparison'):
    """
    Plots multiple spider charts on the same figure to compare different models.

    Args:
    - rl_copp_object (RLCoppeliaManager): Instance of RLCoppeliaManager class just for managing the args and the base path.
    - title (str): The title of the chart.
    """

    # Categories for plotting
    # categories = [
    #     "Convergence Sim Time",
    #     "Actor Performance",
    #     "Critic Performance",
    #     # "Train Episode Efficiency",
    #     # "Train Reward",
    #     "Test Reward",
    #     "Test Episode Efficiency",
    #     "Test Episode Completion Rate",  
    # ]
    categories = [
        "Learning_Convergence",
        "Mean_Reward",
        "Episode_Efficiency",
        "Episode_Completion Rate",  
        "Trajectory_Optimality",
        "Innermost Target_Rate"
    ]

    # Get metrics from testing
    testing_csv_path = os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "testing_metrics", "test_records.csv")
    test_column_names = [
        "Avg reward",
        "Avg time reach target",
        "Percentage terminated",
        "Avg episode distance (m)",
        "Target zone 3 (%)"
    ]
    test_data = utils.get_data_for_spider(testing_csv_path, rl_copp_obj.args, test_column_names)

    # Get metrics from training
    training_csv_path = os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "training_metrics", "train_records.csv")
    # train_column_names = [
    #     "Action time (s)",
    #     "Time to converge (h)",
    #     "train/actor_loss",
    #     "train/critic_loss",
    #     # "rollout/ep_len_mean",
    #     # "rollout/ep_rew_mean"
    # ]

    train_column_names = [
        "Action time (s)",
        "Time to converge (h)",
    ]


    train_data = utils.get_data_for_spider(training_csv_path, rl_copp_obj.args, train_column_names)

    # Concatenate the train and test DataFrames along the columns axis
    df_train = pd.DataFrame(train_data).T  # Transpose to have IDs as rows
    df_test = pd.DataFrame(test_data).T  
    concatenated_df = pd.concat([df_train, df_test], axis=1)
    concatenated_df = concatenated_df.fillna(np.nan)    # Ensure NaN fills for any missing columns in either DataFrame

    data_list, names, labels = utils.process_spider_data(concatenated_df)

    # Override the labels with the names saved in categories list
    for cat in range(len(categories)):
        try:
            labels[cat] = categories[cat]
        except: # If some tag is missing, just pass
            pass

    # Plot the spider graph
    num_vars = len(labels)  # Vars  number
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()   # Create angle for each category
    angles += angles[:1]    # Close th circle
    fig, ax = plt.subplots(figsize=(10.5, 6), subplot_kw=dict(polar=True))


    for data, name in zip(data_list, names):    # Plot each data set
        data = data + data[:1]  # Assure that we are closing the circle
        ax.plot(angles, data, linewidth=2, linestyle='solid', label=name)
        ax.fill(angles, data, alpha=0.25)

    # Labels of the axis
    labels = [label.replace("_", "\n") for label in labels]  # Replace underscores with newlines for better readability
    ax.set_yticklabels([])  # Remove labels from radial axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=18)  # Configure labels

    # Adjust the position of each label manually
    for label in ax.get_xticklabels():
        if "Convergence" in label.get_text():
            label.set_y(label.get_position()[1] - 0.3)  
        elif "Reward" in label.get_text():
            label.set_y(label.get_position()[1] - 0.15) 
        elif "Efficiency" in label.get_text():
            label.set_y(label.get_position()[1] - 0.15)
        elif "Completion" in label.get_text():
            label.set_y(label.get_position()[1] - 0.4)
        elif "Target" in label.get_text():
            label.set_y(label.get_position()[1] - 0.2)
        elif "Trajectory" in label.get_text():
            label.set_y(label.get_position()[1] - 0.2)
        else:
            label.set_y(label.get_position()[1] - 0.1)  

    # Set the radial axis limits
    ax.set_ylim(0, 1.1) 
    
    ax.spines['polar'].set_visible(False) 
    ax.spines['polar'].set_bounds(0, 1) 

    # Add the leyend and title
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.1), fontsize = 18)  # Adjust legend
    # ax.set_title(title, size=16, color='black', y=1.1)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_convergence (rl_copp_obj, model_index, x_axis, show_plots = True, title = "Reward Convergence Analysis"):
    """
    Plots a graph with the reward vs Wall time, steps, episodes or simulatino time, and shows the point at which the reward stabilizes
    (converges) based on a first-order fit.

    Args:
    - rl_copp_object (RLCoppeliaManager): Instance of RLCoppeliaManager class just for managing the args and the base path.
    - title (str): The title of the chart.
    - x_axis (str): Name of the x_axis.


    """
    # CSV File path to get data from
    file_pattern = f"{rl_copp_obj.args.robot_name}_model_{rl_copp_obj.args.model_ids[model_index]}_*.csv"
    files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "training_metrics", file_pattern))
    

    # Calculate the convergence time
    convergence_point, reward_fit, x_axis_values, reward, reward_at_convergence = utils.get_convergence_point (files[0], x_axis, convergence_threshold=0.02)
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
            filename = f"convergence_{rl_copp_obj.args.model_ids[model_index]}_{x_axis}.png"
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    return convergence_point
    

def moving_average(data, window_size=10):
    """
    Applies a moving average filter to smooth the data.

    Args:
    - data (array): Original data to be smoothed.
    - window_size (int): Window size for the moving average.

    Returns:
    - array: Smoothed data.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def plot_metrics_comparison (rl_copp_obj, metric, title = "Comparison"):
    """
    Plot the same metric of multiple models for comparing them. X axis will be the number of steps.

    Args:
    - rl_copp_object (RLCoppeliaManager): Instance of RLCoppeliaManager class just for managing the args and the base path.
    - title (str): The title of the chart.
    """

    # Get the training csv path for later getting the action times from there
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path,"train_records.csv")    # Name of the train records csv to search the algorithm used
    timestep = []
    
    for model_index in range(len(rl_copp_obj.args.model_ids)):
        # Get timestep of the selected model
        model_name = rl_copp_obj.args.robot_name + "_model_" + str(rl_copp_obj.args.model_ids[model_index])
        timestep.append(utils.get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)"))
         
        # CSV File path to get data from
        file_pattern = f"{rl_copp_obj.args.robot_name}_model_{rl_copp_obj.args.model_ids[model_index]}_*.csv"
        files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "training_metrics", file_pattern))
        
        # Read the CSV file
        df = pd.read_csv(files[0])

        # Limit max steps <= 200000 #TODO: remove this and do it automatically, by using the experiment with less steps as a limit
        mask = df['Step'] <= 200000
        df = df[mask]
        
        # Extract steps and rewards
        steps = df['Step'].values
        if metric == "rewards":
            data = df['rollout/ep_rew_mean'].values
            pre_title = "Rewards "
        elif metric == "episodes_length":
            data = df['rollout/ep_len_mean'].values
            pre_title = "Episodes Length "

        y_label = pre_title

        # Smooth the data
        window_size = 100  # Define the window size for smooth level
        smoothed_data = moving_average(data, window_size=window_size)

        # Adjust the steps to match the length of the smoothed data
        smoothed_steps = steps[:len(smoothed_data)]

        # Plot the rewards for each model
        plt.plot(smoothed_steps, smoothed_data, label=f'Model {timestep[model_index]}s')

        # Adjust x axis ticks to show every 50,000 steps
        xticks = np.arange(0, steps.max() + 100, 50000)  
        plt.xticks(xticks, fontsize=16)


    # Add labels and title
    plt.xlabel('Steps', fontsize=20, labelpad=12)

    plt.ylabel(y_label, fontsize=20, labelpad=12)

    plt.tick_params(axis='both', which='major', labelsize=16)  # Aumentar tamaño de los números del grid


    # plt.title(pre_title + title)
    
    # Add legend to differentiate between models
    plt.legend(fontsize=18)

    # Show the plot
    plt.grid(True)
    plt.show()


def plot_metrics_comparison_with_band (rl_copp_obj, metric, title = "Comparison"):
    """
    Plot the same metric of multiple models for comparing them (with mean curve and variability). 
    X axis will be the number of steps.

    Args:
    - rl_copp_object (RLCoppeliaManager): Instance of RLCoppeliaManager class just for managing the args and the base path.
    - title (str): The title of the chart.
    """
    # Define some control variables
    smooth_flag = False  # Set to True to apply smoothing
    smooth_level = 40  # Define the window size for smoothing
    band_flag = False  # Set to True to plot the variability band


    # Get the training csv path for later getting the action times from there
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path, "train_records.csv")  # Name of the train records csv to search the algorithm used
    timestep_to_data = {}

    

    for model_index in range(len(rl_copp_obj.args.model_ids)):

        # CSV File path to get data from
        file_pattern = f"{rl_copp_obj.args.robot_name}_model_{rl_copp_obj.args.model_ids[model_index]}_*.csv"
        files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "training_metrics", file_pattern))
        logging.info(f"Files detected {files}, model index {model_index}, model ID {rl_copp_obj.args.model_ids[model_index]}")
        
        # Read the CSV file
        try:
            df = pd.read_csv(files[0])
        except Exception as e:
            logging.error(f"File not found for model {model_name}. Skipping this model. Error: {e}")   
            del rl_copp_obj.args.model_ids[model_index]
            continue

        # print(f"Processing model {model_index} with ID {rl_copp_obj.args.model_ids[model_index]}")
        # Get timestep of the selected model
        model_name = rl_copp_obj.args.robot_name + "_model_" + str(rl_copp_obj.args.model_ids[model_index])
        timestep = utils.get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)")


        # Limit max steps <= 200000
        mask = df['Step'] <= 200000
        df = df[mask]
        
        # Extract steps and rewards
        steps = df['Step'].values
        if metric == "rewards":
            data = df['rollout/ep_rew_mean'].values
        elif metric == "episodes_length":
            data = df['rollout/ep_len_mean'].values

        # Smooth the data
        if smooth_flag:
            window_size = smooth_level
            smoothed_data = moving_average(data, window_size=window_size)
            smoothed_steps = steps[:len(smoothed_data)]

        else:
            smoothed_data = data
            smoothed_steps = steps[:len(smoothed_data)]

        # Group data by timestep
        if timestep not in timestep_to_data:
            timestep_to_data[timestep] = []
        timestep_to_data[timestep].append((smoothed_steps, smoothed_data))
    
    # Create a color map for the models
    color_map = plt.cm.get_cmap("tab20", len(timestep_to_data))

    # Plot the mean curve and variability band for each timestep group
    plt.figure(figsize=(13, 10))
    for model_index, (timestep, data_list) in enumerate(timestep_to_data.items()):
        # Align data by step and calculate mean and std
        all_steps = [d[0] for d in data_list]
        all_data = [d[1] for d in data_list]

        # Define a common set of steps (e.g., 1000 evenly spaced points)
        common_steps = np.linspace(min([steps[0] for steps in all_steps]), 
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
        color = color_map(model_index)  # Assign a unique color for each model
        plt.plot(common_steps, mean_data, label=f"Model {timestep}s", color=color, linewidth=2)


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

    # Add labels and title
    
    plt.xlabel('Steps', fontsize=20, labelpad=12)
    plt.ylabel(metric.capitalize(), fontsize=20, labelpad=12)
    # Adjust x axis ticks to show every 50,000 steps
    xticks = np.arange(0, steps.max() + 2000, 50000) 
    plt.xticks(xticks, fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(fontsize=18, ncol = 2)
    plt.grid(True)
    plt.tight_layout()
    if rl_copp_obj.args.save_plots:
        filename = f"metrics_comparison_{metric}.png"
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_convergence_comparison (rl_copp_obj, title = "Convergence Comparison "):
    """
    Plot the convergence point of multiple models for comparing them.

    Args:
    - rl_copp_object (RLCoppeliaManager): Instance of RLCoppeliaManager class just for managing the args and the base path.
    - title (str): The title of the chart.
    """

    # Define the x-axis options
    x_axis = ["WallTime", "Steps", "SimTime", "Episodes"]
    # Define y axis label
    y_label = "Rewards "
    
    # Get the training csv path for later getting the action times from there
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path,"train_records.csv")    # Name of the train records csv to search the algorithm used
    timestep = []

    # Generate a color map for the models
    color_map = plt.cm.get_cmap("tab10", len(rl_copp_obj.args.model_ids))

    
    for conv_type in x_axis:
        plt.figure(figsize=(10, 6))

        max_convergence_point = 0  # Track the maximum convergence point for this category

        for model_index in range(len(rl_copp_obj.args.model_ids)):
            # Get timestep of the selected model
            model_name = rl_copp_obj.args.robot_name + "_model_" + str(rl_copp_obj.args.model_ids[model_index])
            timestep.append(utils.get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)"))
            
            # CSV File path to get data from
            file_pattern = f"{rl_copp_obj.args.robot_name}_model_{rl_copp_obj.args.model_ids[model_index]}_*.csv"
            files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "training_metrics", file_pattern))

            # Read the CSV file and obtain the convergence point
            convergence_point, reward_fit, x_axis_values, reward, reward_at_convergence = utils.get_convergence_point (files[0], conv_type, convergence_threshold=0.02)

            # Update the maximum convergence point
            max_convergence_point = max(max_convergence_point, convergence_point)

            # Assign a unique color for each model
            color = color_map(model_index)
            
            # Plot the rewards for each model
            plt.plot(x_axis_values,  reward, color = color)
            
            # Get line label depending on the current conv_type
            if conv_type == "WallTime":
                line_label = f"Convergence Wall Time: {convergence_point:.2f}h"
            elif conv_type == "Steps":
                line_label = f"Convergence Steps: {convergence_point:.2f}"
            elif conv_type == "SimTime":
                line_label = f"Convergence Sim Time: {convergence_point:.2f}h"
            elif conv_type == "Episodes":
                line_label = f"Convergence Episodes: {convergence_point:.2f}"
            plt.axvline(
                convergence_point, 
                color=color, 
                linestyle=':', 
                label=f'Model {timestep[model_index]}s - {line_label}', 
                linewidth=4
                )
            

        # Add labels and title
        if conv_type == "WallTime":
            plt.xlabel('Wall time (hours)', fontsize=26, labelpad=12)
            title_def = title + '(Wall Time)'
        elif conv_type == "Steps":
            plt.xlabel('Steps', fontsize=26, labelpad=12)
            title_def = title + '(Steps)'
        elif conv_type == "SimTime":
            plt.xlabel('Simulation time (hours)', fontsize=26, labelpad=12)
            title_def = title + '(Sim Time)'
        elif conv_type == "Episodes":
            plt.xlabel('Episodes', fontsize=26, labelpad=12)
            title_def = title + '(Episodes)'

        # Set the y-axis label
        plt.ylabel(y_label, fontsize=26, labelpad=10)

        # Adjust the x-axis limit to improve visualization
        plt.xlim(0, max_convergence_point * 1.2)  # Extend the x-axis by 20% beyond the max convergence point

        plt.tick_params(axis='both', which='major', labelsize=22)

        # Set the title
        # plt.title(title_def)
        
        # Add legend to differentiate between models
        # plt.legend(
        #     loc='upper left', 
        #     bbox_to_anchor=(0.33, 1.32),  # Ajustar la posición de la leyenda
        #     fontsize=14
        # )
        plt.legend(
            loc='lower right',  # Posicionar la leyenda dentro del gráfico, abajo a la derecha
            fontsize=18
        )

        # Show the plot
        plt.grid()
        plt.tight_layout()
        if rl_copp_obj.args.save_plots:
            filename = f"convergence_{conv_type}.png"
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()


def plot_histogram (rl_copp_obj, model_index, mode, n_bins = 21, title = "Histogram for "):
    """
    Plots a histogram for showing the value expressed in the 'mode' variable.

    Args:
    - rl_copp_obj (RLCoppeliaManager): Instance of RLCoppeliaManager class just for managing the args and the base path.
    - model_index (int): Index of the model to analyze from the args list.
    - mode (str): Type of data to plot - "speeds" or "target_zones".
    - n_bins (int): Number of bins for the histogram.
    - title (str): The title prefix for the chart.
    """
    

    # Get the training csv path for later getting the action times from there
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path,"train_records.csv")    # Name of the train records csv to search the algorithm used

    hist_data = []

    if mode == "speeds":
        # CSV File path to get data from
        # Capture the desired files through a pattern
        file_pattern = f"{rl_copp_obj.args.robot_name}_model_{rl_copp_obj.args.model_ids[model_index]}_*_otherdata_*.csv"
        subfolder_pattern = f"{rl_copp_obj.args.robot_name}_model_{rl_copp_obj.args.model_ids[model_index]}_*_testing"
        files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "testing_metrics", subfolder_pattern, file_pattern))
        # Read CSV
        df = pd.read_csv(files[0])
        data_keys = ['Angular speed', 'Linear speed']
        data_keys_units = ["rad/s", "m/s"]
        bin_min = [-0.5, 0.1]
        bin_max = [0.5, 0.5]
    else:
        logging.error(f"Specified graphs mode doesn't exist: {mode}")

    for key in data_keys:
        hist_data.append(df[key])

    for i in range(len(data_keys)):
        logging.info(f"{data_keys[i]} stats:")
        logging.info(f"Mean: {hist_data[i].mean():.4f}")
        logging.info(f"Median: {hist_data[i].median():.4f}")
        logging.info(f"Standard deviation: {hist_data[i].std():.4f}")
        logging.info(f"Min: {hist_data[i].min():.4f}")
        logging.info(f"Max: {hist_data[i].max():.4f}")

        model_name = rl_copp_obj.args.robot_name + "_model_" + str(rl_copp_obj.args.model_ids[model_index])
        timestep = (utils.get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)"))

        # Configure the histogram
        plt.figure(figsize=(10, 6))

        # Create bin equally spaced between the specified limits
        bins = np.linspace(bin_min[i], bin_max[i], n_bins)  # 21 bins for having 20 intervals

        # Create the histogram
        plt.hist(hist_data[i], bins=bins, color='skyblue', edgecolor='black', alpha=0.7)

        # Add title and labels
        plt.title(title + data_keys[i] + ": Model " + str(timestep) + "s", fontsize=14)
        plt.xlabel(f"{data_keys[i]} ({data_keys_units[i]})", fontsize=12)
        plt.ylabel('Frequence', fontsize=12)
        plt.grid(axis='y', alpha=0.75)
        plt.legend()

        # Adjust x axis limits to ensure that all the range is visible
        plt.xlim(bin_min[i]-0.05, bin_max[i]+0.05)

        # Show the histogram
        plt.tight_layout()
        plt.show()


def plot_bars(rl_copp_obj, model_index, mode, title="Target Zone Distribution: "):
    """
    Creates a histogram for the target zones with discrete values (1, 2, 3) on the X-axis.
    
    Args:
        rl_copp_obj: Object containing the path and argument information.
        model_index: Index of the model to analyze.
        title: Title for the chart.
    """
    # Get CSV path
    file_pattern = f"{rl_copp_obj.args.robot_name}_model_{rl_copp_obj.args.model_ids[model_index]}_*_test_*.csv"
    subfolder_pattern = f"{rl_copp_obj.args.robot_name}_model_{rl_copp_obj.args.model_ids[model_index]}_*_testing"
    files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "testing_metrics", subfolder_pattern, file_pattern))
    
    # Read CSV file
    df = pd.read_csv(files[0])
    
    # Verify that the DataFrame has the expected columns
    if mode == "target_zones":
        data_keys = ['Target zone']
        
        possible_values = [1, 2, 3]
        labels = ['Target zone 1', 'Target zone 2', 'Target zone 3']

    for key in data_keys:
        data = []
        data = (df[key])
    
        # Count all the samples
        counts = data.value_counts().reindex(possible_values, fill_value=0)
        total_episodes = len(data)

        # Calculate percentages
        percentages = (counts / total_episodes) * 100
        
        # Stats
        logging.info(f"{key} stats:")
        logging.info(f"Total episodes: {total_episodes}")
        for j in possible_values:
            count = counts.get(j, 0)
            percentage = percentages.get(j, 0)
            logging.info(f"Zone {j}: {count} episodes ({percentage:.2f}%)")
        
        # Get timestep value of the selected model
        train_records_csv_name = os.path.join(rl_copp_obj.paths["training_metrics"], "train_records.csv")
        model_name = f"{rl_copp_obj.args.robot_name}_model_{rl_copp_obj.args.model_ids[model_index]}"
        timestep = utils.get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)")
        
        # Create the figure
        plt.figure(figsize=(10, 6))
        
        # Configure bars for discrete values
        x = np.array(possible_values)
        
        
        # Create bars graph
        bars = plt.bar(labels, counts, color=['skyblue', 'lightgreen', 'salmon'], 
                    edgecolor='black', alpha=0.7)
        
        # Add labels
        for bar, count, percentage in zip(bars, counts, percentages):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
        
        # Configure axis and title
        plt.title(f"{title}Model {timestep}s", fontsize=14)
        plt.xlabel('Target Zone', fontsize=12)
        plt.ylabel('Frequence (number of episodes)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Adjust Y axis limits
        max_count = counts.max()
        plt.ylim(0, max_count * 1.15)  # 15% aditional space
        
        # Show the plot
        plt.tight_layout()
        if rl_copp_obj.args.save_plots:
            filename = f"bars_{key}.png"
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()


def plot_grouped_bar_chart(rl_copp_obj, mode, num_intervals=10, title=" Distribution by Intervals"):
    """
    Creates a grouped bar chart for representing the frequency of an especific variable of a model.
    
    Args:
    - rl_copp_obj: RLCoppeliaManager instance
    - mode (str): "speeds" or "target_zones"
    - num_intervals (int): Number of intervals to divide the represented range.
    - title (str): Title for the plot
    """

    # Initialize some variables depending on the selected mode
    if mode == "speeds":
        data_keys = ['Angular speed', 'Linear speed']
        data_keys_units = ["rad/s", "m/s"]
        min_value = [-0.5, 0.1]
        max_value = [0.5, 0.5]

    elif mode == "target_zones":
        data_keys = ['Target zone']
        data_keys_units = [""]
        categories = [1, 2, 3]

    # Get the training csv path for later getting the action times from there
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path,"train_records.csv")    # Name of the train records csv to search the algorithm used
    timestep = []

    # Plot a graph per key data
    for id_data in range(len(data_keys)):

        if mode== "speeds": # continuos intervals case

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
        frequencies = np.zeros((len(rl_copp_obj.args.model_ids), num_groups))

        # Load data and calculate frequencies for each model and interval
        for i, model_idx in enumerate(rl_copp_obj.args.model_ids):

            # Construct the file pattern for each case
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
            
            # Save data from the desired column
            data = []
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
         
        # Create the grouped bar chart
        plt.figure(figsize=(15, 8))
        
        # Set width of bars and positions
        bar_width = 0.8 / len(rl_copp_obj.args.model_ids)
        r = np.arange(num_groups)
        
        # Plot bars for each model
        for i in range(len(timestep)):
            position = r + i * bar_width - (len(rl_copp_obj.args.model_ids) - 1) * bar_width / 2
            plt.bar(position, frequencies[i, :], width=bar_width, 
                    label=f"Model {timestep[i]}s")
        
        # Add labels, title and legend
        plt.xlabel(f"{data_keys[id_data]} Intervals {data_keys_units[id_data]}", fontsize=20, labelpad=15)
        plt.ylabel('Percentage of Samples (%)', fontsize=20, labelpad=15)
        # plt.title(data_keys[id_data] + title, fontsize=16, pad=20)
        plt.xticks(r, interval_labels, rotation=30 if mode == "speeds" else 0, ha='right', fontsize=20)  # rotate labels for speed (many intervals)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        # plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1.05, 1)) 
        
        # Add grid for readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        if rl_copp_obj.args.save_plots:
            filename = f"grouped_bar_chart_{id_data}.png"
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

        # Reset timestep array
        timestep = []
        

def plot_scene_trajs(rl_copp_obj, folder_path):
    """
    Plots the scene and the trajectories followed by the robot during testing.

    This function visualizes the scene elements (robot, obstacles, targets) and overlays 
    the trajectories followed by the robot. It also highlights the final position of the 
    robot and indicates whether it reached a target or not.

    Args:
        rl_copp_obj (RLCoppeliaManager): Instance of the RLCoppeliaManager class for managing 
            paths and arguments.
        folder_path (str): Path to the folder containing the scene and trajectory CSV files.

    Raises:
        ValueError: If the scene file is not found or if there is more than one scene file 
            in the folder.

    Returns:
        None
    """
    # Search scene files
    scene_files = glob.glob(os.path.join(folder_path, "scene_*.csv"))
    if len(scene_files) != 1:
        raise ValueError(f"Expected one scene CSV, found {len(scene_files)}")
    
    scene_path = scene_files[0]
    df_scene = pd.read_csv(scene_path)

    fig, ax = plt.subplots(figsize=(9.4, 6))
    ax.set_xlabel("x (m)", fontsize=18)
    ax.set_ylabel("y (m)", fontsize=18)
    ax.set_xlim(2.5, -2.5)
    ax.set_ylim(2.5, -2.5)
    ax.set_aspect('equal')

    # Configure ticks size
    ax.tick_params(axis='both', which='major', labelsize=18)  

    # Set title
    # ax.set_title("CoppeliaSim Scene Representation", fontsize=16, pad=20)

    # Draw 0.5 m grid
    for i in np.arange(-2.5, 3, 0.5):
        ax.axhline(i, color='lightgray', linewidth=0.5, zorder=0)
        ax.axvline(i, color='lightgray', linewidth=0.5, zorder=0)

    # Draw all the elements of the scene
    for _, row in df_scene.iterrows():
        x, y = row['x'], row['y']
        if row['type'] == 'robot':
            circle = plt.Circle((x, y), 0.35 / 2, color='black', label='Robot', zorder=4)
            ax.add_patch(circle)

            # Indicate orientation using a triangle
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
            circle = plt.Circle((x, y), 0.25 / 2, color='gray', label='Obstacle')
            ax.add_patch(circle)

        elif row['type'] == 'target':
            # Plot the target rings
            target_rings = [(0.5 / 2, 'blue'), (0.25 / 2, 'red'), (0.03 / 2, 'yellow')]
            for radius, color in target_rings:
                circle = plt.Circle((x, y), radius, color=color, fill=True, alpha=0.6)
                ax.add_patch(circle)
    
    traj_files = glob.glob(os.path.join(folder_path, "trajectory_*.csv"))

    # Group trajectories by model ID
    model_trajs = defaultdict(list)
    for file in traj_files:
        parts = file.split('_')
        model_id = parts[-1].split('.')[0]
        model_trajs[model_id].append(os.path.join(folder_path, file))

    colors = plt.cm.get_cmap("tab10")
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path, "train_records.csv")

    model_plot_data = []

    # Interpolate and store data for later ordered plotting
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

    # for data in model_plot_data:
    #     ax.plot(data["x"], data["y"], color=data["color"],
    #             label=data["label"], linewidth=2, zorder=3)


    # Get all target positions
    target_rows = df_scene[df_scene['type'] == 'target']
    targets = [(row['x'], row['y']) for _, row in target_rows.iterrows()]

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
            # closest_target = targets[np.argmin(distances)]
            
            logging.info(f"Distance to closest target: {min_distance:.2f} m")

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

    # Removed duplicated labels
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper right', bbox_to_anchor=(1.5, 1.03), fontsize = 16)

    plt.grid(True)
    plt.show()



def compare_models_boxplots_old2(rl_copp_obj, model_ids):
    """
    Compare a variable betweeen multiple models using boxplots.
    
    Args:
        - rl_copp_obj (RLCoppeliaManager): Instance of RLCoppeliaManager class just for managing the args and the base path.
        - model_ids (list): List of model IDs to compare.
    """
    # Initialize variables
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

    # Get trainings' csv name for searching action times later
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path,"train_records.csv") 

    for model_id in model_ids:
        file_pattern = f"{rl_copp_obj.args.robot_name}_model_{model_id}_*_test_*.csv"
        subfolder_pattern = f"{rl_copp_obj.args.robot_name}_model_{model_id}_*_testing"
        files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "testing_metrics", subfolder_pattern, file_pattern))
        if not files:
            logging.error(f"[!] File not found for model {model_id}")
            continue
            
        # Get action time for each model
        model_name = rl_copp_obj.args.robot_name + "_model_" + str(model_id)
        timestep = float(utils.get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)"))
        model_action_times.append(f"{timestep}s")
        timestep_values.append(timestep)

        for file in files:
            df = pd.read_csv(file)
            df["Model"] = str(timestep) + "s"
            df["Timestep"] = timestep
            combined_data.append(df)

        # Load other data (Linear speed and Angular speed)
        file_pattern = f"{rl_copp_obj.args.robot_name}_model_{model_id}_*_otherdata_*.csv"
        files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "testing_metrics", subfolder_pattern, file_pattern))
        if not files:
            logging.error(f"[!] Other data file not found for model {model_id}")
            continue

        for file in files:
            df = pd.read_csv(file)
            df["Model"] = str(timestep) + "s"
            print(f"Adding data for model {model_id} with action time {timestep}s")
            df["Timestep"] = timestep
            # Add Linear speed and Angular speed to the combined data
            df["Angular speed"] = df["Angular speed"].abs()  # Use absolute value for Angular speed
            combined_data.append(df)


    if not combined_data:
        logging.error("[!] Data not found.")
        return

    full_df = pd.concat(combined_data, ignore_index=True)

    for metric in metrics:
        if metric not in full_df.columns:
            logging.error(f"[!] Column '{metric}' no found")
            continue

    # Add a metric that doesn't appear on the csv file: reward detail, for plotting those cases with a positive reward, so user can observe reward more detailed
    # metrics.append("Reward detail (>=0)")
    print(df["Timestep"])

    for metric in metrics:
    
        plt.figure(figsize=(10, 6))
        if metric == "Reward":
            # Use numeric X-axis for proper spline alignment
            y_label_name = metric
            current_ax = sns.boxplot(data=full_df, x="Model", y=metric)
            
            # Get the unique models and their positions
            unique_models = sorted(full_df["Model"].unique(), key=lambda x: float(x.replace('s', '')))
            x_positions = range(len(unique_models))

            # Calculate the median for each model in the right order
            medians = []
            timesteps_ordered = []
            for model in unique_models:
                median_val = full_df[full_df["Model"] == model][metric].median()
                medians.append(median_val)
                timesteps_ordered.append(float(model.replace('s', '')))
            
            # Create the spline using the x positions and medians
            # if len(x_positions) > 2:  # We need at least 3 points for a spline
            #     x_smooth = np.linspace(0, len(x_positions)-1, 300)
            #     spline = make_interp_spline(x_positions, medians, k=min(2, len(x_positions)-1))
            #     y_smooth = spline(x_smooth)
            #     plt.plot(x_smooth, y_smooth, color='red', linestyle='--', linewidth=2, label='Spline')
            
            # # Mark the median points
            # plt.scatter(x_positions, medians, color='black', marker='x', s=60, zorder=10)
            
            # # Find the optimal timestep
            # if len(x_positions) > 2:
            #     optimal_idx = np.argmax(y_smooth)
            #     optimal_x_pos = x_smooth[optimal_idx]
            #     optimal_y = y_smooth[optimal_idx]
                
            #     # Interpolate the optimal timestep
            #     optimal_timestep = np.interp(optimal_x_pos, x_positions, timesteps_ordered)
                
            #     # Plot the optimal timestep
            #     plt.plot(optimal_x_pos, optimal_y, marker='o', color='green', markersize=12, 
            #             label=f'Optimal: {optimal_timestep:.3f}s', zorder=15)
            
            plt.legend()

        elif metric in ["Time (s)", "Linear speed", "Angular speed", "Distance traveled (m)"]:
            current_ax = sns.boxplot(data=full_df, x="Model", y=metric)
            if metric == "Linear speed":
                y_label_name = metric + " (m/s)"
            elif metric == "Angular speed":
                y_label_name = metric + " (rad/s)"
            elif metric == "Time (s)":
                y_label_name = "Average episode duration (s)"

            
        elif metric == "Reward detail (>=0)": 
            df_reward_detail = full_df[full_df["Reward"] >= 0]
            current_ax = sns.boxplot(data=df_reward_detail, x="Model", y="Reward")


        elif metric == "Target zone":
            # Filtrate episodes with target zone == Nan and also removes those in which the robot starts directly inside the target
            df_target = full_df[
                full_df[metric].notna() &
                (full_df[metric] != 0)
            ]
            df_target = df_target[((df_target["TimeSteps count"]) > 1)]
            
            
            # Assure that the models maintein the order
            model_order = [str(mid) for mid in model_action_times]

            zone_counts = df_target.groupby(["Model", "Target zone"]).size().reset_index(name='count')
            print(zone_counts)

            # Get total counts for each model
            
            # totals = df_target.groupby("Model").size().reset_index(name='total')
            totals = full_df[full_df[metric].notna()].groupby("Model").size().reset_index(name='total')

            # df_target.to_csv("df_target.csv", index=False)

            # Merge counts with totals
            zone_percents = pd.merge(zone_counts, totals, on="Model")
            zone_percents["Zone percentage (%)"] = 100 * zone_percents["count"] / zone_percents["total"]


            tab20 = plt.cm.get_cmap('tab20', 20)      # 20 colores distintos
            set3 = plt.cm.get_cmap('Set3', 12)        # 12 colores (pasteles)

            colors = [tab20(i) for i in range(20)] + [set3(i) for i in range(10)]

            current_ax = sns.barplot(
                data=zone_percents, 
                x="Target zone", 
                y="Zone percentage (%)", 
                hue="Model", 
                hue_order=model_order,
                palette=colors[:len(model_order)]                
                )
            plt.legend(title="Model", ncol = 2, fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))

        elif metric == "Crashes":
            # Ensure that the values of collisions are booleans
            y_label_name = "Collision probability (%)"
            full_df[metric] = full_df[metric].astype(str).str.strip().str.lower().map({
                "true": True, "1": True, "yes": True,
                "false": False, "0": False, "no": False
            })

            # Calculate collsiion percentage
            crash_pct = (
                full_df.groupby("Model")[metric]
                .mean()
                .mul(100)
                .rename("Collision Rate")
                .reset_index()
            )

            current_ax = sns.barplot(data=crash_pct, x="Model", y="Collision Rate")
            plt.ylabel("Episodes with collision (%)")
        
        
        if metric == "Target zone":
            x_label_name = "Target zone"
            y_label_name = "Probability (%)"
        else:
            x_label_name = "Timestep (s)"
            
        current_ax.set_xlabel(x_label_name, fontsize=20, labelpad=10)  
        current_ax.set_ylabel(y_label_name, fontsize=20, labelpad=10)  
        plt.tick_params(axis='both', which='major', labelsize=20)  

        plt.grid(True)
        plt.tight_layout()
        plt.show()



def compare_models_boxplots_old(rl_copp_obj, model_ids):
    """
    Compare a variable betweeen multiple models using boxplots.
    
    Args:
        - rl_copp_obj (RLCoppeliaManager): Instance of RLCoppeliaManager class just for managing the args and the base path.
        - model_ids (list): List of model IDs to compare.
    """
    # Initialize variables
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
    x_label_name = "Timestep (s)"

    # Get trainings' csv name for searching action times later
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path,"train_records.csv") 

    for model_id in model_ids:
        file_pattern = f"{rl_copp_obj.args.robot_name}_model_{model_id}_*_test_*.csv"
        subfolder_pattern = f"{rl_copp_obj.args.robot_name}_model_{model_id}_*_testing"
        files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "testing_metrics", subfolder_pattern, file_pattern))
        if not files:
            logging.error(f"[!] File not found for model {model_id}")
            continue
            
        # Get action time for each model
        model_name = rl_copp_obj.args.robot_name + "_model_" + str(model_id)
        timestep = float(utils.get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)"))
        model_action_times.append(f"{timestep}s")
        timestep_values.append(float(timestep))

        for file in files:
            logging.info(f"File: {file}")
            df = pd.read_csv(file)
            df["Model"] = str(timestep) + "s"
            df["Timestep"] = timestep
            combined_data.append(df)

        # Load other data (Linear speed and Angular speed)
        file_pattern = f"{rl_copp_obj.args.robot_name}_model_{model_id}_*_otherdata_*.csv"
        files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "testing_metrics", subfolder_pattern, file_pattern))
        if not files:
            logging.error(f"[!] Other data file not found for model {model_id}")
            continue

        for file in files:
            df = pd.read_csv(file)
            df["Model"] = str(timestep) + "s"
            logging.info(f"Adding data for model {model_id} with action time {timestep}s")
            df["Timestep"] = float(timestep)
            # Add Linear speed and Angular speed to the combined data
            df["Angular speed"] = df["Angular speed"].abs()  # Use absolute value for Angular speed
            combined_data.append(df)


    if not combined_data:
        logging.error("[!] Data not found.")
        return

    full_df = pd.concat(combined_data, ignore_index=True)

    for metric in metrics:
        if metric not in full_df.columns:
            logging.error(f"[!] Column '{metric}' no found")
            continue

    # Add a metric that doesn't appear on the csv file: reward detail, for plotting those cases with a positive reward, so user can observe reward more detailed
    # metrics.append("Reward detail (>=0)")

    for metric in metrics:
    
        plt.figure(figsize=(10, 6))
        if metric == "Reward":
            full_df["Timestep"] = full_df["Timestep"].astype(float)
            timesteps_ordered = sorted(full_df["Timestep"].unique())

            # Crear el boxplot con eje X numérico
            ax = sns.boxplot(data=full_df, x="Timestep", y=metric, order=timesteps_ordered)

            # Calcular medianas y medias para cada timestep
            medians = []
            means = []
            for ts in timesteps_ordered:
                model_data = full_df[full_df["Timestep"] == ts][metric].dropna()
                median_val = model_data.median()
                mean_val = model_data.mean()
                medians.append(median_val)
                means.append(mean_val)
                logging.info(f"Timestep {ts:.3f}s: Mean Reward = {mean_val:.2f}, Median Reward = {median_val:.2f}")

            # # Dibujar spline si hay suficientes puntos
            # if len(timesteps_ordered) > 2:
            #     x_smooth = np.linspace(min(timesteps_ordered), max(timesteps_ordered), 300)
            #     spline = make_interp_spline(timesteps_ordered, medians, k=min(2, len(timesteps_ordered)-1))
            #     y_smooth = spline(x_smooth)
            #     plt.plot(x_smooth, y_smooth, color='red', linestyle='--', linewidth=2, label='Spline')

            #     # Encontrar timestep óptimo
            #     optimal_idx = np.argmax(y_smooth)
            #     optimal_x = x_smooth[optimal_idx]
            #     optimal_y = y_smooth[optimal_idx]
            #     plt.plot(optimal_x, optimal_y, marker='o', color='green', markersize=12, 
            #             label=f'Optimal: {optimal_x:.3f}s', zorder=15)

            # # Marcar medianas
            # plt.scatter(timesteps_ordered, medians, color='black', marker='x', s=60, zorder=10)


        elif metric in ["Time (s)", "Linear speed", "Angular speed", "Distance traveled (m)"]:
            full_df["Timestep"] = full_df["Timestep"].astype(float)
            timesteps_ordered = sorted(full_df["Timestep"].unique())
            ax = sns.boxplot(data=full_df, x="Model", y=metric)
            if metric == "Linear speed":
                metric = metric + " (m/s)"
            elif metric == "Angular speed":
                metric = metric + " (rad/s)"
            elif metric == "Time (s)":
                metric = "Average episode duration (s)"

            
        elif metric == "Reward detail (>=0)": 
            df_reward_detail = full_df[full_df["Reward"] >= 0]
            ax = sns.boxplot(data=df_reward_detail, x="Model", y="Reward")

        elif metric == "Target zone":
            # Filtrate episodes with target zone == Nan
            df_target = full_df[full_df[metric].notna() & (full_df[metric] != 0)]
            
            # Assure that the models maintein the order
            model_order = [str(mid) for mid in model_action_times]

            zone_counts = df_target.groupby(["Model", "Target zone"]).size().reset_index(name='count')

            # Get total counts for each model
            
            totals = full_df[full_df[metric].notna()].groupby("Model").size().reset_index(name='total')

            # df_target.to_csv("df_target.csv", index=False)

            # Merge counts with totals
            zone_percents = pd.merge(zone_counts, totals, on="Model")
            zone_percents["Zone percentage (%)"] = 100 * zone_percents["count"] / zone_percents["total"]

            tab20 = plt.cm.get_cmap('tab20', 20)
            set3 = plt.cm.get_cmap('Set3', 12)
            colors = [tab20(i) for i in range(20)] + [set3(i) for i in range(4)]

            sns.barplot(
                data=zone_percents, 
                x="Target zone", 
                y="Zone percentage (%)", 
                hue="Model", 
                hue_order=model_order,
                palette=colors[:len(model_order)]                
                )
            plt.legend(title="Model", ncol = 2, fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
            

        elif metric == "Crashes":
            # Ensure that the values of collisions are booleans
            full_df[metric] = full_df[metric].astype(str).str.strip().str.lower().map({
                "true": True, "1": True, "yes": True,
                "false": False, "0": False, "no": False
            })

            # Calculate collsiion percentage
            crash_pct = (
                full_df.groupby("Model")[metric]
                .mean()
                .mul(100)
                .rename("Collision Rate")
                .reset_index()
            )
            metric = "Collision Rate (%)"
            sns.barplot(data=crash_pct, x="Model", y="Collision Rate")
            plt.ylabel("Episodes with collision (%)")
        
        
        # plt.title(f"{metric} comparison", fontsize=16)
        
        # CORRECCIÓN: Solo aplicar la modificación de etiquetas para casos que NO sean "Reward"
        if metric != "Reward":
            # Obtener las etiquetas actuales de los ticks
            current_ticks = plt.xticks()[0]  # Obtiene las posiciones de los ticks
            current_labels = plt.xticks()[1]  # Obtiene las etiquetas de los ticks

            # Eliminar la 's' de las etiquetas
            new_labels = [label.get_text().replace('s', '') for label in current_labels]
            plt.xticks(current_ticks, new_labels, fontsize=16)
            current_ax = plt.gca()
        
        if metric == "Target zone":
            x_label_name = "Target zone"
            metric = "Probability (%)"
        else:
            x_label_name = "Timestep (s)"
            tick_pos = ax.get_xticks()
            tick_labels = [f"{float(t):.3f}" for t in ax.get_xticks()]
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, fontsize=16)
            
        
        
        plt.xlabel("Timestep (s)", fontsize=20, labelpad=10)
        plt.ylabel("Reward", fontsize=20, labelpad=10)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        if rl_copp_obj.args.save_plots:
            filename = f"boxplot_{metric}.png"
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
    

def compare_models_boxplots(rl_copp_obj, model_ids):
    """
    Compare a variable betweeen multiple models using boxplots.
    
    Args:
        - rl_copp_obj (RLCoppeliaManager): Instance of RLCoppeliaManager class just for managing the args and the base path.
        - model_ids (list): List of model IDs to compare.
    """
    # Initialize variables
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
    x_label_name = "Timestep (s)"

    # Get trainings' csv name for searching action times later
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path,"train_records.csv") 

    for model_id in model_ids:
        file_pattern = f"{rl_copp_obj.args.robot_name}_model_{model_id}_*_test_*.csv"
        subfolder_pattern = f"{rl_copp_obj.args.robot_name}_model_{model_id}_*_testing"
        files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "testing_metrics", subfolder_pattern, file_pattern))
        if not files:
            logging.error(f"[!] File not found for model {model_id}")
            continue
            
        # Get action time for each model
        model_name = rl_copp_obj.args.robot_name + "_model_" + str(model_id)
        timestep = float(utils.get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)"))
        model_action_times.append(f"{timestep}s")
        timestep_values.append(float(timestep))

        for file in files:
            logging.info(f"File: {file}")
            df = pd.read_csv(file)
            df["Model"] = str(timestep) + "s"
            df["Timestep"] = timestep
            combined_data.append(df)

        # Load other data (Linear speed and Angular speed)
        file_pattern = f"{rl_copp_obj.args.robot_name}_model_{model_id}_*_otherdata_*.csv"
        files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "testing_metrics", subfolder_pattern, file_pattern))
        if not files:
            logging.error(f"[!] Other data file not found for model {model_id}")
            continue

        for file in files:
            df = pd.read_csv(file)
            df["Model"] = str(timestep) + "s"
            logging.info(f"Adding data for model {model_id} with action time {timestep}s")
            df["Timestep"] = float(timestep)
            # Add Linear speed and Angular speed to the combined data
            df["Angular speed"] = df["Angular speed"].abs()  # Use absolute value for Angular speed
            combined_data.append(df)


    if not combined_data:
        logging.error("[!] Data not found.")
        return

    full_df = pd.concat(combined_data, ignore_index=True)

    for metric in metrics:
        if metric not in full_df.columns:
            logging.error(f"[!] Column '{metric}' no found")
            continue

    # Add a metric that doesn't appear on the csv file: reward detail, for plotting those cases with a positive reward, so user can observe reward more detailed
    # metrics.append("Reward detail (>=0)")

    for metric in metrics:
    
        plt.figure(figsize=(10, 6))
        if metric == "Reward":
            y_label_name = metric
            # Ordenar los datos por timestep para que el boxplot aparezca en orden correcto
            unique_models = sorted(full_df["Model"].unique(), key=lambda x: float(x.replace('s', '')))
            timesteps_ordered = [float(model.replace('s', '')) for model in unique_models]
            
            # Crear boxplots manuales con matplotlib para tener eje X numérico
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Preparar datos para boxplots manuales
            box_data = []
            medians = []
            means = []
            valid_timesteps = []
            
            for i, model in enumerate(unique_models):
                model_data = full_df[full_df["Model"] == model][metric].values
                
                # Debug: imprimir información sobre los datos
                logging.info(f"Processing model: {model}")
                logging.info(f"Model data shape: {model_data.shape}")
                logging.info(f"Model data sample: {model_data[:5] if len(model_data) > 0 else 'No data'}")
                logging.info(f"NaN count: {np.isnan(model_data).sum()}")
                logging.info(f"Inf count: {np.isinf(model_data).sum()}")
                
                # Filtrar valores NaN e infinitos ANTES de calcular estadísticas
                original_size = len(model_data)
                clean_data = model_data[np.isfinite(model_data)]
                logging.info(f"After filtering: {original_size} -> {len(clean_data)} values")
                
                if len(clean_data) > 0:  # Solo procesar si hay datos válidos
                    box_data.append(clean_data)
                    # Calcular estadísticas con datos limpios
                    median_val = np.median(clean_data)
                    mean_val = np.mean(clean_data)
                    medians.append(median_val)
                    means.append(mean_val)
                    valid_timesteps.append(timesteps_ordered[i])
                    
                    # Log the mean and median for the current model
                    logging.info(f"Model {model}: Mean Reward = {mean_val:.2f}, Median Reward = {median_val:.2f}")
                else:
                    logging.warning(f"Model {model}: No valid data found after filtering NaN/Inf, skipping")
            
            if len(box_data) == 0:
                logging.error("No valid data found for any model")
                continue
            
            # Crear boxplots con posiciones numéricas reales
            box_width = min(np.diff(valid_timesteps)) * 0.3 if len(valid_timesteps) > 1 else 0.05
            bp = ax.boxplot(box_data, positions=valid_timesteps, widths=box_width, patch_artist=True)
            
            # Personalizar los boxplots para que se vean como seaborn
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            # Create the spline using the timesteps and medians (solo si hay datos válidos)
            # if len(valid_timesteps) > 2 and all(np.isfinite(medians)):
            #     try:
            #         x_smooth = np.linspace(min(valid_timesteps), max(valid_timesteps), 300)
            #         spline = make_interp_spline(valid_timesteps, medians, k=min(2, len(valid_timesteps)-1))
            #         y_smooth = spline(x_smooth)
            #         ax.plot(x_smooth, y_smooth, color='red', linestyle='--', linewidth=2, label='Spline')
                    
            #         # Find the optimal timestep
            #         optimal_idx = np.argmax(y_smooth)
            #         optimal_timestep = x_smooth[optimal_idx]
            #         optimal_reward = y_smooth[optimal_idx]
                    
            #         # Plot the optimal timestep
            #         ax.plot(optimal_timestep, optimal_reward, marker='o', color='green', markersize=12, 
            #                 label=f'Optimal: {optimal_timestep:.3f}s', zorder=15)
            #     except Exception as e:
            #         logging.warning(f"Could not create spline: {e}")
            
            # # Mark the median points (solo los válidos)
            # if len(valid_timesteps) > 0 and all(np.isfinite(medians)):
            #     ax.scatter(valid_timesteps, medians, color='black', marker='x', s=60, zorder=10)
            
            ax.legend()
            
            # Configurar el eje X como numérico
            if len(valid_timesteps) > 0:
                ax.set_xticks(valid_timesteps)
                ax.set_xticklabels([f"{t}" for t in valid_timesteps], fontsize=14, rotation=90)
            
            # Usar ax en lugar de plt para el resto de configuraciones
            current_ax = ax

        elif metric in ["Time (s)", "Linear speed", "Angular speed", "Distance traveled (m)"]:
            sns.boxplot(data=full_df, x="Model", y=metric)
            if metric == "Linear speed":
                y_label_name = metric + " (m/s)"
            elif metric == "Angular speed":
                y_label_name = metric + " (rad/s)"
            elif metric == "Time (s)":
                y_label_name = "Average episode duration (s)"

            
        elif metric == "Reward detail (>=0)": 
            y_label_name = metric
            df_reward_detail = full_df[full_df["Reward"] >= 0]
            sns.boxplot(data=df_reward_detail, x="Model", y="Reward")

        elif metric == "Target zone":
            y_label_name = metric
            # Filtrate episodes with target zone == Nan
            df_target = full_df[full_df[metric].notna() & (full_df[metric] != 0)]
            
            # Assure that the models maintein the order
            model_order = [str(mid) for mid in model_action_times]

            zone_counts = df_target.groupby(["Model", "Target zone"]).size().reset_index(name='count')

            # Get total counts for each model
            
            totals = full_df[full_df[metric].notna()].groupby("Model").size().reset_index(name='total')

            # df_target.to_csv("df_target.csv", index=False)

            # Merge counts with totals
            zone_percents = pd.merge(zone_counts, totals, on="Model")
            zone_percents["Zone percentage (%)"] = 100 * zone_percents["count"] / zone_percents["total"]

            tab20 = plt.cm.get_cmap('tab20', 20)
            set3 = plt.cm.get_cmap('Set3', 12)
            colors = [tab20(i) for i in range(20)] + [set3(i) for i in range(4)]

            sns.barplot(
                data=zone_percents, 
                x="Target zone", 
                y="Zone percentage (%)", 
                hue="Model", 
                hue_order=model_order,
                palette=colors[:len(model_order)]                
                )
            plt.legend(title="Model", ncol = 2, fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
            

        elif metric == "Crashes":
            # Ensure that the values of collisions are booleans
            full_df[metric] = full_df[metric].astype(str).str.strip().str.lower().map({
                "true": True, "1": True, "yes": True,
                "false": False, "0": False, "no": False
            })

            # Calculate collsiion percentage
            crash_pct = (
                full_df.groupby("Model")[metric]
                .mean()
                .mul(100)
                .rename("Collision Rate")
                .reset_index()
            )
            y_label_name = "Collision Rate (%)"
            sns.barplot(data=crash_pct, x="Model", y="Collision Rate")
            plt.ylabel("Episodes with collision (%)")
        
        
        # plt.title(f"{metric} comparison", fontsize=16)
        
        # CORRECCIÓN: Solo aplicar la modificación de etiquetas para casos que NO sean "Reward"
        if metric != "Reward":
            # Obtener las etiquetas actuales de los ticks
            current_ticks = plt.xticks()[0]  # Obtiene las posiciones de los ticks
            current_labels = plt.xticks()[1]  # Obtiene las etiquetas de los ticks

            # Eliminar la 's' de las etiquetas
            new_labels = [label.get_text().replace('s', '') for label in current_labels]
            plt.xticks(current_ticks, new_labels, fontsize=16)
            current_ax = plt.gca()
        
        if metric == "Target zone":
            x_label_name = "Target zone"
            y_label_name = "Probability (%)"
        else:
            x_label_name = "Timestep (s)"
            
        current_ax.set_xlabel(x_label_name, fontsize=20, labelpad=10)  
        current_ax.set_ylabel(y_label_name, fontsize=20, labelpad=10)  
        current_ax.tick_params(axis='both', which='major', labelsize=20)  

        current_ax.grid(True)
        plt.tight_layout()
        if rl_copp_obj.args.save_plots:
            filename = f"boxplot_{metric}.png"
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    
def plot_lat_curves(rl_copp_obj, model_index):
    # Get the training csv path for later getting the action times from there
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path,"train_records.csv")    # Name of the train records csv to search the algorithm used
    # Get action time 
    model_name = rl_copp_obj.args.robot_name + "_model_" + str(rl_copp_obj.args.model_ids[model_index])
    timestep = (utils.get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)"))

    # CSV File path to get data from
    # Capture the desired files through a pattern
    file_pattern = f"{rl_copp_obj.args.robot_name}_model_{rl_copp_obj.args.model_ids[model_index]}_*_otherdata_*.csv"
    subfolder_pattern = f"{rl_copp_obj.args.robot_name}_model_{rl_copp_obj.args.model_ids[model_index]}_*_testing"
    files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "testing_metrics", subfolder_pattern, file_pattern))

    # Read CSV
    df = pd.read_csv(files[0])

    # Filtrate rows where both values are both zero (beginning of an episode)
    # df_filtered = df[(df["LAT-Sim (s)"] > 0.01) | (df["LAT-Wall (s)"] > 0.01)].copy()

    # Restart the index so steps are consecutive
    # df_filtered.reset_index(drop=True, inplace=True)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["LAT-Sim (s)"], label="LAT-Agent (s)", linewidth=2)
    plt.plot(df.index, df["LAT-Wall (s)"], label="LAT-Wall (s)", linewidth=2)

    # Draw a horizontal line for the timestep value
    plt.axhline(float(timestep), color='red', linestyle='--', linewidth=2, label=f"Timestep = {timestep}s")

    plt.xlabel("Step", fontsize = 20, labelpad=12)
    plt.ylabel("LAT (s)", fontsize = 20, labelpad=12)
    plt.tick_params(axis='both', which='major', labelsize=20)
    # plt.title(f"LAT-Sim and LAT-Wall vs. Steps - Model {timestep}s")
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def interpolate_trajectory(x, y, num_points=100):
    """
    Interpolates a trajectory to generate a specified number of evenly spaced points.

    This function takes the x and y coordinates of a trajectory and interpolates them 
    to produce a new trajectory with a uniform distribution of points along its length.

    Args:
        x (array-like): X-coordinates of the original trajectory.
        y (array-like): Y-coordinates of the original trajectory.
        num_points (int): Number of points for the interpolated trajectory.

    Returns:
        tuple:
            - interp_x (array): Interpolated X-coordinates.
            - interp_y (array): Interpolated Y-coordinates.
    """
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
    total_length = cumulative_dist[-1]
    if total_length == 0:
        return np.full(num_points, x[0]), np.full(num_points, y[0])

    normalized_dist = cumulative_dist / total_length
    interp_x = interp1d(normalized_dist, x, kind='linear')
    interp_y = interp1d(normalized_dist, y, kind='linear')
    uniform_points = np.linspace(0, 1, num_points)
    return interp_x(uniform_points), interp_y(uniform_points)


def draw_uncertainty_ellipse(ax, mean_x, mean_y, cov, color, nsig=1.0, alpha=0.3, zorder=2):
    """Draws an uncertainty ellipse based on a 2x2 covariance matrix.

    The ellipse represents a confidence region for a 2D Gaussian distribution.

    Args:
        ax (matplotlib.axes.Axes): The axes object to draw the ellipse on.
        mean_x (float): X-coordinate of the ellipse center.
        mean_y (float): Y-coordinate of the ellipse center.
        cov (ndarray): 2x2 covariance matrix.
        color (str or tuple): Color of the ellipse.
        nsig (float, optional): Number of standard deviations for the ellipse size. Defaults to 1.0.
        alpha (float, optional): Transparency of the ellipse (0-1). Defaults to 0.3.
        zorder (int, optional): Drawing order (higher means drawn on top). Defaults to 2.

    Returns:
        None: The ellipse is added to the provided axes object.
    """
    # Compute eigenvalues and eigenvectors of covariance matrix
    vals, vecs = np.linalg.eigh(cov)
    
    # Sort eigenvalues and eigenvectors in descending order
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    
    # Calculate rotation angle (in degrees) from eigenvectors
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    
    # Calculate width and height of ellipse (2 * nsig * standard deviation)
    width, height = 2 * nsig * np.sqrt(vals)

    # Create ellipse patch
    ell = Ellipse(
        xy=(mean_x, mean_y),     # Ellipse center
        width=width,             # Major axis length
        height=height,           # Minor axis length
        angle=theta,             # Rotation angle in degrees
        color=color,             # Color
        alpha=alpha,             # Transparency
        zorder=zorder            # Drawing order
    )
    ax.add_patch(ell)


def test_normality_univariate(interpolated_xs, interpolated_ys):
    """
    Performs a Shapiro-Wilk test for normality on each interpolated point across x and y coordinates.

    This function evaluates whether the distribution of interpolated points at each position 
    along the trajectory follows a normal distribution.

    Args:
        interpolated_xs (ndarray): 2D array where each row represents a trajectory's interpolated 
            x-coordinates and each column corresponds to a specific point along the trajectory.
        interpolated_ys (ndarray): 2D array where each row represents a trajectory's interpolated 
            y-coordinates and each column corresponds to a specific point along the trajectory.

    Returns:
        tuple:
            - p_values_x (list): List of p-values from the Shapiro-Wilk test for the x-coordinates.
            - p_values_y (list): List of p-values from the Shapiro-Wilk test for the y-coordinates.
    """
    num_points = interpolated_xs.shape[1]
    p_values_x = []
    p_values_y = []
    for j in range(num_points):
        stat_x, p_x = shapiro(interpolated_xs[:, j])
        stat_y, p_y = shapiro(interpolated_ys[:, j])
        p_values_x.append(p_x)
        p_values_y.append(p_y)
    return p_values_x, p_values_y


def plot_kde_density(ax, xs, ys, cmap="Reds", levels=[0.5, 0.9]):
    """
    Plots KDE contours on the given axes.

    Parameters:
    - ax: Matplotlib axes object where the contours will be plotted.
    - xs: Array-like, x-coordinates of the data points.
    - ys: Array-like, y-coordinates of the data points.
    - cmap: Colormap for the contours.
    - levels: List of contour levels to display, representing probability densities.
    """
    # Stack the data for KDE
    xy = np.vstack([xs, ys])
    kde = gaussian_kde(xy)

    # Define grid over data range
    x_min, x_max = xs.min() - 0.1, xs.max() + 0.1
    y_min, y_max = ys.min() - 0.1, ys.max() + 0.1
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])

    # Evaluate KDE on grid
    density = kde(grid_coords).reshape(xx.shape)

    # Normalize density for contour levels
    density /= density.max()

    # Plot contours
    contour = ax.contour(xx, yy, density, levels=levels, cmap=cmap, alpha=0.7)
    return contour


def draw_robust_uncertainty_ellipse(ax, mean_x, mean_y, points, color='gray', alpha=0.3, zorder=1, nsig=2.0):
    """
    Draws a robust uncertainty ellipse based on a set of 2D points.

    This function computes a robust covariance matrix using the Minimum Covariance Determinant (MCD) 
    estimator and uses it to draw an uncertainty ellipse. If the robust estimation fails, it falls 
    back to the classical covariance matrix.

    Args:
        ax (matplotlib.axes.Axes): The axes object to draw the ellipse on.
        mean_x (float): X-coordinate of the ellipse center.
        mean_y (float): Y-coordinate of the ellipse center.
        points (ndarray): Array of shape (n_samples, 2) containing the 2D points.
        color (str or tuple): Color of the ellipse.
        alpha (float, optional): Transparency of the ellipse (0-1). Defaults to 0.3.
        zorder (int, optional): Drawing order (higher means drawn on top). Defaults to 1.
        nsig (float, optional): Number of standard deviations for the ellipse size. Defaults to 2.0.

    Returns:
        None: The ellipse is added to the provided axes object.
    """
    if len(points) < 2:
        return

    try:
        # Primer intento: robusto
        robust_cov = MinCovDet(support_fraction=0.9).fit(points)
        cov = robust_cov.covariance_
    except Exception as e:
        logging.warning(f"[MCD fallback] Using classical covariance due to error: {e}")
        try:
            cov = np.cov(points.T)
        except Exception as e2:
            logging.warning(f"Failed to compute classical covariance too: {e2}")
            return

    try:
        # Descomposición de la matriz de covarianza
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        # Parámetros de la elipse
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * nsig * np.sqrt(vals)

        ell = Ellipse(
            xy=(mean_x, mean_y),
            width=width,
            height=height,
            angle=theta,
            color=color,
            alpha=alpha,
            zorder=zorder
        )
        ax.add_patch(ell)
    except Exception as e:
        logging.warning(f"Failed to draw ellipse: {e}")


def plot_scene_trajs_with_variability(rl_copp_obj, folder_path, num_points=100, nsig=1.0):
    """
    Plot a scene with interpolated mean trajectories from multiple models,
    including uncertainty ellipses at each point based on covariance across trajectories.

    Args:
        rl_copp_obj: Main RL object providing access to config and paths.
        folder_path (str): Path to the folder containing scene and trajectory CSVs.
        num_points (int): Number of interpolation points per trajectory.
        nsig (float): Number of standard deviations for the uncertainty ellipses (e.g., 1.0 or 2.0).
    """
    files = os.listdir(folder_path)
    scene_file = [f for f in files if f.startswith("scene_") and f.endswith(".csv")][0]
    traj_files = [f for f in files if f.startswith("trajectory_") and f.endswith(".csv")]

    logging.debug(f"scene files: {scene_file}")
    logging.debug(f"traj_files: {traj_files}")
    scene_df = pd.read_csv(os.path.join(folder_path, scene_file))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(2.5, -2.5)
    ax.set_ylim(2.5, -2.5)
    ax.set_aspect('equal')
    ax.set_title(f"Scene with Interpolated Mean Trajectories ({nsig}σ uncertainty)")

    # Draw 0.5 m grid
    for i in np.arange(-2.5, 3, 0.5):
        ax.axhline(i, color='lightgray', linewidth=0.5, zorder=0)
        ax.axvline(i, color='lightgray', linewidth=0.5, zorder=0)

    # Draw static scene elements
    for _, row in scene_df.iterrows():
        x, y = row['x'], row['y']
        if row['type'] == 'robot':
            ax.add_patch(plt.Circle((x, y), 0.35 / 2, color='blue', label='Robot', zorder=2))
            
            # Dibujar orientación (si existe la columna 'theta')
            if 'theta' in row:
                theta = row['theta']

                # Triangle option:
                # Coordenadas del triángulo
                front_length = 0.15
                side_offset = 0.08

                # Punto frontal
                front = (x + front_length * np.cos(theta), y + front_length * np.sin(theta))
                # Puntos laterales
                left = (x + side_offset * np.cos(theta + 2.5), y + side_offset * np.sin(theta + 2.5))
                right = (x + side_offset * np.cos(theta - 2.5), y + side_offset * np.sin(theta - 2.5))

                triangle = plt.Polygon([front, left, right], color='white', zorder=3)
                ax.add_patch(triangle)
                
        elif row['type'] == 'obstacle':
            ax.add_patch(plt.Circle((x, y), 0.25 / 2, color='gray', label='Obstacle'))
        elif row['type'] == 'target':
            for r, c in [(0.25, 'blue'), (0.125, 'red'), (0.015, 'yellow')]:
                ax.add_patch(plt.Circle((x, y), r, color=c, alpha=0.6))

    # Group trajectories by model ID
    model_trajs = defaultdict(list)
    for file in traj_files:
        parts = file.split('_')
        model_id = parts[-1].split('.')[0]
        model_trajs[model_id].append(os.path.join(folder_path, file))

    colors = plt.cm.get_cmap("tab10")
    training_metrics_path = rl_copp_obj.paths["training_metrics"]
    train_records_csv_name = os.path.join(training_metrics_path, "train_records.csv")

    model_plot_data = []

    # Interpolate and store data for later ordered plotting
    for i, (model_id, paths) in enumerate(model_trajs.items()):
        interpolated_xs, interpolated_ys = [], []
        for path in paths:
            df = pd.read_csv(path)
            x_interp, y_interp = interpolate_trajectory(df['x'].values, df['y'].values, num_points)
            interpolated_xs.append(x_interp)
            interpolated_ys.append(y_interp)

        interpolated_xs = np.array(interpolated_xs)
        interpolated_ys = np.array(interpolated_ys)
        
        mean_x = np.mean(interpolated_xs, axis=0)
        mean_y = np.mean(interpolated_ys, axis=0)
        color = colors((i + 1) % 10)

        model_name = rl_copp_obj.args.robot_name + "_model_" + str(model_id)
        timestep = float(utils.get_data_from_training_csv(model_name, train_records_csv_name, column_header="Action time (s)"))

        model_plot_data.append({
            "timestep": timestep,
            "mean_x": mean_x,
            "mean_y": mean_y,
            "interpolated_xs": interpolated_xs,
            "interpolated_ys": interpolated_ys,
            "color": color,
            "label": f"Model {timestep}s"
        })

    # Sort models by timestep before plotting
    model_plot_data.sort(key=lambda d: d["timestep"])

    for data in model_plot_data:
        ax.plot(data["mean_x"], data["mean_y"], color=data["color"],
                label=data["label"], linewidth=2, zorder=3)



        # for j in range(num_points):
        #     if data["interpolated_xs"].shape[0] < 2:
        #         continue  # Cannot compute covariance with less than 2 trajectories
        #     # point_samples = np.stack((interpolated_xs[:, j], interpolated_ys[:, j]), axis=1)
        #     # draw_robust_uncertainty_ellipse(
        #     #     ax, mean_x[j], mean_y[j], point_samples,
        #     #     color=data["color"], alpha=0.3, nsig=nsig
        #     # )
        #     cov = np.cov(data["interpolated_xs"][:, j], data["interpolated_ys"][:, j])
        #     if not np.isnan(cov).any() and not np.isinf(cov).any():
        #         draw_uncertainty_ellipse(ax, data["mean_x"][j], data["mean_y"][j], cov,
        #                              color=data["color"], nsig=nsig)


    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper right')

    plt.grid(True)
    plt.show()


def plot_reward_graphs_from_csv(rl_copp_obj, model):
    """
    Plots reward graphs from a CSV file containing training metrics.
    - Reward vs Steps
    - Reward vs Time (s)
    - Reward vs Episodes
    Args:
        rl_copp_obj (RLCoppeliaManager): Instance of RLCoppeliaManager class to access paths and arguments.
        model (int): Model ID to specify which CSV file to read.
        
    The CSV must contain the following columns: 'Reward', 'Steps', 'Time (s)', 'Episodes'.
    """


    csv_path = os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, f"{rl_copp_obj.args.model_ids[model]}.csv")
    
    # Check if file exists
    if not os.path.isfile(csv_path):
        csv_path = csv_path.replace(".csv", "ms.csv")
        if not os.path.isfile(csv_path):
            logging.error(f"[Error] File not found: {csv_path}")
            return

    # Load the CSV into a DataFrame
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"[Error] Failed to read CSV: {e}")
        return

    # Check required columns
    required_columns = ["Reward", "Steps", "Time (s)", "Episodes"]
    if not all(col in df.columns for col in required_columns):
        logging.error(f"[Error] CSV must contain the following columns: {required_columns}")
        return

    # Plot 1: Reward vs Steps
    plt.figure(figsize=(8, 5))
    plt.plot(df["Steps"], df["Reward"], linestyle='-', color='blue')
    # plt.title("Reward vs Steps")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: Reward vs Time (s)
    plt.figure(figsize=(8, 5))
    plt.plot(df["Time (s)"], df["Reward"], linestyle='-', color='green')
    # plt.title("Reward vs Time (s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 3: Reward vs Episodes
    plt.figure(figsize=(8, 5))
    plt.plot(df["Episodes"], df["Reward"], linestyle='-', color='orange')
    # plt.title("Reward vs Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_reward_comparison_from_csv(rl_copp_obj, x_axis_name="Steps"):
    """
    Plots a comparison of rewards vs steps for multiple models from their respective CSV files.

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
    plt.ylabel("Reward", fontsize=18, labelpad=10)
    # plt.title("Reward Comparison Across Models", fontsize=16)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_boxplots_from_csv(rl_copp_obj):
    """
    Plots boxplots for comparing Reward and Time (s) across multiple models.

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
        plt.show()

    # ---------- Plots ----------
    plot_metric("Time (s)", "Balancing time (s)")
    plot_metric("Reward", "Reward")
    


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
            filename = f"convergence_comparison_{i}.png"
            plt.savefig(filename)
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
            - verbose (int, optional): Verbosity level for logging.

    Supported Plot Types:
        - "spider": Generates a spider chart comparing multiple models across various metrics.
        - "convergence-walltime": Plots reward convergence vs. wall time for each model.
        - "convergence-steps": Plots reward convergence vs. steps for each model.
        - "convergence-simtime": Plots reward convergence vs. simulation time for each model.
        - "convergence-episodes": Plots reward convergence vs. episodes for each model.
        - "convergence-all": Generates all convergence plots for each model.
        - "compare-rewards": Compares rewards across multiple models with smoothing and variability bands.
        - "compare-episodes_length": Compares episode lengths across multiple models.
        - "compare-convergences": Compares convergence points across multiple models.
        - "histogram_speeds": Plots histograms for linear and angular speeds for each model.
        - "grouped_bar_speeds": Creates grouped bar charts for linear and angular speeds across models.
        - "grouped_bar_targets": Creates grouped bar charts for target zone frequencies across models.
        - "bar_target_zones": Plots bar charts for target zone distributions for each model.
        - "plot_scene_trajs": Visualizes the scene and trajectories followed by the robot during testing.
        - "plot_boxplots": Generates boxplots for various metrics across models.
        - "lat_curves": Plots LAT-sim and LAT-wall curves for each model.

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
            plot_metrics_comparison_with_band(rl_copp, "rewards")
        
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
    
    if "histogram_speeds" in args.plot_types:
        plot_type_correct = True
        for model in range(len(args.model_ids)):
            logging.info(f"Plotting histogram for lineal and angular speeds for model {args.model_ids[model]}")
            plot_histogram(rl_copp, model, mode="speeds")

    if "grouped_bar_speeds" in args.plot_types:
        plot_type_correct = True
        logging.info(f"Plotting grouped bar chart for angular and linear speeds for models {args.model_ids}")
        plot_grouped_bar_chart(rl_copp, mode="speeds")
    
    if "grouped_bar_targets" in args.plot_types:
        plot_type_correct = True
        logging.info(f"Plotting grouped bar chart representing the frequency of each target zone for models {args.model_ids}")
        plot_grouped_bar_chart(rl_copp, mode="target_zones")

    if "bar_target_zones" in args.plot_types:
        plot_type_correct = True
        for model in range(len(args.model_ids)):
            logging.info(f"Plotting histogram for target zones for model {args.model_ids[model]}")
            plot_bars(rl_copp, model, mode="target_zones")

    if "plot_scene_trajs" in args.plot_types:
        plot_type_correct = True
        logging.info(f"Plotting a scene image with the trajectories followed by next models: {args.model_ids}")
        if (args.scene_to_load_folder) is None:
            logging.error("Scene config path was not provided, program will exit as it cannot continue.")
            sys.exit()
        scene_folder = os.path.join(rl_copp.paths["scene_configs"], args.scene_to_load_folder)
        logging.info(f"Scene config folder to be loaded: {scene_folder}")
        plot_scene_trajs(rl_copp, scene_folder)

    if "plot_boxplots" in args.plot_types:
        plot_type_correct = True
        logging.info(f"Plotting boxplots for models {args.model_ids}")
        compare_models_boxplots_old2(rl_copp, args.model_ids)

    if "lat_curves" in args.plot_types:
        plot_type_correct = True
        for model in range(len(args.model_ids)):
            logging.info(f"Plotting curves for LAT-sim and LAT-wall for model {args.model_ids[model]}")
            plot_lat_curves(rl_copp, model)

    if "plot_from_csv" in args.plot_types:
        plot_type_correct = True
        # for model in range(len(args.model_ids)):
        #     logging.info(f"Plotting reward graphs from csv {args.model_ids[model]}.csv for robot {args.robot_name}")
        #     plot_reward_graphs_from_csv(rl_copp, model)
        if len(args.model_ids)>1:
            logging.info(f"Plotting a reward comparison graph for models {args.model_ids} for robot {args.robot_name}")
            # plot_reward_comparison_from_csv(rl_copp, "Steps")
            # plot_reward_comparison_from_csv(rl_copp, "Time (s)")
            plot_reward_comparison_from_csv(rl_copp, "Episodes")
            plot_boxplots_from_csv(rl_copp)
    
    
    if not plot_type_correct:
        logging.error(f"Please check plot types: {args.plot_types}")