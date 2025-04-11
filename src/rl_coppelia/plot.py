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

import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
import os

import pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from common import utils
from common.rl_coppelia_manager import RLCoppeliaManager


def plot_spider(rl_copp_obj, title='Models Comparison'):
    """
    Plots multiple spider charts on the same figure to compare different models.

    Parameters:
    - rl_copp_object (RLCoppeliaManager): Instance of RLCoppeliaManager class just for managing the args and the base path.
    - title (str): The title of the chart.
    """

    # Categories for plotting
    categories = [
        "Convergence Time",
        "Actor Performance",
        "Critic Performance",
        # "Train Episode Efficiency",
        # "Train Reward",
        "Test Reward",
        "Test Episode Efficiency",
        "Test Episode Completion Rate",  
    ]

    # Get metrics from testing
    testing_csv_path = os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "testing_metrics", "test_records.csv")
    test_column_names = [
        "Avg reward",
        "Avg time reach target",
        "Percentage terminated"
    ]
    test_data = utils.get_data_for_spider(testing_csv_path, rl_copp_obj.args, test_column_names)

    # Get metrics from training
    training_csv_path = os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "training_metrics", "train_records.csv")
    train_column_names = [
        "Action time (s)",
        "Time to converge (h)",
        "train/actor_loss",
        "train/critic_loss",
        # "rollout/ep_len_mean",
        # "rollout/ep_rew_mean"
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
    num_vars = len(labels)  # Vars number
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()   # Create angle for each category
    angles += angles[:1]    # Close th circle
    _, ax = plt.subplots(figsize=(5, 5), dpi=100, subplot_kw=dict(polar=True))    # Create the figure

    for data, name in zip(data_list, names):    # Plot each data set
        data = data + data[:1]  # Assure that we are closing the circle
        ax.plot(angles, data, linewidth=2, linestyle='solid', label=name)
        ax.fill(angles, data, alpha=0.25)

    # Labels of the axis
    ax.set_yticklabels([])  # Remove labels from radial axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, ha='center', rotation=60)
    ax.spines['polar'].set_visible(False)

    # Set the radial axis limits
    ax.set_ylim(0, 1) 

    # Add the leyend and title
    ax.legend(loc='upper left', bbox_to_anchor=(1.3, 1.1))  # Ajustar posición de la leyenda
    ax.set_title(title, size=16, color='black', y=1.1)

    # Show the plot
    plt.show()


def plot_convergence (rl_copp_obj, model_index, x_axis, title = "Reward Convergence Analysis"):
    """
    Plots the reward-time graph and shows the time at which the reward stabilizes (converges) based on a first-order fit.

    Parameters:
    - rl_copp_object (RLCoppeliaManager): Instance of RLCoppeliaManager class just for managing the args and the base path.
    - title (str): The title of the chart.
    """
    # CSV File path to get data from
    file_pattern = f"{rl_copp_obj.args.robot_name}_model_{rl_copp_obj.args.model_ids[model_index]}_*.csv"
    files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "training_metrics", file_pattern))
    

    # Calculate the convergence time
    convergence_point, reward_fit, x_axis_values, reward, reward_at_convergence = utils.get_convergence_point (files[0], x_axis, convergence_threshold=0.01)
    print(reward_at_convergence)

    # Plot results
    plt.figure(figsize=(8, 5))
    # plt.plot(x_axis_values, reward, label='Original Data', marker='o', linestyle='')
    # plt.plot(x_axis_values, reward_fit, label='Exponential Fit', linestyle='--')
    if x_axis == "Time":
        plt.plot(x_axis_values, reward, label='Original Data', marker='o', linestyle='')
        plt.plot(x_axis_values, reward_fit, label='Exponential Fit', linestyle='--')
        plt.xlabel('Time (hours)')
        plt.axvline(convergence_point, color='r', linestyle=':', label=f'Convergence Time: {convergence_point:.2f}h')
        title = title + ' vs Time'
    elif x_axis == "Step":
        plt.plot(x_axis_values, reward, label='Original Data', marker='o', linestyle='')
        plt.plot(x_axis_values, reward_fit, label='Exponential Fit', linestyle='--')
        plt.xlabel('Step')
        plt.axvline(convergence_point, color='r', linestyle=':', label=f'Convergence Steps: {convergence_point:.2f}')
        title = title +  ' vs Step'
    plt.ylabel('Reward')
    plt.legend()
    plt.title(title + ": Model " + str(rl_copp_obj.args.model_ids[model_index]))
    plt.grid()
    plt.show()
    
    


def plot_metrics_comparison (rl_copp_obj, metric, title = "Comparison"):
    """
    Plot the same metric of multiple models for comparing them. X axis will be the number of steps.

    Parameters:
    - rl_copp_object (RLCoppeliaManager): Instance of RLCoppeliaManager class just for managing the args and the base path.
    - title (str): The title of the chart.
    """
    
    for model_index in range(len(rl_copp_obj.args.model_ids)):
        # CSV File path to get data from
        file_pattern = f"{rl_copp_obj.args.robot_name}_model_{rl_copp_obj.args.model_ids[model_index]}_*.csv"
        files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "training_metrics", file_pattern))
        
        # Read the CSV file
        df = pd.read_csv(files[0])
        
        # Extract steps and rewards
        steps = df['Step'].values
        if metric == "rewards":
            data = df['rollout/ep_rew_mean'].values
            pre_title = "Rewards "
        elif metric == "episodes_length":
            data = df['rollout/ep_len_mean'].values
            pre_title = "Episodes Length "
        y_label = pre_title
        
        # Plot the rewards for each model
        plt.plot(steps, data, label=f'Model {rl_copp_obj.args.model_ids[model_index]}')

    # Add labels and title
    plt.xlabel('Steps')

    plt.ylabel(y_label)

    plt.title(pre_title + title + "- Models " + str(rl_copp_obj.args.model_ids))
    
    # Add legend to differentiate between models
    plt.legend()

    # Show the plot
    plt.show()




def main(args):
    """
    Executes multiple testing runs. This method allows the user to test multiple models just by indicating
    a list of model ids.
    """

    rl_copp = RLCoppeliaManager(args)

    plot_type_correct = False

    if "spider" in args.plot_types:
        plot_type_correct = True
        if len(args.model_ids) <= 1:    # A spider graph doesn't make sense if there are less than 2 models. In fact it doesn't make sense to compare less than 3 models (#TODO)
            logging.error(f"Please, introduce more than one model ID for creating a spider graph. Models specified: {args.model_ids}")
        
        else:
            logging.info(f"Plotting spider graph for comparing the models {args.model_ids}")
            plot_spider(rl_copp)

    if "convergence-time" in args.plot_types:
        plot_type_correct = True
        for model in range(len(args.model_ids)):
            logging.info(f"Plotting convergence-vs-time graph for model {args.model_ids[model]}")
            plot_convergence(rl_copp, model, "Time")
    
    if "convergence-steps" in args.plot_types:
        plot_type_correct = True
        for model in range(len(args.model_ids)):
            logging.info(f"Plotting convergence-vs-steps graph for model {args.model_ids[model]}")
            plot_convergence(rl_copp, model, "Step")

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
    
    if not plot_type_correct:
        logging.error(f"Please check plot types: {args.plot_types}")




    
