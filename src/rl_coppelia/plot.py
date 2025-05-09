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
        "Convergence Sim Time",
        "Mean Reward",
        "Episode Efficiency",
        "Episode Completion Rate",  
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
    logging.info(f"Reward at convergence point: {reward_at_convergence}")

    # TODO: Get action time from training csv for showing it in the graph
    # rl_copp.params_test["sb3_algorithm"] = utils.get_algorithm_for_model(model_name, train_records_csv_name)
    # get_data_from_training_csv

    # Plot results
    plt.figure(figsize=(8, 5))
    # plt.plot(x_axis_values, reward, label='Original Data', marker='o', linestyle='')
    # plt.plot(x_axis_values, reward_fit, label='Exponential Fit', linestyle='--')
    if x_axis == "WallTime":
        plt.plot(x_axis_values, reward, label='Original Data', marker='o', linestyle='')
        plt.plot(x_axis_values, reward_fit, label='Exponential Fit', linestyle='--')
        plt.xlabel('Wall time (hours)')
        plt.axvline(convergence_point, color='r', linestyle=':', label=f'Convergence Wall Time: {convergence_point:.2f}h')
        title = title + ' vs Wall Time'
    elif x_axis == "Steps":
        plt.plot(x_axis_values, reward, label='Original Data', marker='o', linestyle='')
        plt.plot(x_axis_values, reward_fit, label='Exponential Fit', linestyle='--')
        plt.xlabel('Steps')
        plt.axvline(convergence_point, color='r', linestyle=':', label=f'Convergence Steps: {convergence_point:.2f}')
        title = title +  ' vs Steps'
    elif x_axis == "SimTime":
        plt.plot(x_axis_values, reward, label='Original Data', marker='o', linestyle='')
        plt.plot(x_axis_values, reward_fit, label='Exponential Fit', linestyle='--')
        plt.xlabel('Simulation time (hours)')
        plt.axvline(convergence_point, color='r', linestyle=':', label=f'Convergence Sim Time: {convergence_point:.2f}h')
        title = title + ' vs Sim Time'
    elif x_axis == "Episodes":
        plt.plot(x_axis_values, reward, label='Original Data', marker='o', linestyle='')
        plt.plot(x_axis_values, reward_fit, label='Exponential Fit', linestyle='--')
        plt.xlabel('Episodes')
        plt.axvline(convergence_point, color='r', linestyle=':', label=f'Convergence Episodes: {convergence_point:.2f}')
        title = title +  ' vs Episodes'
    plt.ylabel('Reward')
    plt.legend()
    plt.title(title + ": Model " + str(rl_copp_obj.args.model_ids[model_index]))
    plt.grid()
    plt.show()
    
    


def plot_metrics_comparison (rl_copp_obj, metric, title = "Comparison"):
    """
    Plot the same metric of multiple models for comparing them. X axis will be the number of steps.

    Args:
    - rl_copp_object (RLCoppeliaManager): Instance of RLCoppeliaManager class just for managing the args and the base path.
    - title (str): The title of the chart.
    """
    
    for model_index in range(len(rl_copp_obj.args.model_ids)):
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



def plot_histogram (rl_copp_obj, model_index, title = "Histogram for angular speed"):
    """
    Plots a histogram for showing the angular speeds of the robot during the testing
    process.

    Args:
    - rl_copp_object (RLCoppeliaManager): Instance of RLCoppeliaManager class just for managing the args and the base path.
    - title (str): The title of the chart.
    """
    # CSV File path to get data from
    # Patrón flexible que captura cualquier texto entre el ID del modelo y "speeds"
    file_pattern = f"{rl_copp_obj.args.robot_name}_model_{rl_copp_obj.args.model_ids[model_index]}_*_speeds_*.csv"
    files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, "testing_metrics", file_pattern))

    # Read CSV
    df = pd.read_csv(files[0])

    # Reemplaza 'velocidad_angular' con el nombre exacto de tu columna
    angular_velocities = df['Angular speed']

    print(f"Estadísticas de velocidad angular:")
    print(f"Media: {angular_velocities.mean():.4f} rad/s")
    print(f"Mediana: {angular_velocities.median():.4f} rad/s")
    print(f"Desviación estándar: {angular_velocities.std():.4f} rad/s")
    print(f"Mínimo: {angular_velocities.min():.4f} rad/s")
    print(f"Máximo: {angular_velocities.max():.4f} rad/s")

    # Configurar el histograma
    plt.figure(figsize=(10, 6))

    # Crear bins equiespaciados desde -0.5 hasta 0.5
    bins = np.linspace(-0.5, 0.5, 21)  # 21 bins para tener 20 intervalos

    # Crear el histograma
    plt.hist(angular_velocities, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)

    # Añadir títulos y etiquetas
    plt.title(title + ": Model " + str(rl_copp_obj.args.model_ids[model_index]), fontsize=14)
    plt.xlabel('Angular speed (rad/s)', fontsize=12)
    plt.ylabel('Frequence', fontsize=12)
    plt.grid(axis='y', alpha=0.75)
    plt.legend()

    # Ajustar los límites del eje x para asegurar que se vea todo el rango
    plt.xlim(-0.55, 0.55)

    # Mostrar el histograma
    plt.tight_layout()
    plt.show()


def plot_histograms_comparison(rl_copp_obj, model_indices, title="Histogram for angular speed", alpha=0.6, colors=None):
    """
    Plots histograms for showing the angular speeds of multiple robot models during the testing
    process, overlaying them for easy comparison.
    
    Args:
    - rl_copp_object (RLCoppeliaManager): Instance of RLCoppeliaManager class for managing args and base path.
    - model_indices (list): List of model indices to plot.
    - title (str): The title of the chart.
    - alpha (float): Transparency level for the histograms.
    - colors (list, optional): List of colors for the histograms. If None, uses default colors.
    """
    # Configurar el histograma
    plt.figure(figsize=(12, 7))
    
    # Crear bins equiespaciados desde -0.5 hasta 0.5
    bins = np.linspace(-0.5, 0.5, 21)  # 21 bins para tener 20 intervalos
    
    # Si no se proporcionan colores, usar una paleta predeterminada
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_indices)))
    
    print(model_indices)
    # Crear histogramas para cada modelo
    for i, model_idx in enumerate(model_indices):
        print(model_idx)
        # CSV File path
        file_pattern = f"{rl_copp_obj.args.robot_name}_model_{model_idx}_*_speeds_*.csv"
        print(file_pattern)
        files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, 
                                       "testing_metrics", file_pattern))
        print(files)
        if not files:
            print(f"No files found for model {model_idx}")
            continue
            
        # Read CSV
        df = pd.read_csv(files[0])
        angular_velocities = df['Angular speed']
        
        # Imprimir estadísticas
        print(f"\nEstadísticas de velocidad angular para modelo {model_idx}:")
        print(f"Media: {angular_velocities.mean():.4f} rad/s")
        print(f"Mediana: {angular_velocities.median():.4f} rad/s")
        print(f"Desviación estándar: {angular_velocities.std():.4f} rad/s")
        print(f"Mínimo: {angular_velocities.min():.4f} rad/s")
        print(f"Máximo: {angular_velocities.max():.4f} rad/s")
        
        # Crear el histograma con densidad=True para normalizar y permitir comparación justa
        plt.hist(angular_velocities, bins=bins, alpha=alpha, color=colors[i], 
                 edgecolor='black', label=f'Model {model_idx}',
                 density=True, histtype='stepfilled')
    
    # Añadir títulos y etiquetas
    plt.title(title, fontsize=14)
    plt.xlabel('Angular speed (rad/s)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='upper right')
    
    # Añadir una línea vertical en el cero
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # Ajustar los límites del eje x para asegurar que se vea todo el rango
    plt.xlim(-0.55, 0.55)
    
    # Mostrar el histograma
    plt.tight_layout()
    plt.show()


def plot_histograms_comparison_v2(rl_copp_obj, model_indices, num_intervals=5, title_prefix="Angular Speed Distribution"):
    """
    Plots multiple histograms for angular speeds, each focusing on a specific range.
    Each plot compares all specified models within that interval.
    
    Args:
    - rl_copp_obj: RLCoppeliaManager instance
    - model_indices (list): List of model indices to compare
    - num_intervals (int): Number of intervals to divide the speed range (-0.5 to 0.5)
    - title_prefix (str): Prefix for the title of each plot
    """
    # Carga los datos de todos los modelos primero
    model_data = {}
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_indices)))
    
    for i, model_idx in enumerate(model_indices):
        file_pattern = f"{rl_copp_obj.args.robot_name}_model_{model_idx}_*_speeds_*.csv"
        files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, 
                                      "testing_metrics", file_pattern))
        
        if not files:
            print(f"No files found for model {model_idx}")
            continue
            
        df = pd.read_csv(files[0])
        angular_velocities = df['Angular speed']
        
        # Imprimir estadísticas
        print(f"\nEstadísticas para modelo {model_idx}:")
        print(f"Media: {angular_velocities.mean():.4f} rad/s")
        print(f"Mediana: {angular_velocities.median():.4f} rad/s")
        print(f"Desviación estándar: {angular_velocities.std():.4f} rad/s")
        print(f"Mínimo: {angular_velocities.min():.4f} rad/s")
        print(f"Máximo: {angular_velocities.max():.4f} rad/s")
        
        model_data[model_idx] = {
            'velocities': angular_velocities,
            'color': colors[i],
            'model_id': model_idx
        }
    
    # Definir los intervalos
    min_speed = -0.5
    max_speed = 0.5
    interval_size = (max_speed - min_speed) / num_intervals
    intervals = [(min_speed + i * interval_size, min_speed + (i + 1) * interval_size) 
                for i in range(num_intervals)]
    
    # Crear un gráfico para cada intervalo
    for interval_start, interval_end in intervals:
        plt.figure(figsize=(10, 6))
        
        # Definir bins solo para este intervalo (más detallado)
        bins = np.linspace(interval_start, interval_end, 15)  # 15 bins para cada intervalo
        
        # Crear histogramas para cada modelo en este intervalo
        for model_idx, data in model_data.items():
            # Filtrar velocidades solo para este intervalo
            mask = (data['velocities'] >= interval_start) & (data['velocities'] <= interval_end)
            filtered_velocities = data['velocities'][mask]
            
            if len(filtered_velocities) > 0:  # Solo graficar si hay datos en este rango
                plt.hist(filtered_velocities, bins=bins, 
                         alpha=0.7, color=data['color'], 
                         edgecolor='black', 
                         label=f'Model {data["model_id"]}')
        
        # Añadir etiquetas y leyenda
        plt.title(f"{title_prefix}: [{interval_start:.2f}, {interval_end:.2f}] rad/s", fontsize=14)
        plt.xlabel('Angular speed (rad/s)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Ajustar los límites del eje x para mostrar solo este intervalo
        plt.xlim(interval_start, interval_end)
        
        plt.tight_layout()
        plt.show()


def plot_histograms_comparison_v3(rl_copp_obj, model_indices, num_intervals=10, title="Angular Speed Distribution by Intervals"):
    """
    Creates a grouped bar chart where each group represents a velocity interval,
    and within each group there's one bar per model.
    
    Args:
    - rl_copp_obj: RLCoppeliaManager instance
    - model_indices (list): List of model indices to compare
    - num_intervals (int): Number of intervals to divide the speed range (-0.5 to 0.5)
    - title (str): Title for the plot
    """
    # Define velocity intervals
    min_speed = -0.5
    max_speed = 0.5
    interval_size = (max_speed - min_speed) / num_intervals
    intervals = [(min_speed + i * interval_size, min_speed + (i + 1) * interval_size) 
                for i in range(num_intervals)]
    
    # Prepare data structure for frequencies
    model_names = [f"Model {idx}" for idx in model_indices]
    interval_labels = [f"[{a:.2f}, {b:.2f}]" for a, b in intervals]
    
    # Matrix to store frequencies: rows=models, columns=intervals
    frequencies = np.zeros((len(model_indices), len(intervals)))
    
    # Load data and calculate frequencies for each model and interval
    for i, model_idx in enumerate(model_indices):
        file_pattern = f"{rl_copp_obj.args.robot_name}_model_{model_idx}_*_speeds_*.csv"
        files = glob.glob(os.path.join(rl_copp_obj.base_path, "robots", rl_copp_obj.args.robot_name, 
                                       "testing_metrics", file_pattern))
        
        if not files:
            print(f"No files found for model {model_idx}")
            continue
            
        df = pd.read_csv(files[0])
        angular_velocities = df['Angular speed']
        
        # Count frequencies for each interval
        for j, (interval_start, interval_end) in enumerate(intervals):
            mask = (angular_velocities >= interval_start) & (angular_velocities < interval_end)
            frequencies[i, j] = np.sum(mask)
        
        # Normalize to percentage if desired
        frequencies[i, :] = frequencies[i, :] / len(angular_velocities) * 100
    
    # Create the grouped bar chart
    plt.figure(figsize=(15, 8))
    
    # Set width of bars and positions
    bar_width = 0.8 / len(model_indices)
    r = np.arange(len(intervals))
    
    # Plot bars for each model
    for i, model_idx in enumerate(model_indices):
        position = r + i * bar_width - (len(model_indices) - 1) * bar_width / 2
        plt.bar(position, frequencies[i, :], width=bar_width, 
                label=f"Model {model_idx}")
    
    # Add labels, title and legend
    plt.xlabel('Angular Speed Intervals (rad/s)', fontsize=12)
    plt.ylabel('Percentage of Samples (%)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(r, interval_labels, rotation=45, ha='right')
    plt.legend()
    
    # Add grid for readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Optional: Return the frequency data for further analysis
    return frequencies, interval_labels, model_names


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
        for model in range(len(args.model_ids)):
            logging.info(f"Plotting all the convergence graphs for model {args.model_ids[model]}")
            convergence_modes = ["WallTime", "Steps", "SimTime", "Episodes"]
            for i in range(len(convergence_modes)):
                plot_convergence(rl_copp, model, convergence_modes[i])

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
    
    if "histogram_speeds" in args.plot_types:
        plot_type_correct = True
        for model in range(len(args.model_ids)):
            logging.info(f"Plotting histogram for angular speeds for model {args.model_ids[model]}")
            plot_histogram(rl_copp, model)

    if "histogram_speed_comparison" in args.plot_types:
        plot_type_correct = True
        if len(args.model_ids) <= 1:    
            logging.error(f"Please, introduce more than one model ID for comparing different histograms. Models specified: {args.model_ids}")
        else:
            logging.info(f"Plotting histogram for angular speeds for models {args.model_ids}")
            plot_histograms_comparison_v3(rl_copp, args.model_ids)
    
    if not plot_type_correct:
        logging.error(f"Please check plot types: {args.plot_types}")




    
