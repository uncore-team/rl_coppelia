from __future__ import annotations

import csv
from datetime import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QLabel, QCheckBox, QSpinBox, QFileDialog, QLabel,
    QHBoxLayout, QToolTip, QDialog, QGroupBox, QMessageBox, QComboBox, QDoubleSpinBox,
    QTableWidgetItem, QTableWidget, QDialogButtonBox
)
from PyQt5.QtCore import Qt
import logging

from matplotlib import pyplot as plt
from common import utils
from rl_coppelia import cli, train, test, plot, auto_training, auto_testing, retrain, sat_training
from pathlib import Path
from PyQt5.QtCore import QTimer, QThread, pyqtSignal,QSize,QPoint,QEvent
from PyQt5.QtWidgets import QProgressBar, QGridLayout, QHBoxLayout, QTextEdit, QSizePolicy, QScrollArea, QAbstractItemView,QListWidgetItem, QListWidget, QComboBox
from PyQt5.QtGui import QIcon, QPixmap, QIntValidator, QDoubleValidator
import pkg_resources
from shutil import which
import time
import psutil
import pandas as pd
from matplotlib.patches import Circle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from gui.services import get_rl_coppelia_path_from_bashrc

PLOT_TYPES = ["spider", "convergence-walltime", "convergence-simtime", "convergence-steps", "convergence-episodes", 
              "convergence-all", "compare-rewards", "compare-episodes_length", "compare-convergences",
                "histogram_speeds", "grouped_bar_speeds", "grouped_bar_targets", "bar_target_zones",
                "plot_scene_trajs", "plot_boxplots", "lat_curves", "plot_from_csv"]

PLOT_TYPE_DESCRIPTIONS = {
    "spider": "Radar chart comparing performance metrics.",
    "convergence-walltime": "Plot showing how quickly the models converge in terms of time.",
    "convergence-steps": "Shows convergence in terms of learning steps.",
    "compare-rewards": "Bar or line chart comparing final rewards between models.",
    "compare-episodes_length": "Shows average episode lengths.",
    "histogram_speeds": "Histogram of robot speeds during operation.",
    "histogram_speed_comparison": "Compares speed distributions between models.",
    "hist_target_zones": "Histogram showing time spent in each target zone.",
    "bar_target_zones": "Bar chart comparing frequency of reaching each target zone.",
}

PARAM_TOOLTIPS = {
    "var_action_time_flag": "Add the action time as a learning variable for the agent, so it will be variable.",
    "fixed_actime": "Fixed duration (in seconds) for each action.",
    "bottom_actime_limit": "Minimum allowed action time when 'var_action_time_flag' is set.",
    "upper_actime_limit": "Maximum allowed action time when 'var_action_time_flag' is set.",
    "bottom_lspeed_limit": "Minimum linear speed limit.",
    "upper_lspeed_limit": "Maximum linear speed limit.",
    "bottom_aspeed_limit": "Minimum angular speed limit.",
    "upper_aspeed_limit": "Maximum angular speed limit.",
    "finish_episode_flag": "Enable the robot to decide when to finish the episodes.",
    "dist_thresh_finish_flag": "Distance threshold for considering an episode successful if 'finish_episode_flag' is set.",
    "obs_time": "Include time in observation space.",
    "reward_dist_1": "Threshold for entering in target zone 1.",
    "reward_1": "Value for reward when reaching target zone 1.",
    "reward_dist_2": "Threshold for entering in target zone 2.",
    "reward_2": "Value for reward when reaching target zone 2.",
    "reward_dist_3": "Threshold for entering in target zone 3.",
    "reward_3": "Value for reward when reaching target zone 3.",
    "max_count": "Max number of steps per episode.",
    "max_time": "Max time per episode.",
    "max_dist": "Max distance allowed (robot-target).",
    "finish_flag_penalty": "Penalty if distance between robot and target is under the threshold 'dist_thresh_finish_flag' if 'finish_episode_flag' is set.",
    "overlimit_penalty": "Penalty for exceed time or distance limit.",
    "crash_penalty": "Penalty for collision.",
    "max_crash_dist": "Distance between robot and object to consider a crash.",
    "max_crash_dist_critical": "Distance between robot and object to consider a crash (for lateral collisions)",
    "sb3_algorithm": "RL algorithm to use (e.g., SAC, PPO).",
    "policy": "Network policy type (e.g., MlpPolicy).",
    "total_timesteps": "Number of training steps.",
    "callback_frequency": "How often to save a callback.",
    "n_training_steps": "Steps before each training update.",
    "testing_iterations": "Number of testing episodes."
}

PLOT_TOOLTIPS = """<b>Available Plot Types:</b><ul>
<li><b>spider</b>: Radar chart comparing key performance metrics.</li>
<li><b>convergence-walltime</b>: Shows model convergence during learning over real time.</li>
<li><b>convergence-simtime</b>: Shows model convergence during learning over simulation time.</li>
<li><b>convergence-steps</b>: Shows model convergence during learning over steps.</li>
<li><b>convergence-episodes</b>:Shows model convergence during learning over episodes.</li>
<li><b>convergence-all</b>: Shows a model convergence comparison across walltime, simtime, steps and episodes.</li>
<li><b>compare-rewards</b>: Bar chart comparing total rewards across models.</li>
<li><b>compare-episodes_length</b>: Compares average episode lengths.</li>
<li><b>compare-convergences</b>: Overlays multiple convergence curves.</li>
<li><b>histogram_speeds</b>: Histogram of robot speeds across runs.</li>
<li><b>grouped_bar_speeds</b>: Grouped bar chart of average speeds by model.</li>
<li><b>grouped_bar_targets</b>: Grouped bars showing success in reaching target zones.</li>
<li><b>bar_target_zones</b>: Bar chart of frequency per target zone.</li>
<li><b>plot_scene_trajs</b>: Visual plot of robot trajectories over the scene.</li>
<li><b>plot_boxplots</b>: Boxplots of reward and episode metrics.</li>
<li><b>lat_curves</b>: LAT curves for action execution times.</li>
<li><b>plot_from_csv</b>: Generate boxplots and a learning reward comparison chart from custom CSV file data.</li>
</ul>"""


class ManualParamsDialog(QDialog):
    """
    Dialog window for manually editing and saving robot training parameter files.

    This interface allows users to edit default environment and training parameters,
    modify them through input widgets, and save the result as a new JSON file.

    Attributes:
        params_saved (pyqtSignal): Signal emitted with the filename when a new param file is saved.
    """
    params_saved = pyqtSignal(str)  # Signal to emit when parameters are saved

    def __init__(self, base_path, robot_name, parent=None):
        """
        Initializes the manual parameter definition dialog.

        Args:
            base_path (str): Path to the root directory of the project.
            robot_name (str): Name of the robot whose parameters are being edited.
            parent (QWidget, optional): Parent widget of the dialog. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowTitle("Manual parameter definition")
        self.base_path = base_path  # Base path to access configuration files
        self.robot_name = robot_name    # Robot identifier

        self.default_path = os.path.join(self.base_path, "configs", "params_default_file.json") # Path to default parameter file
        self.target_dir = os.path.join(self.base_path, "configs")   # Directory to save new parameter files
        os.makedirs(self.target_dir, exist_ok=True)     # Create the directory if it doesn't exist

        with open(self.default_path, "r") as f:     # Load default parameters from JSON file
            self.default_data = json.load(f)

        self.field_widgets = {}  # Maps (section, key) tuples to their associated input widgets

        layout = QVBoxLayout()  # Main layout of the dialog
        for section in ["params_env", "params_train"]:  # Add editable sections for environment and training
            box = self.build_section(section, self.default_data.get(section, {}))
            layout.addWidget(box)

        self.file_name_input = QLineEdit()  # Input field for the new file name
        self.file_name_input.setPlaceholderText("Enter name for new param file")  # Set placeholder text
        layout.addWidget(QLabel("New file name (without .json):"))  # Label for file name input
        layout.addWidget(self.file_name_input)  # Add input to layout

        buttons_layout = QHBoxLayout()  # Layout for dialog buttons
        buttons_layout.addStretch()  # Push buttons to the right

        self.warning_label = QLabel("")  # Label to show warnings (e.g., file exists)
        self.warning_label.setStyleSheet("color: red;")  # Make warning text red
        layout.addWidget(self.warning_label)  # Add warning label to main layout

        apply_button = QPushButton("Apply")  # Button to confirm changes
        apply_button.clicked.connect(self.apply_changes)  # Connect to apply_changes handler

        cancel_button = QPushButton("Cancel")  # Button to cancel dialog
        cancel_button.clicked.connect(self.reject)  # Close dialog without saving

        apply_button.setAutoDefault(False)  # Disable default button behavior
        cancel_button.setAutoDefault(False)
        apply_button.setDefault(False)
        cancel_button.setDefault(False)

        buttons_layout.addWidget(cancel_button)  # Add cancel button to layout
        buttons_layout.addWidget(apply_button)  # Add apply button to layout

        layout.addLayout(buttons_layout)  # Add button layout to main layout

        self.setLayout(layout)  # Set main layout for the dialog


    def build_section(self, section_name, fields):
        """
        Creates a QGroupBox containing labeled input fields for a parameter section.

        Args:
            section_name (str): Name of the section ('params_env' or 'params_train').
            fields (dict): Dictionary with parameter keys and default values.

        Returns:
            QGroupBox: Group box containing the labeled inputs.
        """
        if section_name == "params_env":
            section_name = "Parameters - Environment"
        elif section_name == "params_train":
            section_name = "Parameters - Training"
        group = QGroupBox(section_name)  # Create container for the section
        layout = QGridLayout()  # Use grid layout to align labels and fields
        layout.setHorizontalSpacing(25)  # Add space between label and field columns

        row = 0
        col = 0
        for key, value in fields.items():
            label = QLabel(key)
            label.setToolTip(PARAM_TOOLTIPS.get(key, ""))  # Show tooltip if available

            if isinstance(value, bool):
                field = QCheckBox()  # Use checkbox for boolean parameters
                field.setChecked(value)
            else:
                field = QLineEdit(str(value))  # Use line edit for numeric or text inputs
                if isinstance(value, int):
                    field.setValidator(QIntValidator())  # Restrict input to integers
                elif isinstance(value, float):
                    field.setValidator(QDoubleValidator())  # Restrict input to floats

            self.field_widgets[(section_name, key)] = field  # Save widget reference for later

            layout.addWidget(label, row, col * 2)  # Place label
            layout.addWidget(field, row, col * 2 + 1)  # Place field next to label

            col += 1
            if col == 2:  # Wrap to next row after 2 columns
                col = 0
                row += 1

        group.setLayout(layout)
        return group
    

    def apply_changes(self):
        '''
        Gathers input values, validates them, and saves to a new JSON file.
        Emits a signal with the new filename if successful.
        '''
        new_data = json.loads(json.dumps(self.default_data))  # Deep copy of original data

        for (section, key), widget in self.field_widgets.items():
            if section == "Parameters - Environment":
                section = "params_env"
            elif section == "Parameters - Training":
                section = "params_train"

            if isinstance(widget, QCheckBox):
                value = widget.isChecked()  # Get boolean value from checkbox
            else:
                text = widget.text()
                if text.strip() == "":
                    continue  # Skip empty fields
                original = self.default_data[section][key]
                try:
                    if isinstance(original, int):
                        value = int(text)  # Convert input to int if expected
                    elif isinstance(original, float):
                        value = float(text)  # Convert input to float if expected
                    else:
                        value = text  # Leave as string
                except ValueError:
                    value = text  # Fallback to string if conversion fails

            new_data[section][key] = value  # Update modified value

        name = self.file_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing name", "Please provide a name for the new parameters file.")  # Show warning for missing filename
            return

        file_path = os.path.join(self.target_dir, f"{name}.json")
        if os.path.exists(file_path):
            self.warning_label.setText(f"❌ File '{file_path}' already exists.")  # Warn if file exists
            return
        else:
            self.warning_label.setText("")  # Clear previous warning

        try:
            with open(file_path, "w") as f:
                json.dump(new_data, f, indent=4)  # Save parameters to JSON
            self.params_saved.emit(f"{name}.json")  # Emit signal with filename
            self.accept()  # Close the dialog
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Could not save file: {e}")  # Show error if saving fails


class RobotAngleSpinBox(QDoubleSpinBox):
    '''
    A QDoubleSpinBox that commits its value on Enter key press and clears focus.
    '''
    def __init__(self, parent=None, on_commit=None):
        super().__init__(parent)
        self.on_commit = on_commit

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.on_commit:
                self.on_commit(self.value())
            self.clearFocus()
            event.accept()
        else:
            super().keyPressEvent(event)


class AutoClearFocusLineEdit(QLineEdit):
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.clearFocus()
            event.accept()
        else:
            super().keyPressEvent(event)

    def focusOutEvent(self, event):
        self.clearFocus()
        super().focusOutEvent(event)


class CustomSceneDialog(QDialog):
    '''
    Dialog window for creating or editing custom scenes with targets and obstacles.
    Allows users to place elements on a 2D map, set robot orientation, and save the scene configuration.
    '''
    def __init__(self, base_path, robot_name, parent=None, edit_mode=False):
        super().__init__(parent)
        self.setWindowTitle("Create Custom Scene")
        self.base_path = base_path
        self.robot_name = robot_name
        self.scene_elements = []
        self.scene_items = []  # (patch, data)
        self.scene_elements.append(["robot", 0, 0, -1.1415]) 
        self.edit_mode = edit_mode
        self.parent = parent

        # self.setFixedSize(650, 600)
        self.setMinimumSize(600, 600)

        # Main layout with padding
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10) 

        # Upper layout: two columns
        upper_layout = QHBoxLayout()

        # Right column: scene controls
        right_column = QVBoxLayout()
        right_column.setSpacing(3)

        # Scene name input
        self.name_input = AutoClearFocusLineEdit()
        self.name_input.setPlaceholderText("Enter scene folder name (no spaces)")
        scene_name_block = QVBoxLayout()
        
        scene_name_block.setSpacing(0)
        scene_name_block.addWidget(QLabel("Scene folder name:"))
        scene_name_block.addWidget(self.name_input)
        scene_name_block.setContentsMargins(0, 0, 0, 0)
        right_column.addLayout(scene_name_block)

        # Element to place input
        self.element_type_combo = QComboBox()
        self.element_type_combo.addItems(["target", "obstacle"])
        element_block = QVBoxLayout()
        
        element_block.setSpacing(2)
        element_block.addWidget(QLabel("Element to place:"))
        element_block.addWidget(self.element_type_combo)
        element_block.setContentsMargins(0, 10, 0, 0)
        right_column.addLayout(element_block)

        # Rotation controls
        rotation_label = QLabel("Rotate the robot:")
        rotation_layout = QHBoxLayout()
        rotation_layout.setSpacing(6)

        self.robot_orientation_input = RobotAngleSpinBox(on_commit=self.update_robot_orientation_from_enter)
        self.robot_orientation_input.setDecimals(3)
        self.robot_orientation_input.setSingleStep(0.1)
        self.robot_orientation_input.setRange(-3.1416, 3.1416)
        self.robot_orientation_input.setValue(-1.141)
        self.robot_orientation_input.setSuffix(" rad")
        self.robot_orientation_input.setFocusPolicy(Qt.ClickFocus)
        self.robot_orientation_input.valueChanged.connect(self.update_robot_orientation_from_input)

        self.rotate_left_button = QPushButton("⟲")
        self.rotate_right_button = QPushButton("⟳")
        self.rotate_left_button.setFixedWidth(30)
        self.rotate_right_button.setFixedWidth(30)
        self.rotate_left_button.setToolTip("Rotate 15° counter-clockwise")
        self.rotate_right_button.setToolTip("Rotate 15° clockwise")
        self.rotate_left_button.setFocusPolicy(Qt.NoFocus)
        self.rotate_right_button.setFocusPolicy(Qt.NoFocus)
        self.rotate_left_button.clicked.connect(lambda: self.rotate_robot(-0.2618))
        self.rotate_right_button.clicked.connect(lambda: self.rotate_robot(0.2618))

        rotation_layout.addWidget(rotation_label)
        rotation_layout.addWidget(self.robot_orientation_input)
        rotation_layout.addStretch()
        rotation_layout.addWidget(self.rotate_left_button)
        rotation_layout.addWidget(self.rotate_right_button)
        rotation_layout.setContentsMargins(0, 15, 0, 0) 

        right_column.addLayout(rotation_layout)


        # Left column: information
        left_column = QVBoxLayout()
        info_label = QLabel()
        info_label.setWordWrap(True)
        info_label.setText("""
        <div style="text-align: justify;">
        <b>Welcome to the Scene Editor!</b><br><br>
        With this tool you can create or modify scenes by adding new <b>targets</b> and <b>obstacles</b>, and changing the <b>orientation of the robot</b>.<br><br>
        It's easy to use: just select the type of object you want to place, and click anywhere on the map to position it. <br>
        To remove an object, simply click on it again.<br><br>
        Enjoy! 🛠️
        </div>
        """)
        info_label.setStyleSheet("text-align: justify;")
        left_column.addWidget(info_label)

        # Add columns to the upper layout
        upper_layout.addLayout(left_column)
        upper_layout.addLayout(right_column)
        upper_layout.setContentsMargins(0, 0, 0, 0) 

        right_column.setContentsMargins(10, 0, 10, 0)  # left, top, right, bottom
        left_column.setContentsMargins(10, 10, 10, 0) 

        # Add upper layout to the main layout
        main_layout.addLayout(upper_layout)


        # Canvas for plotting the scene
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.installEventFilter(self)
        self.ax.set_xlim(2.5, -2.5)
        self.ax.set_ylim(-2.5, 2.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.canvas.mpl_connect("button_press_event", self.handle_click)
        main_layout.addWidget(self.canvas)

        # Accept/Cancel buttons (aligned right, fixed width, spaced)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()  # Push buttons to the right

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setFixedWidth(100)

        self.accept_btn = QPushButton("Modify" if self.edit_mode else "Create")
        self.accept_btn.setFixedWidth(100)

        self.cancel_btn.clicked.connect(self.reject)
        self.accept_btn.clicked.connect(self.accept)

        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addSpacing(10)  # Space between buttons
        btn_layout.addWidget(self.accept_btn)

        main_layout.addLayout(btn_layout)

        self.plot_scene()

        self.installEventFilter(self)
        self.name_input.setFocus()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            global_pos = event.globalPos()

            # Forzar pérdida de foco si se hace clic fuera del spinbox
            if not self.robot_orientation_input.geometry().contains(self.mapFromGlobal(global_pos)):
                self.robot_orientation_input.clearFocus()

            # Forzar pérdida de foco si se hace clic fuera del name_input
            if not self.name_input.geometry().contains(self.mapFromGlobal(global_pos)):
                self.name_input.clearFocus()

        return super().eventFilter(obj, event)


    def normalize_angle(self, theta):
        """Wrap angle to [-π, π]."""
        from math import pi
        return (theta + pi) % (2 * pi) - pi
    

    def update_robot_orientation_from_enter(self, new_theta):
        for elem in self.scene_elements:
            if elem[0] == "robot":
                elem[3] = new_theta
                break
        self.plot_scene()


    def rotate_robot(self, delta):
        for elem in self.scene_elements:
            if elem[0] == "robot":
                elem[3] = self.normalize_angle(elem[3] + delta)
                self.robot_orientation_input.setValue(elem[3])
                break
        self.plot_scene()

    def update_robot_orientation_from_input(self):
        new_theta = self.robot_orientation_input.value()
        for elem in self.scene_elements:
            if elem[0] == "robot":
                elem[3] = new_theta
                break
        self.plot_scene()


    def handle_click(self, event):
        if not event.inaxes:
            return
    
        clicked_x, clicked_y = event.xdata, event.ydata
        selected_type = self.element_type_combo.currentText()

        # Try to delete an object if clicked near one (excluding robot)
        for i, elem in enumerate(self.scene_elements):
            if elem[0] != "robot":
                ex, ey = elem[1], elem[2]
                distance = np.sqrt((clicked_x - ex) ** 2 + (clicked_y - ey) ** 2)
                if distance < 0.1:
                    del self.scene_elements[i]
                    self.plot_scene()
                    return

        self.scene_elements.append([selected_type, clicked_x, clicked_y])
        self.plot_scene()

    def plot_scene(self):
        self.ax.clear()
        self.scene_items.clear()
        self.ax.set_xlim(2.5, -2.5)
        self.ax.set_ylim(2.5, -2.5)
        self.ax.grid(True)
        

        for elem in self.scene_elements:
            if elem[0] == "robot":
                circle = plt.Circle((elem[1], elem[2]), 0.35 / 2, color='black', label='robot', zorder=4)
                self.ax.add_patch(circle)
                self.ax.arrow(elem[1], elem[2], 0.3 * np.cos(elem[3]), 0.3 * np.sin(elem[3]), head_width=0.1, color='black')
                self.scene_items.append((circle, {"type": elem[0], "x": elem[1], "y": elem[2], "theta": elem[3]}))
            elif elem[0] == "target":
                target_rings = [(0.5 / 2, 'blue'), (0.25 / 2, 'red'), (0.03 / 2, 'yellow')]
                for radius, color in target_rings:
                    circle = plt.Circle((elem[1], elem[2]), radius, color=color, fill=True, alpha=0.6)
                    self.ax.add_patch(circle)
                self.scene_items.append((circle, {"type": elem[0], "x": elem[1], "y": elem[2]}))

            elif elem[0] == "obstacle":
                circle = plt.Circle((elem[1], elem[2]), 0.25 / 2, color='gray', label='obstacle')
                self.ax.add_patch(circle)
                self.scene_items.append((circle, {"type": elem[0], "x": elem[1], "y": elem[2]}))

        self.canvas.draw()
    
    def load_scene(self, folder_name):
        """Load an existing scene from a CSV file."""
        import pandas as pd

        scene_path = os.path.join(self.base_path, "robots", self.robot_name, "scene_configs", folder_name, "scene.csv")
        if not os.path.isfile(scene_path):
            QMessageBox.warning(self, "File not found", f"Scene CSV not found: {scene_path}")
            return

        try:
            df = pd.read_csv(scene_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read scene CSV: {e}")
            return

        self.name_input.setText(folder_name)
        self.scene_elements = []

        for _, row in df.iterrows():
            t = row["type"].strip().lower()
            x = float(row["x"])
            y = float(row["y"])
            theta = float(row["theta"]) if t == "robot" and not pd.isna(row["theta"]) else None
            if t == "robot":
                self.scene_elements.append([t, x, y, theta])
                self.robot_orientation_input.setValue(theta)
            else:
                self.scene_elements.append([t, x, y])

        self.plot_scene()

    def accept(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing name", "Please provide a scene folder name.")
            return

        # Actualizar elementos desde los patches
        self.scene_elements = []
        for _, data in self.scene_items:
            elem_type = data["type"]
            x = data["x"]
            y = data["y"]
            theta = data.get("theta", "")
            self.scene_elements.append([elem_type, x, y, theta])

        scene_dir = os.path.join(self.base_path, "robots", self.robot_name, "scene_configs", name)
        os.makedirs(scene_dir, exist_ok=True)
        csv_path = os.path.join(scene_dir, "scene.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["type", "x", "y", "theta"])
            writer.writeheader()
            for elem in self.scene_elements:
                writer.writerow({
                    "type": elem[0],
                    "x": elem[1],
                    "y": elem[2],
                    "theta": elem[3] if elem[0] == "robot" else ""
    })
        self.selected_scene_folder = name
        super().accept()
        if self.edit_mode:
            QMessageBox.information(self, "Scene modified", f"Scene '{name}' has been modified successfully.")
            self.parent.logs_text.append(f"<span style='color:green;'> --- </span> Scene '{name}' has been modified successfully.")
        else:
            QMessageBox.information(self, "Scene created", f"Scene '{name}' has been created successfully.")
            self.parent.logs_text.append(f"<span style='color:green;'> --- </span> Scene '{name}' has been created successfully.")

    def get_selected_scene_folder(self):
        return getattr(self, "selected_scene_folder", None)


class NewEnvDialog(QDialog):
    """Dialog to design the observation space for a new robot environment.

    The user can add rows representing observation variables with lower/upper bounds.
    Example default rows: distance [0, 5], angle [-pi, pi].
    """

    def __init__(self, parent=None, robot_name: str = "MyRobot"):
        """Initialize dialog.

        Args:
            parent: Optional parent widget.
            robot_name: Provisional name for the new robot.
        """
        super().__init__(parent)
        self.setWindowTitle(f"New Env for '{robot_name}' - Observation Space")
        self.setModal(True) 
        self.robot_name = robot_name
        self._spec = None   # <-- store validated spec here

        layout = QVBoxLayout(self)

        # Optional flags as in your BurgerBotEnv
        self.include_time_checkbox = QCheckBox("Include time in observation space (obs_time)")
        self.include_time_checkbox.setChecked(False)

        # Table for variables
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["name", "low", "high"])
        self.table.horizontalHeader().setStretchLastSection(True)

        # Defaults similar to your example
        self.add_row("distance", "0", "5")
        self.add_row("angle", "-3.1415926535", "3.1415926535")

        # Buttons row
        row_bar = QHBoxLayout()
        add_btn = QPushButton("Add variable")
        del_btn = QPushButton("Remove selected")
        row_bar.addWidget(add_btn)
        row_bar.addWidget(del_btn)
        row_bar.addStretch()
        add_btn.clicked.connect(lambda: self.add_row("", "0", "1"))
        del_btn.clicked.connect(self.remove_selected)

        # Standard buttons (OK/Cancel) – Qt closes dialogs automatically on accept/reject
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        # Connect once, no extra slots hooked to OK
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout.addWidget(QLabel("Define observation variables and bounds:"))
        layout.addWidget(self.table)
        layout.addLayout(row_bar)
        layout.addSpacing(6)
        layout.addWidget(self.button_box)

        self.resize(620, 380)

    def add_row(self, name: str, low: str, high: str) -> None:
        """Append a new row to the table."""
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(name))
        self.table.setItem(r, 1, QTableWidgetItem(low))
        self.table.setItem(r, 2, QTableWidgetItem(high))

    def remove_selected(self) -> None:
        """Remove selected rows."""
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def get_spec(self):
        """Return the observation spec and flags.

        Returns:
            dict: {
                "include_time": bool,
                "vars": list[{"name": str, "low": float, "high": float}]
            }
        Raises:
            ValueError: If any row is invalid or bounds are inconsistent.
        """
        spec = {"include_time": self.include_time_checkbox.isChecked(), "vars": []}
        for r in range(self.table.rowCount()):
            name_item = self.table.item(r, 0)
            low_item = self.table.item(r, 1)
            high_item = self.table.item(r, 2)
            if not name_item or not low_item or not high_item:
                raise ValueError("Empty cells are not allowed.")
            name = name_item.text().strip()
            if not name:
                raise ValueError("Variable name cannot be empty.")
            try:
                low = float(low_item.text().strip())
                high = float(high_item.text().strip())
            except Exception:
                raise ValueError(f"Bounds must be numeric at row {r+1}.")
            if high <= low:
                raise ValueError(f"Upper bound must be greater than lower bound for '{name}'.")
            spec["vars"].append({"name": name, "low": low, "high": high})
        if not spec["vars"]:
            raise ValueError("You must define at least one variable.")
        return spec
    
    def result_spec(self):
        """Return the validated spec stored during accept()."""
        return self._spec

    def accept(self):
        """Validate before closing."""
        try:
            self._spec = self.get_spec()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid spec", str(exc))
            return
        super().accept() 


class EditParamsDialog(QDialog):
    '''
    Dialog window for editing existing robot training parameter files.
    Allows users to modify parameters through input widgets and save changes back to the JSON file.
    '''
    def __init__(self, base_path, filename, parent=None):
        '''
        Initializes the parameter editing dialog.
        Args:
            base_path (str): Path to the root directory of the project.
            filename (str): Name of the parameter file to edit.
            parent (QWidget, optional): Parent widget of the dialog. Defaults to None.
        '''
        super().__init__(parent)
        self.setWindowTitle("Edit Parameters File")
        self.setMinimumSize(800, 600)
        self.base_path = base_path
        self.filename = filename
        self.file_path = os.path.join(base_path, "configs", filename)

        self.form_widgets = {}  # key: (section, param) -> widget

        layout = QVBoxLayout(self)

        # Scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        # Load existing parameters
        self.load_existing_params()

        # Buttons
        button_row = QHBoxLayout()
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red;")
        layout.addWidget(self.error_label)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_changes)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        save_button.setAutoDefault(False)
        save_button.setDefault(False)
        cancel_button.setAutoDefault(False)
        cancel_button.setDefault(False)

        button_row.addStretch()
        button_row.addWidget(cancel_button)
        button_row.addWidget(save_button)
        layout.addLayout(button_row)


    def load_existing_params(self) -> None:
        '''
        Loads existing parameters from the JSON file and creates input widgets.
        '''
        if not os.path.exists(self.file_path):
            self.scroll_layout.addWidget(QLabel("❌ File not found."))
            return

        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.scroll_layout.addWidget(QLabel(f"❌ Failed to load file: {e}"))
            return

        for section, params in data.items():
            section_label = QLabel(f"<b>{section}</b>")
            self.scroll_layout.addWidget(section_label)

            grid = QGridLayout()
            row, col = 0, 0
            for key, value in params.items():
                label = QLabel(key)
                label.setToolTip(self.get_param_description(key))
                widget = self.create_input_widget(value)
                self.form_widgets[(section, key)] = widget

                grid.addWidget(label, row, col * 2)
                grid.addWidget(widget, row, col * 2 + 1)

                col += 1
                if col >= 2:
                    row += 1
                    col = 0

            self.scroll_layout.addLayout(grid)


    def create_input_widget(self, value) -> QWidget:
        '''
        Creates an appropriate input widget based on the value type.
        Args:
            value: The parameter value to determine the widget type.
        Returns:
            QWidget: The created input widget.
        '''
        if isinstance(value, bool):
            checkbox = QCheckBox()
            checkbox.setChecked(value)
            return checkbox
        else:
            line_edit = QLineEdit(str(value))
            return line_edit


    def save_changes(self) -> None:
        '''
        Gathers input values, validates them, and saves to the JSON file.
        '''
        updated_data = {}

        for (section, key), widget in self.form_widgets.items():
            if section not in updated_data:
                updated_data[section] = {}

            if isinstance(widget, QCheckBox):
                updated_data[section][key] = widget.isChecked()
            else:
                text = widget.text().strip()
                try:
                    val = json.loads(text)
                except:
                    val = text  # Deja como string si no es parseable
                updated_data[section][key] = val

        try:
            with open(self.file_path, 'w') as f:
                json.dump(updated_data, f, indent=4)
            self.accept()
        except Exception as e:
            self.error_label.setText(f"❌ Failed to save file: {e}")


    def get_param_description(self, key) -> str:
        '''
        Retrieves the description for a given parameter key.
        Args:
            key (str): The parameter key.
        Returns:
            str: The parameter description or the key itself if not found.
        '''
        return PARAM_TOOLTIPS.get(key, key)