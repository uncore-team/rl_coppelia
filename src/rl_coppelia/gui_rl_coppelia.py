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
    QHBoxLayout, QToolTip, QDialog, QGroupBox, QMessageBox, QComboBox, QDoubleSpinBox
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
        rotation_layout.setContentsMargins(0, 15, 0, 0)  # Más espacio arriba

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



class EditParamsDialog(QDialog):
    def __init__(self, base_path, filename, parent=None):
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


    def load_existing_params(self):
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

    def create_input_widget(self, value):
        if isinstance(value, bool):
            checkbox = QCheckBox()
            checkbox.setChecked(value)
            return checkbox
        else:
            line_edit = QLineEdit(str(value))
            return line_edit

    def save_changes(self):
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

    def get_param_description(self, key):
        descriptions = {
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
        return descriptions.get(key, key)
    

class ProcessThread(QThread):
    progress_signal = pyqtSignal(int)  # Signal for updating progress bar
    finished_signal = pyqtSignal()    # Signal for indicating the process has finished
    error_signal = pyqtSignal(str)    # Signal for managing errors

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.terminal_title = ""
        self.was_stopped_manually = False

    def stop(self):
        self.was_stopped_manually = True
        if hasattr(self, 'process') and self.process and self.process.poll() is None:
            self.process.terminate()

    def run(self):
        """Execute each train/test in a separate thread, so the user can run multiple processes simultanously."""
        try:
            self.was_stopped_manually = False  # Reset flag

            # Execute the command using subprocess
            self.process = subprocess.Popen(
                self.args,
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True
            )

            # Read the output of the process for updating the progress bar in real time
            for line in self.process.stdout:
                logging.debug(line.strip())  # Console debug
                if "Testing Episodes" in line:
                    try:
                        match = re.search(r"(\d+)%", line)
                        if match:
                            progress = int(match.group(1))
                            self.progress_signal.emit(progress)
                        else:
                            logging.warning(f"Could not parse progress from line: {line.strip()}")
                    except ValueError:
                        logging.warning(f"Could not parse progress from line: {line.strip()}")

                if "Training Progress" in line:
                    match = re.search(r"(\d+)%", line)
                    if match:
                        self.progress_signal.emit(int(match.group(1)))


            self.process.wait()

            # Avoid error indicators if the process was manually stopped
            if self.was_stopped_manually:
                return

            if self.process.returncode != 0:
                self.error_signal.emit("Process returned non-zero exit code.")
            else:
                self.finished_signal.emit()

        except Exception as e:
            self.error_signal.emit(str(e))


class WelcomeScreen(QWidget):
    """Welcome screen."""

    def __init__(self, on_continue_callback):
        super().__init__()
        self.on_continue_callback = on_continue_callback
        self.init_ui()

    def init_ui(self):
        from PyQt5.QtGui import QPixmap

        main_layout = QHBoxLayout()

        # Left side: logo
        logo_label = QLabel()
        logo_path = pkg_resources.resource_filename("rl_coppelia", "assets/uncore.png")
        logo_pixmap = QPixmap(logo_path)
        logo_pixmap = logo_pixmap.scaledToHeight(180, Qt.SmoothTransformation)
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignCenter)

        logo_layout = QVBoxLayout()
        logo_layout.addStretch()
        logo_layout.addWidget(logo_label)
        logo_layout.addStretch()

        logo_container = QWidget()
        logo_container.setLayout(logo_layout)
        main_layout.addWidget(logo_container, stretch=1)

        # Right side: text and button
        right_layout = QVBoxLayout()

        title = QLabel("<h1>Welcome to RL Coppelia GUI</h1>")
        title.setAlignment(Qt.AlignLeft)
        right_layout.addWidget(title)

        team_text = QLabel("""
            <p>
            Created by <b>UnCoRE: UNexpected COgnitive, Robotics & Education Team</b>, a enthusiastic bunch of 
            researchers and teachers working at Universidad de Málaga.
            </p>
        """)
        team_text.setWordWrap(True)
        right_layout.addWidget(team_text)

        desc = QLabel("""
            <p>
            This application helps you manage, test, train and analyze reinforcement learning experiments
            in robotic environments simulated in CoppeliaSim.
            </p>
            <p>
            Have fun!
            </p>
                      
        """)
        desc.setWordWrap(True)
        right_layout.addWidget(desc)

        license_text = QLabel("""
            <p style='font-size:10pt; color:gray;'>
            Licensed under the <b>GNU General Public License v3.0</b>
            </p>
        """)
        license_text.setWordWrap(True)
        right_layout.addWidget(license_text)

        # 🚀 Botón “Let's go”
        continue_btn = QPushButton("Let's go")
        continue_btn.setFixedSize(150, 50)
        continue_btn.clicked.connect(self.on_continue_callback)
        continue_btn.setStyleSheet("""
            QPushButton {
                background-color: #007ACC;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 10px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #005fa3;
            }
            QPushButton:pressed {
                background-color: #00487a;
            }
        """)

        # Add some extra space before the button
        right_layout.addSpacing(30)

        # Centered layout with respect the button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(continue_btn)
        btn_layout.addStretch()
        right_layout.addLayout(btn_layout)

        right_layout.addStretch()

        right_container = QWidget()
        right_container.setLayout(right_layout)
        main_layout.addWidget(right_container, stretch=2)

        self.setLayout(main_layout)


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL Coppelia Manager")
        self.setGeometry(200, 200, 500, 300)

        self.welcome_screen = WelcomeScreen(self.load_main_interface)
        self.setCentralWidget(self.welcome_screen)


        self.plot_tooltip_label = QLabel(self)
        self.plot_tooltip_label.setText(PLOT_TOOLTIPS)
        # self.plot_tooltip_label.setWindowFlags(Qt.ToolTip)
        self.plot_tooltip_label.setStyleSheet("""
            QLabel {
                background-color: #ffffe0;
                border: 1px solid gray;
                padding: 8px;
                font-size: 10pt;
            }
        """)
        self.plot_tooltip_label.hide()

        self.plot_tooltip_pinned = False

        self.last_validated_scene_path = ""

        


    def on_tab_changed(self, index):
        """Trigger refresh of file-based inputs when switching tabs."""
        current_tab = self.tabs.tabText(index)

        if current_tab == "Train":
            self.populate_input_names(self.train_params_file_input, category="params_file")

        elif current_tab == "Plot":
            robot_name = self.plot_robot_name_input.currentText()
            if robot_name and not robot_name.startswith("Select"):
                self.update_model_ids_for_selected_robot()
                self.populate_input_names(self.plot_scene_to_load_folder_input, category="scene_configs")

        elif current_tab == "Test scene":
            robot_name = self.test_scene_robot_name_input.currentText()
            if robot_name and not robot_name.startswith("Select"):
                self.update_model_ids_for_selected_robot()
                self.populate_input_names(self.test_scene_scene_to_load_folder_input, category="scene_configs")
        elif current_tab == "Auto Training":
            robot_name = self.auto_train_robot_name_input.currentText()
            if robot_name and not robot_name.startswith("Select"):
                self.populate_input_names(self.auto_train_session_name_input, category="session_folders")



    def create_info_button_with_tooltip(self, tooltip_html: str) -> QPushButton:
        """Creates an info button that displays a floating tooltip on hover or click."""
        button = QPushButton("ℹ️")
        button.setCursor(Qt.PointingHandCursor)
        button.setFixedSize(24, 24)
        button.setProperty("pinned", False)
        button.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
                font-size: 16px;
                color: black;
            }
            QPushButton:hover {
                color: #007ACC;
            }
            QPushButton:pressed {
                color: #004b8d;
            }
            QPushButton[pinned="true"] {
                color: #007ACC;
                font-weight: bold;
            }
        """)

        # Tooltip QLabel
        tooltip_label = QLabel(self)
        tooltip_label.setText(tooltip_html)
        tooltip_label.setWindowFlags(Qt.ToolTip)
        tooltip_label.setStyleSheet("""
            QLabel {
                background-color: #ffffe0;
                border: 1px solid gray;
                padding: 8px;
                font-size: 10pt;
            }
        """)
        tooltip_label.hide()

        # Estado
        tooltip_pinned = {"value": False}  # mutable para capturar dentro de lambdas

        def show_tooltip():
            pos = button.mapToGlobal(button.rect().bottomRight())
            tooltip_label.move(pos + QPoint(10, 10))
            tooltip_label.adjustSize()
            tooltip_label.show()

        def toggle_tooltip():
            if tooltip_pinned["value"]:
                tooltip_label.hide()
                tooltip_pinned["value"] = False
                button.setProperty("pinned", False)
            else:
                show_tooltip()
                tooltip_pinned["value"] = True
                button.setProperty("pinned", True)
            button.setStyle(button.style())  # force refresh

        # Conexiones
        button.clicked.connect(toggle_tooltip)
        button.installEventFilter(self)

        # Guardar para filtro
        if not hasattr(self, "_info_tooltips"):
            self._info_tooltips = {}
        self._info_tooltips[button] = (tooltip_label, tooltip_pinned)

        return button

    def eventFilter(self, obj, event):
        if hasattr(self, "_info_tooltips") and obj in self._info_tooltips:
            label, pinned = self._info_tooltips[obj]
            if event.type() == QEvent.Enter and not pinned["value"]:
                pos = obj.mapToGlobal(obj.rect().bottomRight())
                label.move(pos + QPoint(10, 10))
                label.adjustSize()
                label.show()
            elif event.type() == QEvent.Leave and not pinned["value"]:
                label.hide()
        return super().eventFilter(obj, event)



    def capture_cli_output(self, callable_fn, argv=None):
        """Run a CLI-style callable, capturing stdout, stderr and logging output.

        Args:
            callable_fn (function): A callable to run (e.g., cli.main).
            argv (list, optional): Argument list to simulate CLI input.

        Returns:
            tuple: (combined_output: str, errors: list[str], success: bool)
        """
        import io
        import logging
        import contextlib

        stdout_capture = io.StringIO()
        log_capture = io.StringIO()

        log_handler = logging.StreamHandler(log_capture)
        log_handler.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.addHandler(log_handler)

        success = True

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stdout_capture):
            try:
                callable_fn(argv)
            except Exception:
                success = False
            finally:
                logger.removeHandler(log_handler)

        combined_output = stdout_capture.getvalue() + "\n" + log_capture.getvalue()
        errors = [line for line in combined_output.splitlines() if " - ERROR - " in line]

        return combined_output.strip(), errors, success


    def refresh_active_tab(self):
        """Trigger the same updates as if the tab was changed."""
        index = self.tabs.currentIndex()
        self.on_tab_changed(index)

    
    def sync_refresh_button_height(self):
        """Set the refresh button height to match the tab bar once it's rendered."""
        if self.tabs and self.tabs.tabBar():
            tab_height = self.tabs.tabBar().sizeHint().height()
            if tab_height > 0:
                self.refresh_button.setFixedHeight(tab_height)


    def load_main_interface(self):
        """Load the main interface after the welcome screen."""
        self.resize(1000, 600)
        expanded_path = os.path.abspath(__file__)
        self.base_path = str(Path(expanded_path).parents[2])
        logging.debug(f"Base path configured as: {self.base_path}")
        self.model_name = ""
        self.robot_name = ""
        self.experiment_id = 0

        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout horizontal principal
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # === IZQUIERDA: Tabs y botón Refresh ===

        # Contenedor con layout vertical
        tabs_container = QWidget()
        tabs_layout = QVBoxLayout()
        tabs_layout.setContentsMargins(0, 0, 0, 0)
        tabs_layout.setSpacing(0)
        tabs_container.setLayout(tabs_layout)

        # Layout superior para pestañas + botón
        tabs_bar = QWidget()
        tabs_bar_layout = QHBoxLayout()
        tabs_bar_layout.setContentsMargins(0, 0, 0, 0)
        tabs_bar.setLayout(tabs_bar_layout)

        # Botón Refresh
        refresh_icon_path = pkg_resources.resource_filename("rl_coppelia", "assets/refresh_icon.png")
        refresh_icon = QIcon(QPixmap(refresh_icon_path))

        self.refresh_button = QPushButton()
        self.refresh_button.setIcon(refresh_icon)
        self.refresh_button.setIconSize(QSize(20, 20))  # Ajusta tamaño si lo necesitas
        self.refresh_button.setFlat(True)
        self.refresh_button.setToolTip("Refresh inputs from disk")
        self.refresh_button.clicked.connect(self.refresh_active_tab)

        # Estilo plano sin borde ni fondo
        self.refresh_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                margin-right: 4px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
                border-radius: 4px;
            }
        """)

        self.refresh_button.clicked.connect(self.refresh_active_tab)

        # QTabWidget
        self.tabs = QTabWidget()
        self.tabs.setCornerWidget(self.refresh_button, Qt.TopRightCorner)

        tabs_bar_layout.addWidget(self.tabs, alignment=Qt.AlignVCenter)
        tabs_bar_layout.addStretch()

        
        tabs_bar_layout.addWidget(self.refresh_button, alignment=Qt.AlignVCenter)

        # Añadir la barra al layout vertical
        tabs_layout.addWidget(tabs_bar)
        tabs_layout.addWidget(self.tabs)

        # Añadir tabs al layout principal
        main_layout.addWidget(tabs_container, stretch=3)

        # === DERECHA: Panel lateral ===
        side_panel_widget = QWidget()
        side_panel_widget.setLayout(self.create_side_panel())
        main_layout.addWidget(side_panel_widget, stretch=1)

        # === Añadir pestañas ===
        self.tabs.addTab(self.create_train_tab(), "Train")
        self.tabs.addTab(self.create_test_tab(), "Test")
        self.tabs.addTab(self.create_test_scene_tab(), "Test scene")
        self.tabs.addTab(self.create_plot_tab(), "Plot")
        self.tabs.addTab(self.create_auto_training_tab(), "Auto Training")
        # self.tabs.addTab(self.create_auto_testing_tab(), "Auto Testing")
        # self.tabs.addTab(self.create_retrain_tab(), "Retrain")
        self.tabs.addTab(self.create_manage_tab(), "Manage")

        # === Conexión cambio de pestaña ===
        self.tabs.currentChanged.connect(self.on_tab_changed)

        # === Ajustar altura del botón tras renderizado ===
        QTimer.singleShot(100, self.sync_refresh_button_height)



    def create_styled_button(self, text: str, on_click: callable) -> QPushButton:
        """Create a consistently styled action button."""
        button = QPushButton(text)
        button.setFixedSize(220, 50)
        button.setStyleSheet("""
            QPushButton {
                background-color: #007ACC;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 10px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #005fa3;
            }
            QPushButton:pressed {
                background-color: #00487a;
            }
        """)
        button.clicked.connect(on_click)
        return button


    def update_processes_label(self):
        """Update text of the proccesses box according to the amount of them."""
        count = self.processes_container.count()
        if count == 0:
            self.processes_label.setText("No processes yet")
        else:
            self.processes_label.setText("Current processes:")


    def clear_logs (self):
        """Clear the logs text area."""
        self.logs_text.clear()


    def create_side_panel(self):
        """Creates the side panel with scrollable logs and a scrollable process list."""
        from PyQt5.QtWidgets import QScrollArea

        layout = QVBoxLayout()

        # Logs header with button
        logs_header = QWidget()
        logs_layout = QHBoxLayout(logs_header)
        logs_layout.setContentsMargins(0, 0, 0, 0)

        logs_title = QLabel("Logs")
        logs_title.setStyleSheet("font-weight: bold;")

        clear_logs_button = QPushButton("Clean logs")
        clear_logs_button.setFixedHeight(22)
        clear_logs_button.setStyleSheet("padding: 2px 6px;")
        clear_logs_button.setToolTip("Click to remove all log messages")
        clear_logs_button.clicked.connect(self.clear_logs)

        logs_layout.addWidget(logs_title)
        logs_layout.addStretch()
        logs_layout.addWidget(clear_logs_button)

        layout.addWidget(logs_header)

        # Logs area with scroll
        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)

        logs_scroll = QScrollArea()
        logs_scroll.setWidgetResizable(True)
        logs_scroll.setWidget(self.logs_text)
        logs_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        layout.addWidget(logs_scroll, stretch=1)

        # Processes section
        self.processes_label = QLabel("No processes yet")
        layout.addWidget(self.processes_label)

        process_scroll_content = QWidget()
        self.processes_container = QVBoxLayout()
        process_scroll_content.setLayout(self.processes_container)

        process_scroll = QScrollArea()
        process_scroll.setWidgetResizable(True)
        process_scroll.setWidget(process_scroll_content)

        layout.addWidget(process_scroll, stretch=1)

        return layout


    def update_progress_bar(self, value):
        """Updates the progress bar with a given value."""
        self.progress_bar.setValue(value)


    def update_test_inputs_when_model_name_introduced(self):
        """Update the robot name based on the selected model name."""
        self.model_name = self.test_model_name_input.text()
        if self.model_name:
            # Extract the robot name from the model name
            parts = self.model_name.split("/")
            parts = parts[-1].split("_")  # Split by "_model_" to get the robot name
            self.robot_name = parts[0] 
            self.experiment_id = parts[2]

            if len(parts) > 1:
                # Write robot name to the input field
                robot_name = parts[0]  # The part before "_model_"
                self.test_robot_name_input.setText(robot_name)

                # Construct the scene file path
                scene_path = os.path.join(self.base_path, "scenes", f"{parts[0]}_scene.ttt")
                # Check if the scene file exists
                if not os.path.exists(scene_path):
                    warning_message = f"WARNING: {scene_path} does not exist. Please check the model name."
                    logging.warning(warning_message)
                    # self.test_scene_path_input.setText(warning_message)
                    self.logs_text.append(f"<span style='color:orange;'>{warning_message}</span>")
                else:
                    message = f"Scene file {scene_path} found for robot {robot_name}."
                    logging.info(message)
                    self.test_scene_path_input.setText(scene_path)
                    self.logs_text.append(f"<span style='color:green;'> --- </span> {message}")

                # Construct the params file path
                params_file = utils.find_params_file(self.base_path, robot_name, self.experiment_id)
                if params_file:
                    params_file_path = os.path.join(self.base_path, "robots", robot_name, "parameters_used", params_file)
                    # Check if the params file exists
                    if not os.path.exists(params_file_path):
                        warning_message = f"WARNING: {params_file_path} does not exist. Please check the model name."
                        logging.warning(warning_message)
                        # self.test_params_file_input.setText(warning_message)
                        self.logs_text.append(f"<span style='color:orange;'>{warning_message}</span>")
                    else:
                        message = f"Params file {params_file_path} found for robot {robot_name}."
                        logging.info(message)
                        self.test_params_file_input.setText(params_file_path)
                        self.logs_text.append(f"<span style='color:green;'> --- </span> {message}")
                else:
                    warning_message = f"WARNING: No params file found for experiment ID '{robot_name}_model_{self.experiment_id}'."
                    logging.warning(warning_message)
                    self.logs_text.append(f"<span style='color:orange;'>{warning_message}</span>")
                
            else:
                self.test_robot_name_input.setText("")  # Clear if format is invalid
                self.test_scene_path_input.setText("")
                self.test_params_file_input.setText("")
        else:
            self.test_robot_name_input.setText("")  # Clear if no model name is provided
            self.test_scene_path_input.setText("")
            self.test_params_file_input.setText("")


    def get_rl_coppelia_path_from_bashrc(self):
        """Retrieve the rl_coppelia path from ~/.bashrc."""
        bashrc_path = os.path.expanduser("~/.bashrc")
        if not os.path.exists(bashrc_path):
            return None
        
        try:
            with open(bashrc_path, "r") as bashrc:
                content = bashrc.read()
                
            # Search in PYTHONPATH
            pythonpath_matches = re.findall(r'(?:export\s+)?PYTHONPATH[^=]*=(.+)', content)
            for match in pythonpath_matches:
                # Clean the result (remove spaces and "")
                clean_match = match.strip().strip('"').strip("'")
                paths = clean_match.split(":")
                for path in paths:
                    path = path.strip()
                    if path and "rl_coppelia" in path:
                        # Expand environment variables (if so)
                        expanded_path = os.path.expandvars(path)
                        if os.path.exists(expanded_path):
                            base_path = str(Path(expanded_path).parents[1])  # Get the parent directory of rl_coppelia
                            if base_path != self.base_path:
                                self.base_path = base_path  # Update the base path
                                logging.warning(f"Found rl_coppelia path in PYTHONPATH: {base_path}, but it does not match the expected base path: {self.base_path}")
                            return base_path  
            
            # Search in PATH it the previous search was unsuccessful
            path_matches = re.findall(r'(?:export\s+)?PATH[^=]*=(.+)', content)
            for match in path_matches:
                clean_match = match.strip().strip('"').strip("'")
                paths = clean_match.split(":")
                for path in paths:
                    path = path.strip()
                    if path and "rl_coppelia" in path:
                        expanded_path = os.path.expandvars(path)
                        if os.path.exists(expanded_path):
                            base_path = str(Path(expanded_path).parents[1])  # Get the parent directory of rl_coppelia
                            if base_path != self.base_path:
                                self.base_path = base_path
                                logging.warning(f"Found rl_coppelia path in PYTHONPATH: {base_path}, but it does not match the expected base path: {self.base_path}")
                            return base_path
                            
        except Exception as e:
            logging.error(f"Error reading .bashrc: {e}")
            
        return None
    

    def browse_zip_file(self):
        """Open a file dialog to select a ZIP file, starting in the rl_coppelia directory."""
        rl_coppelia_path = self.get_rl_coppelia_path_from_bashrc()
        
        # If rl_coppelia path was found, then it will be used as main directory for searching files
        if rl_coppelia_path and os.path.exists(rl_coppelia_path):
            start_path = rl_coppelia_path
            logging.info(f"Starting file dialog in rl_coppelia directory: {start_path}")
        else:
            start_path = os.path.expanduser("~")
            logging.warning(f"rl_coppelia path not found, starting in home directory: {start_path}")
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select ZIP File", 
            start_path, 
            "ZIP Files (*.zip)"
        )
        
        if file_path:
            self.test_model_name_input.setText(file_path)

    def remove_zip_extension(self, file_path):
        """Remove the .zip extension from the file name if it exists."""
        base_name, extension = os.path.splitext(file_path)
        if extension.lower() == ".zip":
            return base_name
        return file_path

    def update_train_scene_path_from_robot(self):
        """Auto-fill the scene path and validate it when a robot name is entered."""
        if self.train_robot_name_combo.currentText() == "Create a new one!":
            robot_name = self.train_new_robot_name_input.text().strip()
        else:
            robot_name = self.train_robot_name_combo.currentText()

        if not robot_name:
            self.train_scene_path_input.clear()
            self.train_scene_path_input.setStyleSheet("")
            return

        scene_path = os.path.join(self.base_path, "scenes", f"{robot_name}_scene.ttt")
        self.train_scene_path_input.setText(scene_path)
        self.validate_scene_path()


    def validate_scene_path(self):
        """Check if the current scene path exists, update style and logs only if changed."""
        scene_path = self.train_scene_path_input.text().strip()

        if scene_path == self.last_validated_scene_path:
            return  # Avoid duplicated logs

        self.last_validated_scene_path = scene_path  # Update last validated path

        if not scene_path:
            self.train_scene_path_input.setStyleSheet("")
            self.train_scene_path_input.setToolTip("")
            return

        if not os.path.isfile(scene_path):
            warning = f" --- ⚠️ Scene file not found: {scene_path}"
            logging.warning(warning)
            self.logs_text.append(f"<span style='color:orange;'>{warning}</span>")
            self.train_scene_path_input.setStyleSheet("background-color: #fff8c4;")
            self.train_scene_path_input.setToolTip("Scene file does not exist.")
        else:
            log_text = f" --- Scene file found: {scene_path}"
            logging.debug(log_text)
            self.logs_text.append(f"<span style='color:green;'>{log_text}</span>")
            self.train_scene_path_input.setStyleSheet("")
            self.train_scene_path_input.setToolTip("")


    def handle_train_params_selection(self, text):
        if text == "Manual parameters":
            dialog = ManualParamsDialog(self.base_path, self.train_robot_name_combo.currentText(), self)
            dialog.params_saved.connect(self.on_manual_params_saved)
            result = dialog.exec_()
            if result != QDialog.Accepted:
                # If the dialog was cancelled, reset the combo box to the first item
                self.train_params_file_input.setCurrentIndex(0)

        if text and text not in ["Select a parameters file...", "Manual parameters"]:
            self.edit_params_button.setVisible(True)
        else:
            self.edit_params_button.setVisible(False)

    def on_manual_params_saved(self, filename):
        self.populate_input_names(self.train_params_file_input, category="params_file")

        # Buscar y seleccionar el índice del nuevo archivo
        for i in range(self.train_params_file_input.count()):
            item_text = self.train_params_file_input.itemText(i)
            if filename in item_text:
                self.train_params_file_input.setCurrentIndex(i)

                # Actualizar visibilidad del botón manualmente
                if filename and filename not in ["Select a parameters file...", "Manual parameters"]:
                    self.edit_params_button.setVisible(True)
                else:
                    self.edit_params_button.setVisible(False)
                break

        self.logs_text.append(
            f"<span style='color:green;'> --- Created new parameters file: <b>{filename}</b></span>"
        )

    def open_edit_params_dialog(self):
        text = self.train_params_file_input.currentText()
        if text and text not in ["Select a parameters file...", "Manual parameters"]:
            filename = text.split()[0]  # Ignore extended info

            dialog = EditParamsDialog(self.base_path, filename, self)
            result = dialog.exec_()

            if result == QDialog.Accepted:
                # Show success log
                self.logs_text.append(
                    f"<span style='color:green;'> --- ✅ Parameters updated successfully in <b>{filename}</b>.</span>"
                )

                # Refresh the combo box
                self.populate_input_names(self.train_params_file_input, category="params_file")

                # Select the modified file in the combo box
                for i in range(self.train_params_file_input.count()):
                    if filename in self.train_params_file_input.itemText(i):
                        self.train_params_file_input.setCurrentIndex(i)
                        break


    def create_train_tab(self):
        """Tab for training configuration."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Form for training parameters
        form = QFormLayout()

        # Robot selection
        self.train_robot_name_combo = QComboBox()
        self.train_robot_name_combo.addItem("Select a robot...")
        self.train_robot_name_combo.model().item(0).setEnabled(False)
        self.populate_input_names(self.train_robot_name_combo, category="robot")
        self.train_robot_name_combo.addItem("Create a new one!")
        self.train_robot_name_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        

        # Label and field for new name
        self.train_new_robot_label = QLabel("Introduce name:")
        self.train_new_robot_name_input = QLineEdit()
        self.train_new_robot_label.hide()
        self.train_new_robot_name_input.hide()

        self.train_new_robot_label.setMaximumWidth(100)
        self.train_new_robot_name_input.setMaximumWidth(200)
        
        self.train_robot_name_combo.currentTextChanged.connect(self.handle_train_robot_selection)
        # self.train_robot_name_combo.currentTextChanged.connect(self.update_train_scene_path_from_robot)
        self.train_new_robot_name_input.editingFinished.connect(self.update_train_scene_path_from_robot)


        # Horizontal combined widget
        robot_row = QWidget()
        robot_row_layout = QHBoxLayout(robot_row)
        robot_row_layout.setContentsMargins(0, 0, 0, 0)
        robot_row_layout.setSpacing(10)

        robot_row_layout.addWidget(self.train_robot_name_combo)
        robot_row_layout.addWidget(self.train_new_robot_label)
        robot_row_layout.addWidget(self.train_new_robot_name_input)
        robot_row_layout.addStretch()  # force alignement

        self.train_scene_path_input = QLineEdit()
        self.train_scene_path_input.setPlaceholderText("Enter scene path (optional)")
        self.train_scene_path_input.editingFinished.connect(self.validate_scene_path)

        self.train_params_file_input = QComboBox()
        self.train_params_file_input.addItem("Select a configuration file...")
        self.train_params_file_input.model().item(0).setEnabled(False)
        
        self.train_params_file_input.addItem("Manual parameters")
        self.populate_input_names(self.train_params_file_input, category="params_file")
        self.train_params_file_input.currentTextChanged.connect(self.handle_train_params_selection)

        params_file_row = QWidget()
        params_layout = QHBoxLayout(params_file_row)
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.addWidget(self.train_params_file_input)

        # Gear button
        self.edit_params_button = QPushButton()
        self.edit_params_button.setIcon(QIcon(pkg_resources.resource_filename("rl_coppelia", "assets/gear_icon.png")))
        self.edit_params_button.setFixedSize(24, 24)
        self.edit_params_button.setToolTip("Edit selected parameter file")
        self.edit_params_button.setVisible(False)
        self.edit_params_button.clicked.connect(self.open_edit_params_dialog)
        params_layout.addWidget(self.edit_params_button)

        # Disable parallel mode (optional)
        self.train_dis_parallel_mode_checkbox = QCheckBox("Disable Parallel Mode")

        # Disable GUI (optional)
        self.train_no_gui_checkbox = QCheckBox("Disable GUI")

        self.train_verbose_input = QSpinBox()
        self.train_verbose_input.setRange(-1, 4)
        self.train_verbose_input.setValue(3)

        form.addRow("Robot Name (required):", robot_row)
        form.addRow("Scene Path (optional):", self.train_scene_path_input)
        form.addRow("Params File:", params_file_row)
        form.addRow("Options: ", self.train_dis_parallel_mode_checkbox)
        form.addRow("", self.train_no_gui_checkbox)
        form.addRow("Verbose Level (default: 1):", self.train_verbose_input)

        # Buttons
        train_start_button = self.create_styled_button("Start Training", self.start_train)

        # Layout centrado vertical y horizontal
        button_layout = QVBoxLayout()
        button_layout.addStretch()

        centered_h = QHBoxLayout()
        centered_h.addStretch()
        centered_h.addWidget(train_start_button)
        centered_h.addStretch()

        button_layout.addLayout(centered_h)
        button_layout.addStretch()

        layout.addLayout(form)
        layout.addLayout(button_layout)

        tab.setLayout(layout)
        return tab
    
    def handle_train_robot_selection(self, text):
        is_custom = text == "Create a new one!"
        self.train_new_robot_label.setVisible(is_custom)
        self.train_new_robot_name_input.setVisible(is_custom)

        if not is_custom:
            self.update_train_scene_path_from_robot()  



    def create_test_tab(self):
        """Tab for testing configuration."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Form for testing parameters
        form = QFormLayout()

        # Model name (required)
        self.test_model_name_input = QLineEdit()
        self.test_model_name_input.setPlaceholderText("Select a ZIP file...")
        self.browse_zip_button = QPushButton("Browse ZIP")
        self.browse_zip_button.setFixedHeight(24)
        self.browse_zip_button.setStyleSheet("padding: 2px 8px; font-size: 10pt;")
        self.browse_zip_button.clicked.connect(self.browse_zip_file)
        
        self.test_model_name_input.textChanged.connect(self.update_test_inputs_when_model_name_introduced)  # Connect text change to update_robot_name


        # Robot name (optional)
        self.test_robot_name_input = QLineEdit()
        self.test_robot_name_input.setPlaceholderText("Enter robot name (default: burgerBot)")

        # Scene path (optional)
        self.test_scene_path_input = QLineEdit()
        self.test_scene_path_input.setPlaceholderText("Enter scene path (optional)")

        # Save scene (optional)
        self.save_scene_checkbox = QCheckBox("Save Scene")

        # Save trajectory (optional)
        self.save_traj_checkbox = QCheckBox("Save Trajectory")

        # Disable parallel mode (optional)
        self.dis_parallel_mode_checkbox = QCheckBox("Disable Parallel Mode")

        # Disable GUI (optional)
        self.no_gui_checkbox = QCheckBox("Disable GUI")

        # Params file (optional)
        self.test_params_file_input = QLineEdit()
        self.test_params_file_input.setPlaceholderText("Enter path to params file (optional)")

        # Iterations (optional, default: 50)
        self.test_iterations_input = QSpinBox()
        self.test_iterations_input.setRange(1, 1000)
        self.test_iterations_input.setValue(50)

        # Verbose level (optional, default: 3)
        self.verbose_input = QSpinBox()
        self.verbose_input.setRange(-1, 4)
        self.verbose_input.setValue(3)

       

        

        # Add fields to the form
        zip_row = QWidget()
        zip_layout = QHBoxLayout()
        zip_layout.setContentsMargins(0, 0, 0, 0)
        zip_row.setLayout(zip_layout)

        zip_layout.addWidget(self.test_model_name_input)
        zip_layout.addWidget(self.browse_zip_button)

        self.browse_zip_button.setFixedHeight(28)
        self.browse_zip_button.setFixedWidth(100)
        self.browse_zip_button.setStyleSheet("padding: 2px 8px; font-size: 10pt;")

        form.addRow("Model ZIP File (required):", zip_row)
        form.addRow("Robot Name (optional):", self.test_robot_name_input)
        form.addRow("Scene Path (optional):", self.test_scene_path_input)
        form.addRow("Options: ", self.save_scene_checkbox)
        form.addRow("", self.save_traj_checkbox)
        form.addRow("", self.dis_parallel_mode_checkbox)
        form.addRow("", self.no_gui_checkbox)
        form.addRow("Params File (optional):", self.test_params_file_input)
        form.addRow("Iterations (default: 50):", self.test_iterations_input)
        form.addRow("Verbose Level (default: 1):", self.verbose_input)

        # Buttons
        test_button = self.create_styled_button("Start Testing", self.start_testing)

        # Layout centrado vertical y horizontal
        button_layout = QVBoxLayout()
        button_layout.addStretch()

        centered_h = QHBoxLayout()
        centered_h.addStretch()
        centered_h.addWidget(test_button)
        centered_h.addStretch()

        button_layout.addLayout(centered_h)
        button_layout.addStretch()

        layout.addLayout(form)
        layout.addLayout(button_layout)

        tab.setLayout(layout)
        return tab
    

    def handle_test_scene_robot_selection(self):
        """Update model IDs and scene folders when a robot is selected in Test Scene tab."""
        if self.robot_name and not self.robot_name.startswith("Select"):
            self.populate_input_names(self.test_scene_scene_to_load_folder_input, category="scene_configs")

    def handle_auto_train_robot_selection(self):
        """Update session folders when a robot is selected in Auto train tab."""
        self.robot_name = self.auto_train_robot_name_input.currentText()
        if self.robot_name and not self.robot_name.startswith("Select"):
            self.populate_input_names(self.auto_train_session_name_input, category="session_folders")


    def handle_scene_folder_change(self, folder_name):
        """
        Called when the user selects a scene folder. Loads element info from CSV.
        """
        if folder_name == "Custom your scene":
            robot_name = self.test_scene_robot_name_input.currentText().strip()
            if not robot_name or robot_name.startswith("Select"):
                self.logs_text.append("❌ Please select a robot before creating a custom scene.")
                return

            dialog = CustomSceneDialog(self.base_path, robot_name, self)
            if dialog.exec_() == QDialog.Accepted:
                created_folder = dialog.selected_scene_folder
                self.populate_input_names(self.test_scene_scene_to_load_folder_input, "scene_configs")
                index = self.test_scene_scene_to_load_folder_input.findText(created_folder)
                if index >= 0:
                    self.test_scene_scene_to_load_folder_input.setCurrentIndex(index)
            return

        if folder_name in ["Select a scene folder to load...", "Custom scene", "Scene configs directory not found", "No scene configs found"] or not folder_name: 
            self.test_scene_scene_info_label.hide()
            self.test_scene_view_scene_button.hide()
            self.test_scene_scene_info_row.hide()
            self.test_scene_edit_scene_button.hide()
            self.test_scene_delete_scene_button.hide()

            return

        try:
            scene_dir = os.path.join(self.base_path, "robots", self.robot_name, "scene_configs", folder_name)
            csv_file = next(
                (os.path.join(scene_dir, f) for f in os.listdir(scene_dir) if "scene" in f.lower() and f.endswith(".csv")),
                None
            )
            if not csv_file:
                self.test_scene_scene_info_label.setText("❌ No CSV found.")
                self.test_scene_scene_info_label.show()
                self.test_scene_view_scene_button.hide()
                return

            df = pd.read_csv(csv_file)
            num_targets = (df["type"].str.lower() == "target").sum()
            num_obstacles = (df["type"].str.lower() == "obstacle").sum()
            self.test_scene_scene_info_label.setText(f"Scene contains: {num_targets} targets, {num_obstacles} obstacles")
            self.test_scene_scene_info_label.show()
            self.test_scene_view_scene_button.show()
            self.test_scene_scene_info_row.show()

            self.current_scene_csv_path = csv_file  # Save for preview

            trajs_path = os.path.join(scene_dir, "trajs")

            if os.path.isdir(trajs_path) and os.listdir(trajs_path):
                warning = f"⚠️ {folder_name} folder already contains a 'trajs' directory with data. The test will overwrite existing trajectories."
                logging.warning(warning)
                self.logs_text.append(f"<span style='color:orange;'> --- {warning}</span>")
            else:
                message = f"✅ {folder_name} is ready for testing."
                logging.info(message)
                self.logs_text.append(f"<span style='color:green;'> --- </span> {message}")


        except Exception as e:
            logging.warning(f"Error loading scene info: {e}")
            self.test_scene_scene_info_label.setText("❌ Error loading scene data")
            self.test_scene_scene_info_label.show()
            self.test_scene_view_scene_button.hide()

        # Show/hide edit and delete buttons
        if folder_name and not folder_name.startswith("Select") and folder_name != "Custom scene":
            self.test_scene_edit_scene_button.setVisible(True)
            self.test_scene_delete_scene_button.setVisible(True)
        else:
            self.test_scene_edit_scene_button.setVisible(False)
            self.test_scene_delete_scene_button.setVisible(False)


    def handle_edit_scene(self):
        '''
        Edit the selected scene folder using the CustomSceneDialog.
        '''
        folder_name = self.test_scene_scene_to_load_folder_input.currentText().strip()
        if not folder_name or folder_name in ["Select a scene folder to load...", "Custom scene", "Scene configs directory not found", "No scene configs found"]:
            return

        dialog = CustomSceneDialog(self.base_path, self.robot_name, self, edit_mode=True)
        dialog.load_scene(folder_name)  
        if dialog.exec_() == QDialog.Accepted:
            self.populate_input_names(self.test_scene_scene_to_load_folder_input, "scene_configs")
            idx = self.test_scene_scene_to_load_folder_input.findText(folder_name)
            if idx >= 0:
                self.test_scene_scene_to_load_folder_input.setCurrentIndex(idx)

    def handle_delete_scene(self):
        '''
        Delete the selected scene folder after user confirmation.
        '''
        folder_name = self.test_scene_scene_to_load_folder_input.currentText().strip()
        if not folder_name or folder_name in ["Select a scene folder to load...", "Custom scene"]:
            return

        reply = QMessageBox.question(self, "Delete Scene", f"Are you sure you want to delete scene '{folder_name}'?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            scene_path = os.path.join(self.base_path, "robots", self.robot_name, "scene_configs", folder_name)
            try:
                shutil.rmtree(scene_path)
                self.populate_input_names(self.test_scene_scene_to_load_folder_input, "scene_configs")
                self.test_scene_scene_to_load_folder_input.setCurrentIndex(0)
                message = f"Scene folder '{folder_name}' deleted successfully."
                logging.info(message)
                self.logs_text.append(f"<span style='color:green;'> --- {message}</span>")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not delete scene: {e}")

    
    def handle_show_scene_preview(self):
        """Display the scene image when the user clicks the eye button."""
        if hasattr(self, "current_scene_csv_path") and os.path.isfile(self.current_scene_csv_path):
            self.show_scene_preview_dialog(self.current_scene_csv_path)



    def plot_scene(self, csv_file):
        """
        Generate a scene visualization from a CSV and return the image path.

        Args:
            csv_file (str): Path to the scene CSV file.

        Returns:
            str: Path to the saved PNG image.
        """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import os

        df = pd.read_csv(csv_file)

        fig, ax = plt.subplots(figsize=(6, 6))
        robot = df[df["type"].str.lower() == "robot"]
        targets = df[df["type"].str.lower() == "target"]
        obstacles = df[df["type"].str.lower() == "obstacle"]

        # Obstacles
        for i, row in obstacles.iterrows():
            circle = plt.Circle((row["x"], row["y"]), 0.25 / 2, color='gray', label='Obstacle')
            ax.add_patch(circle)


        # Targets
        for label_idx, (_, row) in enumerate(targets.iterrows()):
            x, y = row["x"], row["y"]
            idx = chr(65 + label_idx)  # A, B, C...
            ax.add_patch(Circle((x, y), 0.25, color="blue", alpha=0.3))   # Outer circle
            ax.add_patch(Circle((x, y), 0.125, color="red", alpha=0.5))   # Middle circle
            ax.add_patch(Circle((x, y), 0.015, color="yellow", alpha=0.8))  # Inner circle
            ax.text(x, y - 0.1, idx, fontsize=14, fontweight="bold", ha="center")

        # Robot
        for _, row in robot.iterrows():
            circle = plt.Circle((row["x"], row["y"]), 0.35 / 2, color='black', label='Robot', zorder=4)
            ax.add_patch(circle)

            # Indicate orientation using a triangle
            if 'theta' in row:
                theta = row['theta']
                # Triangle dimensions
                front_length = 0.15
                side_offset = 0.08

                # Front point
                front = (row["x"] + front_length * np.cos(theta), row["y"] + front_length * np.sin(theta))
                # Side points
                left = (row["x"] + side_offset * np.cos(theta + 2.5), row["y"] + side_offset * np.sin(theta + 2.5))
                right = (row["x"] + side_offset * np.cos(theta - 2.5), row["y"] + side_offset * np.sin(theta - 2.5))

                triangle = plt.Polygon([front, left, right], color='white', zorder=4)
                ax.add_patch(triangle)
          
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        # Removed duplicated labels
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), 
                unique.keys(), 
                loc="upper right", 
                labelspacing=1.2
                )
        ax.set_xlim(2.5, -2.5)
        ax.set_ylim(2.5, -2.5)
        plt.tight_layout()

        output_path = os.path.join("/tmp", "scene_preview.png")
        plt.savefig(output_path)
        plt.close()
        return output_path

    def show_scene_preview_dialog(self, csv_path):
        """
        Show a QDialog with the generated scene image.

        Args:
            csv_path (str): Path to the CSV scene file.
        """
        try:
            from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout
            from PyQt5.QtGui import QPixmap

            img_path = self.plot_scene(csv_path)
            dialog = QDialog(self)
            dialog.setWindowTitle("Scene Preview")
            layout = QVBoxLayout()
            label = QLabel()
            label.setPixmap(QPixmap(img_path).scaledToWidth(600, Qt.SmoothTransformation))
            layout.addWidget(label)
            dialog.setLayout(layout)
            dialog.exec_()
        except Exception as e:
            logging.error(f"Could not show scene preview: {e}")
            self.logs_text.append(f"<span style='color:red;'>⚠️ Failed to show scene preview: {e}</span>")



    def create_test_scene_tab(self):
        """Tab for testing multiple models over a predefined scene (Test Scene mode)."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Form layout
        form = QFormLayout()

        # Robot name (combo)
        self.test_scene_robot_name_input = QComboBox()
        self.test_scene_robot_name_input.addItem("Select a robot...")
        self.test_scene_robot_name_input.model().item(0).setEnabled(False)
        self.populate_input_names(self.test_scene_robot_name_input, category="robot")
        self.test_scene_robot_name_input.currentIndexChanged.connect(self.update_model_ids_for_selected_robot)

        # self.test_scene_robot_name_input.currentTextChanged.connect(self.handle_test_scene_robot_selection)
        form.addRow("Robot Name (required):", self.test_scene_robot_name_input)

        # Model IDs (checkbox list)
        self.test_scene_model_ids_input = QListWidget()
        self.test_scene_model_ids_input.setFixedHeight(200)
        self.test_scene_model_ids_input.setSelectionMode(QAbstractItemView.NoSelection)
        form.addRow("Model IDs (required):", self.test_scene_model_ids_input)

        # Scene folder
        self.test_scene_scene_to_load_folder_input = QComboBox()
        self.test_scene_scene_to_load_folder_input.currentTextChanged.connect(self.handle_scene_folder_change) 
        
        # Botones de acción junto al combo
        self.test_scene_edit_scene_button = QPushButton()
        self.test_scene_edit_scene_button.setIcon(QIcon(pkg_resources.resource_filename("rl_coppelia", "assets/edit_icon.png")))
        self.test_scene_edit_scene_button.setToolTip("Edit selected scene")
        self.test_scene_edit_scene_button.setFixedSize(24, 24)
        self.test_scene_edit_scene_button.setVisible(False)
        self.test_scene_edit_scene_button.clicked.connect(self.handle_edit_scene)

        self.test_scene_delete_scene_button = QPushButton()
        self.test_scene_delete_scene_button.setIcon(QIcon(pkg_resources.resource_filename("rl_coppelia", "assets/delete_icon.png")))
        self.test_scene_delete_scene_button.setToolTip("Delete selected scene")
        self.test_scene_delete_scene_button.setFixedSize(24, 24)
        self.test_scene_delete_scene_button.setVisible(False)
        self.test_scene_delete_scene_button.clicked.connect(self.handle_delete_scene)

        # Layout horizontal para la fila
        self.test_scene_scene_folder_row = QHBoxLayout()
        self.test_scene_scene_folder_row.addWidget(self.test_scene_scene_to_load_folder_input)
        self.test_scene_scene_folder_row.addWidget(self.test_scene_edit_scene_button)
        self.test_scene_scene_folder_row.addWidget(self.test_scene_delete_scene_button)

        form.addRow("Scene Folder:", self.test_scene_scene_folder_row)

        # Label for scene summary
        self.test_scene_scene_info_label = QLabel()
        self.test_scene_scene_info_label.hide()  # Hidden by default

        # Button to show scene preview
        self.test_scene_view_scene_button = QPushButton("Check scene!")
        self.test_scene_view_scene_button.setToolTip("Show scene preview")
        self.test_scene_view_scene_button.clicked.connect(self.handle_show_scene_preview)
        self.test_scene_view_scene_button.hide()

        # Fila combinada
        self.test_scene_scene_info_row = QWidget()
        self.test_scene_scene_info_layout = QHBoxLayout()
        self.test_scene_scene_info_layout.setContentsMargins(0, 0, 0, 0)
        self.test_scene_scene_info_row.setLayout(self.test_scene_scene_info_layout)
        self.test_scene_scene_info_layout.addWidget(self.test_scene_scene_info_label)
        self.test_scene_scene_info_layout.addStretch()
        self.test_scene_scene_info_layout.addWidget(self.test_scene_view_scene_button)
        form.addRow("", self.test_scene_scene_info_row)
        self.test_scene_scene_info_row.hide() 

        # Iterations per model
        self.test_scene_iters_input = QSpinBox()
        self.test_scene_iters_input.setRange(1, 9999)
        self.test_scene_iters_input.setValue(10)
        form.addRow("Iterations per model (default 10):", self.test_scene_iters_input)

        # Verbose
        self.test_scene_verbose_input = QSpinBox()
        self.test_scene_verbose_input.setRange(0, 3)
        self.test_scene_verbose_input.setValue(1)
        form.addRow("Verbose Level (default: 1):", self.test_scene_verbose_input)

        # Options
        self.test_scene_no_gui_checkbox = QCheckBox("Disable GUI")
        form.addRow("Options:", self.test_scene_no_gui_checkbox)

        # Start button
        test_scene_button = self.create_styled_button("Test Scene", self.start_test_scene)

        # Center layout
        button_layout = QVBoxLayout()
        button_layout.addStretch()

        centered_h = QHBoxLayout()
        centered_h.addStretch()
        centered_h.addWidget(test_scene_button)
        centered_h.addStretch()

        button_layout.addLayout(centered_h)
        button_layout.addStretch()

        layout.addLayout(form)
        layout.addLayout(button_layout)

        tab.setLayout(layout)
        return tab

    

    def handle_plot_type_change(self):
        selected_types = [cb.text() for cb in self.plot_types_checkboxes if cb.isChecked()]
        show_scene_input = "plot_scene_trajs" in selected_types
        logging.info(f"Selected types: {selected_types}")

        if show_scene_input:
            robot_name = self.plot_robot_name_input.currentText()
            if robot_name != "Select a robot...":
                self.robot_name = robot_name  
                self.populate_input_names(self.plot_scene_to_load_folder_input, category="scene_configs")
            self.scene_folder_row.show()
        else:
            self.scene_folder_row.hide()


    def create_plot_tab(self):
        """Tab for plot configuration."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Form for plot parameters
        form = QFormLayout()
        self.plot_robot_name_input = QComboBox()
        self.plot_robot_name_input.setEditable(False)
        self.populate_input_names(self.plot_robot_name_input, "robot")
        self.plot_robot_name_input.currentIndexChanged.connect(self.update_model_ids_for_selected_robot)
        self.plot_model_ids_input = QListWidget()
        self.plot_model_ids_input.setFixedHeight(200)
        self.plot_model_ids_input.setSelectionMode(QAbstractItemView.NoSelection)
        self.plot_types_checkboxes = []
        

        grid_widget = QWidget()
        grid_layout = QGridLayout()
        grid_widget.setLayout(grid_layout)

        info_button = self.create_info_button_with_tooltip(PLOT_TOOLTIPS)

        cols = 2
        for index, plot_type in enumerate(PLOT_TYPES):
            checkbox = QCheckBox(plot_type)
            row = index // cols
            col = index % cols
            grid_layout.addWidget(checkbox, row, col)
            self.plot_types_checkboxes.append(checkbox)

        for checkbox in self.plot_types_checkboxes:
            checkbox.stateChanged.connect(self.handle_plot_type_change)

        
        plot_label_with_icon = QWidget()
        plot_label_layout = QHBoxLayout()
        plot_label_layout.setContentsMargins(0, 0, 0, 0)
        plot_label_with_icon.setLayout(plot_label_layout)

        plot_label = QLabel("Plot Types:")
        plot_label_layout.addWidget(plot_label)
        plot_label_layout.addWidget(info_button)
        plot_label_layout.addStretch()

        self.scene_folder_row = QWidget()
        scene_form_layout = QFormLayout()
        scene_form_layout.setContentsMargins(0, 0, 0, 0)
        scene_form_layout.setHorizontalSpacing(60)
        self.scene_folder_row.setLayout(scene_form_layout)

        # Combo
        self.plot_scene_to_load_folder_input = QComboBox()
        scene_form_layout.addRow("Scene Folder:", self.plot_scene_to_load_folder_input)

        self.scene_folder_row.hide()

        


        form.addRow("Robot name (required):", self.plot_robot_name_input)
        form.addRow("Model IDs (required):", self.plot_model_ids_input)
        form.addRow(plot_label_with_icon, grid_widget)
        form.addRow(self.scene_folder_row)

        self.update_input_placeholder(self.plot_model_ids_input, "Select a robot first")

        # Buttons
        plot_button = self.create_styled_button("Generate Plots", self.start_plot)

        button_layout = QVBoxLayout()
        button_layout.addStretch()

        centered_h = QHBoxLayout()
        centered_h.addStretch()
        centered_h.addWidget(plot_button)
        centered_h.addStretch()

        button_layout.addLayout(centered_h)
        button_layout.addStretch()

        layout.addLayout(form)
        layout.addLayout(button_layout)


        tab.setLayout(layout)
        return tab
    

    def update_input_placeholder(self, widget, text):
        """
        Show a placeholder in a QListWidget or QComboBox.
        
        Args:
            widget (QListWidget or QComboBox): The target input widget.
            text (str): Placeholder message to show.
        """
        if widget is None:
            logging.warning("No widget provided for placeholder update.")
            return

        from PyQt5.QtWidgets import QListWidget, QComboBox

        if isinstance(widget, QListWidget):
            widget.clear()
            placeholder = QListWidgetItem(text)
            placeholder.setFlags(Qt.NoItemFlags)
            placeholder.setForeground(Qt.gray)
            widget.addItem(placeholder)

        elif isinstance(widget, QComboBox):
            widget.clear()
            widget.addItem(text)
            widget.setEnabled(False)
            widget.model().item(0).setEnabled(False)

        else:
            logging.warning(f"Unsupported widget type: {type(widget)}")


    def update_model_ids_for_selected_robot(self):
        """Update the available models based on the selected robot."""
        current_tab = self.tabs.tabText(self.tabs.currentIndex())
        
        if current_tab == "Plot":
            self.robot_name = self.plot_robot_name_input.currentText()
            target_list_widget = self.plot_model_ids_input
        elif current_tab == "Test scene":
            self.robot_name = self.test_scene_robot_name_input.currentText()
            target_list_widget = self.test_scene_model_ids_input
        else:
            logging.warning(f"Unsupported tab: {current_tab}")
            return  # Unsupported tab

        if self.robot_name.startswith("Select"):
            if current_tab == "Plot":
                self.update_input_placeholder(self.plot_model_ids_input, "Select a robot first")
            elif current_tab == "Test scene":
                self.update_input_placeholder(self.test_scene_model_ids_input, "Select a robot first")
        
        else:

            model_dir = os.path.join(self.base_path, "robots", self.robot_name, "models")
            if not os.path.isdir(model_dir):
                if current_tab == "Plot":
                    self.update_input_placeholder(self.plot_model_ids_input, "No models found for this robot")
                elif current_tab == "Test scene":
                    self.update_input_placeholder(self.test_scene_model_ids_input, "No models found for this robot")
            
            else:

                # Load action times from train_records.csv file
                action_times = {}
                csv_path = os.path.join(self.base_path, "robots", self.robot_name, "training_metrics", "train_records.csv")
                if os.path.isfile(csv_path):
                    with open(csv_path, newline="") as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            model_name = row.get("Exp_id") 
                            action_time = row.get("Action time (s)")
                            if model_name and action_time:
                                action_times[model_name.strip()] = action_time.strip()

                # Find valid model IDs
                model_ids = []
                for entry in os.listdir(model_dir):
                    subdir_path = os.path.join(model_dir, entry)
                    if os.path.isdir(subdir_path):
                        match = re.match(rf"{self.robot_name}_model_(\d+)", entry)
                        if match:
                            model_id = match.group(1)
                            expected_file = f"{self.robot_name}_model_{model_id}_last.zip"
                            expected_path = os.path.join(subdir_path, expected_file)
                            if os.path.isfile(expected_path):
                                model_ids.append(model_id)

                if not model_ids:
                    if current_tab == "Plot":
                        self.update_input_placeholder(self.plot_model_ids_input, "No valid models found")
                    elif current_tab == "Test scene":
                        self.update_input_placeholder(self.test_scene_model_ids_input, "No valid models found")
                
                else:

                    # Update target QListWidget with checkboxes and info
                    target_list_widget.clear()
                    for model_id in sorted(model_ids, key=int):
                        full_model_name = f"{self.robot_name}_model_{model_id}"
                        time_str = action_times.get(full_model_name, "n/a")

                        item = QListWidgetItem()
                        item.setSizeHint(QSize(0, 20))

                        widget = QWidget()
                        layout = QHBoxLayout(widget)
                        layout.setContentsMargins(0, 0, 0, 0)

                        checkbox = QCheckBox()
                        checkbox.setProperty("model_id", model_id)
                        checkbox.setText(model_id)
                        layout.addWidget(checkbox)

                        label = QLabel(f"<span style='color:gray;'>Action time: {time_str}s</span>")
                        label.setTextFormat(Qt.RichText)
                        layout.addWidget(label)

                        layout.addStretch()

                        target_list_widget.addItem(item)
                        target_list_widget.setItemWidget(item, widget)

        # Update scene folder options if necessary
        if current_tab == "Plot":
            self.handle_plot_type_change()
        elif current_tab == "Test scene":
            self.handle_test_scene_robot_selection()
        elif current_tab == "Auto traininig":
            self.handle_auto_train_robot_selection()


    def populate_robot_names(self):
        """Load available robot names from robots/ directory into the dropdown."""
        robots_dir = os.path.join(self.base_path, "robots")
        if os.path.isdir(robots_dir):
            robot_names = sorted(
                [name for name in os.listdir(robots_dir) if os.path.isdir(os.path.join(robots_dir, name))]
            )
            self.plot_robot_name_input.clear()
            self.plot_robot_name_input.addItems(robot_names)
        else:
            logging.warning(f"Robots directory not found at: {robots_dir}")

        self.plot_robot_name_input.clear()
        self.plot_robot_name_input.addItem("Select a robot...")
        self.plot_robot_name_input.model().item(0).setEnabled(False)  # No seleccionable

        self.plot_robot_name_input.addItems(robot_names)
        self.plot_robot_name_input.setCurrentIndex(0)


    def parse_params_json(self, file_name, search_dir):
        file_path = os.path.join(search_dir, file_name)

        # Intenta leer parámetros específicos
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            time_val = data.get("params_env", {}).get("fixed_actime", "n/a")
            algo_val = data.get("params_train", {}).get("sb3_algorithm", "n/a")
            steps_val = data.get("params_train", {}).get("total_timesteps", "n/a")

            time_str = f"{time_val}s" if isinstance(time_val, (int, float)) else "n/a"
            algo_str = algo_val if algo_val else "n/a"
            steps_str = str(steps_val) if steps_val else "n/a"

            # Construir el texto enriquecido
            display_text = f"{file_name}   —   Action time: {time_str} | Algorithm: {algo_str} | Steps: {steps_str}"

        except Exception as e:
            logging.warning(f"Could not parse {file_name}: {e}")
            display_text = f"{file_name}   —   Invalid or missing data"

        return display_text


    def populate_input_names(self, input_widget, category):
        """Load available <category> names into a dropdown menu."""

        if category == "robot":
            search_dir = os.path.join(self.base_path, "robots")
            default_text = "Select a robot..."
            warning_text = "Robots directory not found at: "
        elif category == "scene_configs":
            if not hasattr(self, "robot_name") or not self.robot_name:
                self.logs_text.append("<span style='color:red;'>DEBUG: robot_name no estaba definido al cargar escena.</span>")
                logging.warning("Robot name not set when loading scene_configs.")
                return
            search_dir = os.path.join(self.base_path, "robots", self.robot_name, "scene_configs")
            default_text = "Select a scene folder to load..."
            warning_text = "Scene configs directory not found at: "
        elif category == "params_file":
            search_dir = os.path.join(self.base_path, "configs")
            default_text = "Select a parameters file..."
            warning_text = "Configs directory not found at: "

        elif category == "session_folders":
            if not hasattr(self, "robot_name") or not self.robot_name:
                self.logs_text.append("<span style='color:red;'>DEBUG: robot_name no estaba definido al cargar session_folders.</span>")
                logging.warning("Robot name not set when loading session_folders.")
                return
            search_dir = os.path.join(self.base_path, "robots", self.robot_name, "auto_trainings")
            default_text = "Select a session folder for auto training..."
            warning_text = "Session folders for auto training directory not found at: "
            

        else:
            logging.warning(f"Unknown input category: {category}")
            return
        
        # Guardar si existía la opción manual
        preserve_manual = category == "params_file" and "Manual parameters" in [input_widget.itemText(i) for i in range(input_widget.count())]


        input_widget.clear()
        input_widget.addItem(default_text)
        input_widget.model().item(0).setEnabled(False)

        if os.path.isdir(search_dir):
            if category == "params_file":
                files = sorted([
                    name for name in os.listdir(search_dir)
                    if name.endswith(".json") and os.path.isfile(os.path.join(search_dir, name))
                ])
                for file_name in files:
                    input_widget.addItem(self.parse_params_json(file_name, search_dir))
                logging.info(f"{category} --> Files found: {files}")

            else:
                possible_names = sorted([
                    name for name in os.listdir(search_dir)
                    if os.path.isdir(os.path.join(search_dir, name))
                ])
                input_widget.addItems(possible_names)
                input_widget.setCurrentIndex(0)
                logging.info(f"{category} --> Folders found: {possible_names}")
                if possible_names is None or len(possible_names) == 0:
                    if category == "scene_configs":
                        self.update_input_placeholder(input_widget, "No scene configs found")
                    elif category == "robot":
                        self.update_input_placeholder(input_widget, "No robots found")
                    elif category == "session_folders":
                        self.update_input_placeholder(input_widget, "No session folders found")
                else:
                    input_widget.setEnabled(True)

        else:
            logging.warning(warning_text + search_dir)
            self.update_input_placeholder(input_widget, "Scene configs directory not found")
            
        if preserve_manual:
            input_widget.insertSeparator(input_widget.count())
            input_widget.addItem("Manual parameters")
            input_widget.setCurrentIndex(0)

        if category == "scene_configs":
            input_widget.insertSeparator(input_widget.count())
            input_widget.addItem("Custom your scene")


    def handle_auto_train_session_name_change(self):
        """Check if a session folder for auto training exists, and if it contains json files."""
        if self.auto_train_session_name_input:
            if not (self.auto_train_session_name_input.currentText().strip().startswith("Select") or self.auto_train_session_name_input.currentText().strip().startswith("Scene")):
                session_name = self.auto_train_session_name_input.currentText()
                # Get the directory containing the parameter files for the session.
                session_dir = os.path.join(self.base_path, "robots", self.robot_name, "auto_trainings", session_name)
                    
                # Create the directory if it doesn't exist
                os.makedirs(session_dir, exist_ok=True)

                # Check if the directory is empty
                if not (session_name.strip().startswith("Select") or session_name.strip().startswith("Scene")):
                    if not os.listdir(session_dir):
                        warning = f"ERROR: The directory {session_dir} is empty. Please add the desired param.json files for training."
                        logging.critical(warning)
                        self.logs_text.append(f"<span style='color:red;'> --- {warning}</span>")
                        return
                    else:
                        # Search all the json files inside the provided folder.
                        param_files = glob.glob(os.path.join(session_dir, "*.json"))
                        message = f"Found {len(param_files)} parameter files in {session_dir}."
                        logging.info(message)
                        self.logs_text.append(f"<span style='color:green;'> --- </span>{message}")
    

    def create_auto_training_tab(self):
        """Tab for auto training configuration."""
        tab = QWidget()
        layout = QVBoxLayout()

        form = QFormLayout()

        # Robot name (combo)
        self.auto_train_robot_name_input = QComboBox()
        self.auto_train_robot_name_input.addItem("Select a robot...")
        self.auto_train_robot_name_input.model().item(0).setEnabled(False)
        self.populate_input_names(self.auto_train_robot_name_input, category="robot")
        self.auto_train_robot_name_input.currentIndexChanged.connect(self.handle_auto_train_robot_selection)
        form.addRow("Robot Name (required):", self.auto_train_robot_name_input)

        # Session Name (required)
        self.auto_train_session_name_input = QComboBox()
        self.auto_train_session_name_input.currentTextChanged.connect(self.handle_auto_train_session_name_change) 
        form.addRow("Session Name (required):", self.auto_train_session_name_input)

        # Disable parallel mode (optional)
        self.auto_train_disable_parallel_checkbox = QCheckBox("Disable Parallel Mode")
        form.addRow("Options: ", self.auto_train_disable_parallel_checkbox)

        # Max workers (optional, only relevant if parallel mode is active)
        self.auto_train_max_workers_input = QSpinBox()
        self.auto_train_max_workers_input.setRange(1, 10)
        self.auto_train_max_workers_input.setValue(3)
        form.addRow("Max Workers (default: 3):", self.auto_train_max_workers_input)

        # Verbose level
        self.auto_train_verbose_input = QSpinBox()
        self.auto_train_verbose_input.setRange(0, 3)
        self.auto_train_verbose_input.setValue(1)
        form.addRow("Verbose Level (default: 1)):", self.auto_train_verbose_input)

        # Start button
        auto_train_button = self.create_styled_button("Start Auto Training", self.start_auto_training)

        # Layout centrado
        button_layout = QVBoxLayout()
        button_layout.addStretch()

        centered_h = QHBoxLayout()
        centered_h.addStretch()
        centered_h.addWidget(auto_train_button)
        centered_h.addStretch()

        button_layout.addLayout(centered_h)
        button_layout.addStretch()

        layout.addLayout(form)
        layout.addLayout(button_layout)
        tab.setLayout(layout)
        return tab


    def create_auto_testing_tab(self):
        """Tab for auto testing configuration."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Form for auto testing parameters
        form = QFormLayout()
        self.auto_test_robot_name_input = QLineEdit()
        self.auto_test_model_ids_input = QLineEdit()
        self.auto_test_iterations_input = QSpinBox()
        self.auto_test_iterations_input.setRange(1, 1000)

        form.addRow("Robot Name:", self.auto_test_robot_name_input)
        form.addRow("Model IDs (comma-separated):", self.auto_test_model_ids_input)
        form.addRow("Iterations:", self.auto_test_iterations_input)

        # Buttons
        auto_test_button = QPushButton("Start Auto Testing")
        auto_test_button.clicked.connect(self.start_auto_testing)
        layout.addLayout(form)
        layout.addWidget(auto_test_button)

        tab.setLayout(layout)
        return tab

    def create_retrain_tab(self):
        """Tab for retraining configuration."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Form for retraining parameters
        form = QFormLayout()
        self.retrain_model_name_input = QLineEdit()
        self.retrain_steps_input = QSpinBox()
        self.retrain_steps_input.setRange(1, 1000000)

        form.addRow("Model Name:", self.retrain_model_name_input)
        form.addRow("Retrain Steps:", self.retrain_steps_input)

        # Buttons
        retrain_button = QPushButton("Start Retraining")
        retrain_button.clicked.connect(self.start_retraining)
        layout.addLayout(form)
        layout.addWidget(retrain_button)

        tab.setLayout(layout)
        return tab
    

    def create_manage_tab(self):
        """
        Create the 'Manage' tab to view and administer robots and their trained models.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Main horizontal layout
        main_layout = QHBoxLayout()

        # --- Left: List of robots ---
        robot_panel = QVBoxLayout()
        robot_label = QLabel("Available Robots:")
        self.robot_list = QListWidget()
        self.robot_list.itemSelectionChanged.connect(self.handle_robot_selected_in_manage_tab)

        self.delete_robot_button = QPushButton("Delete Robot")
        self.delete_robot_button.setIcon(QIcon.fromTheme("edit-delete"))
        self.delete_robot_button.clicked.connect(self.handle_delete_robot)
        self.delete_robot_button.setEnabled(False)

        robot_panel.addWidget(robot_label)
        robot_panel.addWidget(self.robot_list)
        robot_panel.addWidget(self.delete_robot_button)

        # --- Right: Models for selected robot ---
        model_panel = QVBoxLayout()
        model_label = QLabel("Models for selected robot:")
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.delete_model_button = QPushButton("Delete Selected Model(s)")
        self.delete_model_button.setIcon(QIcon.fromTheme("edit-delete"))
        self.delete_model_button.clicked.connect(self.handle_delete_models)
        self.delete_model_button.setEnabled(False)

        model_panel.addWidget(model_label)
        model_panel.addWidget(self.model_list)
        model_panel.addWidget(self.delete_model_button)

        # Add both panels to the main layout
        main_layout.addLayout(robot_panel, 1)
        main_layout.addLayout(model_panel, 2)

        layout.addLayout(main_layout)
        tab.setLayout(layout)

        self.populate_robot_list()
        return tab

    def populate_robot_list(self):
        """Populate the list of available robots."""
        self.robot_list.clear()
        robots_path = os.path.join(self.base_path, "robots")
        if not os.path.exists(robots_path):
            return
        for robot in sorted(os.listdir(robots_path)):
            robot_dir = os.path.join(robots_path, robot)
            if os.path.isdir(robot_dir):
                self.robot_list.addItem(robot)

    def handle_robot_selected_in_manage_tab(self):
        """When a robot is selected, populate its models."""
        selected_items = self.robot_list.selectedItems()
        self.model_list.clear()
        self.delete_robot_button.setEnabled(bool(selected_items))
        self.delete_model_button.setEnabled(False)

        if not selected_items:
            return

        robot_name = selected_items[0].text()
        models_dir = os.path.join(self.base_path, "robots", robot_name, "models")

        if not os.path.exists(models_dir):
            return

        for root, _, files in os.walk(models_dir):
            for f in sorted(files):
                if f.endswith(".zip"):
                    rel_path = os.path.relpath(os.path.join(root, f), models_dir)
                    self.model_list.addItem(rel_path)

        self.model_list.itemSelectionChanged.connect(lambda: self.delete_model_button.setEnabled(bool(self.model_list.selectedItems())))

    def handle_delete_robot(self):
        """Delete the selected robot folder after confirmation."""
        selected_items = self.robot_list.selectedItems()
        if not selected_items:
            return

        robot_name = selected_items[0].text()
        reply = QMessageBox.question(self, "Confirm Deletion", f"Are you sure you want to delete the robot '{robot_name}' and all its data?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            robot_path = os.path.join(self.base_path, "robots", robot_name)
            try:
                shutil.rmtree(robot_path)
                self.logs_text.append(f"<span style='color:red;'> --- Robot '{robot_name}' deleted.</span>")
            except Exception as e:
                logging.warning(f"Failed to delete robot '{robot_name}': {e}")
                self.logs_text.append(f"<span style='color:red;'> --- Error deleting robot '{robot_name}'.</span>")
            self.populate_robot_list()
            self.model_list.clear()
            self.delete_robot_button.setEnabled(False)
            self.delete_model_button.setEnabled(False)

    def handle_delete_models(self):
        """Delete selected models for the selected robot."""
        robot_items = self.robot_list.selectedItems()
        model_items = self.model_list.selectedItems()
        if not robot_items or not model_items:
            return

        robot_name = robot_items[0].text()
        model_names = [item.text() for item in model_items]

        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the selected model(s):\n{', '.join(model_names)}",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            deleted = 0
            for model_name in model_names:
                model_path = os.path.join(self.base_path, "robots", robot_name, "models", model_name)
                try:
                    os.remove(model_path)
                    deleted += 1
                except Exception as e:
                    logging.warning(f"Failed to delete model '{model_name}': {e}")
            self.logs_text.append(f"<span style='color:red;'> --- Deleted {deleted} model(s) from '{robot_name}'.</span>")
            self.handle_robot_selected_in_manage_tab()



    def browse_scene(self):
        """Open a file dialog to select a scene file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Scene File", "", "Scene Files (*.ttt)")
        if file_path:
            self.scene_path_input.setText(file_path)


    def start_train(self):

        """Start the training process."""
        if self.train_robot_name_combo.currentText() == "Create a new one!":
            robot_name = self.train_new_robot_name_input.text().strip()
        else:
            robot_name = self.train_robot_name_combo.currentText()

        process_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        terminal_title = f"CoppeliaTerminal_{process_timestamp}"

        args = [
            "rl_coppelia", "train",
            "--robot_name", robot_name,
            "--timestamp", str(process_timestamp),
            "--verbose", str(self.train_verbose_input.value())
        ]

        if self.train_params_file_input.currentText():
            params_filename = self.train_params_file_input.currentText().split()[0]
            if params_filename != "Select":
                params_fullpath = os.path.join(self.base_path, "configs", params_filename)
                args += ["--params_file", params_fullpath]
        if self.train_dis_parallel_mode_checkbox.isChecked():
            args.append("--dis_parallel_mode")
        if self.train_no_gui_checkbox.isChecked():
            args.append("--no_gui")
            
        logging.info(f"Starting training with args: {args}")


        # Crear thread
        train_thread = ProcessThread(args) 
        train_thread.terminal_title = terminal_title

        # Crear widget visual del proceso
        process_widget = QWidget()
        process_layout = QVBoxLayout()
        process_widget.setLayout(process_layout)

        process_widget.process_type = "Train"
        process_widget.timestamp = process_timestamp
        process_widget.model_name = robot_name

        info_label = QLabel(f"<b>Train</b> — {process_timestamp}")
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        stop_button = QPushButton("Stop")
        stop_button.clicked.connect(lambda: self.stop_specific_process(train_thread, process_widget))

        process_layout.addWidget(info_label)
        process_layout.addWidget(progress_bar)
        process_layout.addWidget(stop_button)

        self.processes_container.addWidget(process_widget)
        self.update_processes_label()

        # Conectar señales
        train_thread.progress_signal.connect(progress_bar.setValue)
        train_thread.finished_signal.connect(lambda: self.on_process_finished(process_widget))
        train_thread.error_signal.connect(lambda msg: self.on_process_error(msg, process_widget))

        train_thread.start()

        # Deactivate 'Start training' button for few seconds
        button = self.sender()
        if isinstance(button, QPushButton):
            self.disable_button_with_countdown(button, seconds=20)
        
        stop_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_message = f"<span style='color:green;'> --- </span>Training process started successfully at <b>{stop_time}</b>."
        self.logs_text.append(start_message)



    def stop_specific_process(self, process_thread, process_widget):
        """Stop an individual train/test process and remove its widget."""
        if process_thread.isRunning():
            process_thread.stop()
            logging.info("Stopping thread...")

        if hasattr(process_thread, 'terminal_title'):
            logging.info(f"Closing terminal: {process_thread.terminal_title}")
            try:
                subprocess.run(["wmctrl", "-c", process_thread.terminal_title], check=True)
            except subprocess.CalledProcessError as e:
                logging.warning(f"⚠️ wmctrl could not close the terminal: {e}")

                try: 
                    logging.info("Trying to close terminal by PID...")
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        if 'gnome-terminal' in proc.info['name'] or 'gnome-terminal' in ' '.join(proc.info['cmdline']):
                            cmdline = ' '.join(proc.info['cmdline'])
                            if process_thread.terminal_title in cmdline:
                                try:
                                    proc.terminate()
                                    proc.wait(timeout=5)
                                    logging.info(f"Terminal '{process_thread.terminal_title}' closed by PID {proc.pid}")
                                    return True
                                except Exception as e:
                                    logging.warning(f"Failed to close terminal '{process_thread.terminal_title}': {e}")
                    logging.warning(f"No terminal found with title containing '{process_thread.terminal_title}'")
                except Exception as e:
                    logging.error(f"Error while trying to close terminal: {e}") 

        # Log parada manual
        process_type = getattr(process_widget, 'process_type', 'Unknown')
        timestamp = getattr(process_widget, 'timestamp', 'Unknown')
        model_name = getattr(process_widget, 'model_name', 'UnknownModel')
        stop_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        message = f"<span style='color:orange;'> --- ⏹️ Process <b>{process_type}</b> <i>{timestamp}</i> of <code>{model_name}</code> was manually stopped at <b>{stop_time}</b>.</span>"
        self.logs_text.append(message)

        process_widget.setParent(None)
        self.update_processes_label()


    def on_process_finished(self, process_widget):
        """Handle successful process completion and log it to the GUI."""
        process_type = getattr(process_widget, 'process_type', 'Unknown')
        timestamp = getattr(process_widget, 'timestamp', 'Unknown')
        model_name = getattr(process_widget, 'model_name', 'UnknownModel')
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        message = f"<span style='color:green;'> --- Success: </span> Process <b>{process_type}</b> <i>{timestamp}</i> of <code>{model_name}</code> finished successfully at <b>{end_time}</b>."
        self.logs_text.append(message)

        process_widget.setParent(None)
        self.update_processes_label()


    def on_process_error(self, error_message, process_widget):
        """Handle error during process execution and log it."""
        process_type = getattr(process_widget, 'process_type', 'Unknown')
        timestamp = getattr(process_widget, 'timestamp', 'Unknown')
        model_name = getattr(process_widget, 'model_name', 'UnknownModel')
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        message = (
            f"<span style='color:red;'> --- ❌ Process <b>{process_type}</b> <i>{timestamp}</i> of <code>{model_name}</code> "
            f"failed at <b>{end_time}</b> with error:<br>{error_message}</span>"
        )
        self.logs_text.append(message)

        process_widget.setParent(None)
        self.update_processes_label()


    def disable_button_with_countdown(self, button, seconds=8):
        """Disable a button and show a countdown in its text."""
        original_text = button.text()
        button.setEnabled(False)

        def update_text():
            nonlocal seconds
            if seconds > 0:
                button.setText(f"Wait... ({seconds})")
                seconds -= 1
            else:
                self.timer.stop()
                button.setText(original_text)
                button.setEnabled(True)

        self.timer = QTimer(self)
        self.timer.timeout.connect(update_text)
        self.timer.start(1000)  # 1 segundo


    def start_testing(self):
        """Start the testing process (multi-threaded and UI-tracked)."""
        self.model_name = self.test_model_name_input.text()
        if not self.model_name:
            warning_msg = "<span style='color:orange;'> --- ⚠️ Warning: please select a valid model name.</span>"
            self.logs_text.append(warning_msg)
            logging.warning("Please select a valid model name.")
            return
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        terminal_title = f"CoppeliaTerminal_{timestamp}"

        args = [
            "rl_coppelia", "test",
            "--model_name", self.remove_zip_extension(self.test_model_name_input.text()),
            "--iterations", str(self.test_iterations_input.value()),
            "--timestamp", str(timestamp),
            "--verbose", str(self.verbose_input.value())
        ]

        # Optional args
        if self.save_traj_checkbox.isChecked():
            args.append("--save_traj")
        if self.test_params_file_input.text():
            args += ["--params_file", self.test_params_file_input.text()]
        if self.test_robot_name_input.text():
            args += ["--robot_name", self.test_robot_name_input.text()]
        if self.test_scene_path_input.text():
            args += ["--scene_path", self.test_scene_path_input.text()]
        if self.dis_parallel_mode_checkbox.isChecked():
            args.append("--dis_parallel_mode")
        if self.no_gui_checkbox.isChecked():
            args.append("--no_gui")

        logging.info(f"Starting testing with args: {args}")

        # Create test thread
        test_thread = ProcessThread(args)
        test_thread.terminal_title = terminal_title

        # Create a widget for the process
        process_widget = QWidget()
        process_layout = QVBoxLayout()
        process_widget.setLayout(process_layout)

        # Save metadata
        process_widget.process_type = "Test"
        process_widget.timestamp = timestamp
        process_widget.model_name = self.model_name or "UnknownModel"

        info_label = QLabel(f"<b>Test</b> — {timestamp}")
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        stop_button = QPushButton("Stop")
        stop_button.clicked.connect(lambda: self.stop_specific_process(test_thread, process_widget))

        process_layout.addWidget(info_label)
        process_layout.addWidget(progress_bar)
        process_layout.addWidget(stop_button)

        self.processes_container.addWidget(process_widget)
        self.update_processes_label()

        # Conect signales with this specific progress bar
        test_thread.progress_signal.connect(progress_bar.setValue)
        test_thread.finished_signal.connect(lambda: self.on_process_finished(process_widget))
        test_thread.error_signal.connect(lambda msg: self.on_process_error(msg, process_widget))

        test_thread.start()

        # Deactivate 'Start testing' button for few seconds
        button = self.sender()
        if isinstance(button, QPushButton):
            self.disable_button_with_countdown(button, seconds=8)

    def start_test_scene(self):
        """Start the test_scene process using selected models and scene folder."""
        robot_name = self.test_scene_robot_name_input.currentText().strip()
        if robot_name == "Select a robot...":
            self.logs_text.append("<span style='color:orange;'>⚠️ Please select a robot name.</span>")
            return

        # Leer modelos seleccionados
        selected_model_ids = []
        for i in range(self.test_scene_model_ids_input.count()):
            item = self.test_scene_model_ids_input.item(i)
            widget = self.test_scene_model_ids_input.itemWidget(item)
            if widget:
                checkbox = widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    selected_model_ids.append(checkbox.text().strip())

        if not selected_model_ids:
            self.logs_text.append("<span style='color:orange;'>⚠️ Please select at least one model ID.</span>")
            return

        # Leer carpeta de escena
        scene_folder = self.test_scene_scene_to_load_folder_input.currentText().strip()
        if not scene_folder or scene_folder.startswith("Select"):
            self.logs_text.append("<span style='color:orange;'>⚠️ Please select a scene folder.</span>")
            return

        iters = self.test_scene_iters_input.value()
        verbose = self.test_scene_verbose_input.value()
        no_gui = self.test_scene_no_gui_checkbox.isChecked()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        terminal_title = f"TestSceneTerminal_{timestamp}"

        args = [
            "rl_coppelia", "test_scene",
            "--robot_name", robot_name,
            "--model_ids", *selected_model_ids,
            "--scene_to_load_folder", scene_folder,
            "--iters_per_model", str(iters),
            "--timestamp", str(timestamp),
            "--verbose", str(verbose)
        ]
        if no_gui:
            args.append("--no_gui")

        # Crear y configurar el thread
        thread = ProcessThread(args)
        thread.terminal_title = terminal_title

        # Crear bloque visual
        process_widget = QWidget()
        layout = QVBoxLayout(process_widget)

        process_widget.process_type = "Test Scene"
        process_widget.timestamp = timestamp
        process_widget.model_name = robot_name

        info_label = QLabel(f"<b>Test Scene</b> — {timestamp}")
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        stop_button = QPushButton("Stop")
        stop_button.clicked.connect(lambda: self.stop_specific_process(thread, process_widget))

        layout.addWidget(info_label)
        layout.addWidget(progress_bar)
        layout.addWidget(stop_button)

        self.processes_container.addWidget(process_widget)
        self.update_processes_label()

        # Conectar señales
        thread.progress_signal.connect(progress_bar.setValue)
        thread.finished_signal.connect(lambda: self.on_process_finished(process_widget))
        thread.error_signal.connect(lambda msg: self.on_process_error(msg, process_widget))

        thread.start()

        # Desactivar el botón unos segundos
        button = self.sender()
        if isinstance(button, QPushButton):
            self.disable_button_with_countdown(button, seconds=8)

        self.logs_text.append(f"<span style='color:green;'> --- Starting Test Scene at {timestamp}</span>")


    def start_plot(self):
        """Generate plots."""
        # Check if robot name was selected
        robot_name = self.plot_robot_name_input.currentText()
        if robot_name == "Select a robot...":   # It means that the user did not pick any robot
            self.logs_text.append("<span style='color:orange;'> --- ⚠️ Please select a robot.</span>")
            return

        selected_ids = []
        for i in range(self.plot_model_ids_input.count()):
            item = self.plot_model_ids_input.item(i)
            widget = self.plot_model_ids_input.itemWidget(item)
            if widget:
                checkbox = widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    selected_ids.append(int(checkbox.property("model_id")))

        selected_types = [
            cb.text() for cb in self.plot_types_checkboxes if cb.isChecked()
        ]


        if not selected_ids:
            if "plot_scene_trajs" not in selected_types:
                self.logs_text.append("<span style='color:orange;'> --- ⚠️ Please select at least one model ID.</span>")
                return
            else:
                selected_ids.append(9999)

        if not selected_types:
            self.logs_text.append("<span style='color:orange;'> --- ⚠️ Please select at least one plot type.</span>")
            return

        args = [
            "plot",
            "--robot_name", robot_name,
            "--model_ids", *map(str, selected_ids),
            "--plot_types", *selected_types,
            "--verbose", str(10)
        ]

        if "plot_scene_trajs" in selected_types:  
            scene_folder = self.plot_scene_to_load_folder_input.currentText()
            if scene_folder and scene_folder != "Select a folder...":
                args.extend(["--scene_to_load_folder", scene_folder])

        logging.info(f"Generating plots with args: {args}")
        output, errors, success = self.capture_cli_output(cli.main, argv=args)

        if errors:
            self.logs_text.append("<span style='color:red;'> --- ❌ Errors detected during plotting:</span>")
            for err in errors:
                self.logs_text.append(f"<pre>{err}</pre>")
        elif not success:
            self.logs_text.append("<span style='color:red;'> --- ❌ Exception occurred during plotting.</span>")
        else:
            if 9999 in selected_ids:
                success_text = f"<span style='color:green;'> --- Plots generated successfully</span>: scene folder {scene_folder}, plot type: {', '.join(selected_types)}."
            else:
                success_text = f"<span style='color:green;'> --- Plots generated successfully</span>: models {selected_ids} with plot types: {', '.join(selected_types)}."
            self.logs_text.append(success_text)



    def start_auto_training(self):
        """Start auto training."""
        session_name = self.auto_train_session_name_input.currentText().strip()
        robot_name = self.auto_train_robot_name_input.currentText().strip()
        dis_parallel = self.auto_train_disable_parallel_checkbox.isChecked()
        max_workers = self.auto_train_max_workers_input.value()
        verbose = self.auto_train_verbose_input.value()

        if not session_name or not robot_name:
            self.logs_text.append("<span style='color:orange;'>⚠️ Please provide both session and robot names.</span>")
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        terminal_title = f"AutoTrain_{timestamp}"

        args = [
            "rl_coppelia", "auto_training",
            "--session_name", session_name,
            "--robot_name", robot_name,
            "--max_workers", str(max_workers),
            "--timestamp", str(timestamp),
            "--verbose", str(verbose)
        ]
        if dis_parallel:
            args.append("--dis_parallel_mode")

        logging.info(f"Starting auto training with args: {args}")

        thread = ProcessThread(args)
        thread.terminal_title = terminal_title

        # Widget visual
        process_widget = QWidget()
        layout = QVBoxLayout()
        process_widget.setLayout(layout)

        process_widget.process_type = "Auto Train"
        process_widget.timestamp = timestamp
        process_widget.model_name = robot_name

        info_label = QLabel(f"<b>Auto Train</b> — {timestamp}")
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        stop_button = QPushButton("Stop")
        stop_button.clicked.connect(lambda: self.stop_specific_process(thread, process_widget))

        layout.addWidget(info_label)
        layout.addWidget(progress_bar)
        layout.addWidget(stop_button)

        self.processes_container.addWidget(process_widget)
        self.update_processes_label()

        thread.progress_signal.connect(progress_bar.setValue)
        thread.finished_signal.connect(lambda: self.on_process_finished(process_widget))
        thread.error_signal.connect(lambda msg: self.on_process_error(msg, process_widget))

        thread.start()

        # Disable button for cooldown
        button = self.sender()
        if isinstance(button, QPushButton):
            self.disable_button_with_countdown(button, seconds=8)

        self.logs_text.append(f"<span style='color:green;'> --- Starting Auto Training at {timestamp}</span>")


    def start_auto_testing(self):
        """Start auto testing."""
        args = {
            "robot_name": self.auto_test_robot_name_input.text(),
            "model_ids": [int(x) for x in self.auto_test_model_ids_input.text().split(",")],
            "iterations": self.auto_test_iterations_input.value(),
        }
        logging.info(f"Starting auto testing with args: {args}")
        auto_testing.main(args)

    def start_retraining(self):
        """Start retraining."""
        args = {
            "model_name": self.retrain_model_name_input.text(),
            "retrain_steps": self.retrain_steps_input.value(),
        }
        logging.info(f"Starting retraining with args: {args}")
        retrain.main(args)


def main():
    utils.logging_config_gui()
    app = QApplication(sys.argv)
    logo_path = pkg_resources.resource_filename("rl_coppelia", "assets/uncore.png")

    app.setWindowIcon(QIcon(logo_path))
    window = MainApp()
    window.show()
    sys.exit(app.exec_())