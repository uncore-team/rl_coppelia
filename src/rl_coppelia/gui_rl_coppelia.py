import contextlib
import csv
from datetime import datetime
import io
import os
import re
import subprocess
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QLabel, QCheckBox, QSpinBox, QFileDialog
)
from PyQt5.QtCore import Qt
import logging
from common import utils
from rl_coppelia import cli, train, test, plot, auto_training, auto_testing, retrain, sat_training
from pathlib import Path
from PyQt5.QtCore import QTimer, QThread, pyqtSignal,QSize
from PyQt5.QtWidgets import QProgressBar, QGridLayout, QHBoxLayout, QTextEdit, QSizePolicy, QScrollArea, QAbstractItemView,QListWidgetItem, QListWidget, QComboBox
from PyQt5.QtGui import QIcon
import pkg_resources


PLOT_TYPES = ["spider", "convergence-time", "convergence-steps", "compare-rewards", "compare-episodes_length", 
                "histogram_speeds", "histogram_speed_comparison", "hist_target_zones", "bar_target_zones"]


class TestThread(QThread):
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
        """Execute each test in a separate thread, so the user can run multiple tests simultanously."""
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


    def load_main_interface(self):
        """Load the main interface after the welcome screen."""
        self.resize(1000, 600)
        expanded_path = os.path.abspath(__file__)
        self.base_path = str(Path(expanded_path).parents[2])
        logging.debug(f"Base path configured as: {self.base_path}")
        self.model_name = ""
        self.robot_name = ""
        self.experiment_id = 0

        main_layout = QHBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_train_tab(), "Train")
        self.tabs.addTab(self.create_test_tab(), "Test")
        self.tabs.addTab(self.create_plot_tab(), "Plot")
        self.tabs.addTab(self.create_auto_training_tab(), "Auto Training")
        self.tabs.addTab(self.create_auto_testing_tab(), "Auto Testing")
        self.tabs.addTab(self.create_retrain_tab(), "Retrain")
        main_layout.addWidget(self.tabs, stretch=3)

        side_panel_widget = QWidget()
        side_panel_widget.setLayout(self.create_side_panel())
        main_layout.addWidget(side_panel_widget, stretch=1)


    def create_styled_button(self, text: str, on_click: callable) -> QPushButton:
        """Create a consistently styled action button."""
        button = QPushButton(text)
        button.setFixedSize(180, 50)
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


    def create_side_panel(self):
        """Creates the side panel with scrollable logs and a scrollable process list."""
        from PyQt5.QtWidgets import QScrollArea

        layout = QVBoxLayout()

        # Logs
        layout.addWidget(QLabel("Logs"))

        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setLineWrapMode(QTextEdit.NoWrap)

        logs_scroll = QScrollArea()
        logs_scroll.setWidgetResizable(True)
        logs_scroll.setWidget(self.logs_text)
        logs_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        layout.addWidget(logs_scroll, stretch=1)  # It will fill the upper half

        # Contenedor de procesos
        self.processes_label = QLabel("No processes yet")  # init the fesult message
        layout.addWidget(self.processes_label)

        process_scroll_content = QWidget()
        self.processes_container = QVBoxLayout()
        process_scroll_content.setLayout(self.processes_container)

        process_scroll = QScrollArea()
        process_scroll.setWidgetResizable(True)
        process_scroll.setWidget(process_scroll_content)

        layout.addWidget(process_scroll, stretch=1)  # It fills the lower half

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
                    self.test_scene_path_input.setText(warning_message)
                else:
                    logging.info(f"Scene file {scene_path} found for robot {robot_name}.")
                    self.test_scene_path_input.setText(scene_path)

                # Construct the params file path
                params_file_path = os.path.join(self.base_path, "robots", robot_name, "parameters_used", f"params_file_model_{self.experiment_id}.json")
                # Check if the params file exists
                if not os.path.exists(params_file_path):
                    warning_message = f"WARNING: {params_file_path} does not exist. Please check the model name."
                    logging.warning(warning_message)
                    self.test_params_file_input.setText(warning_message)
                else:
                    logging.info(f"Params file {params_file_path} found for robot {robot_name}.")
                    self.test_params_file_input.setText(params_file_path)
                
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
            print(f"Error reading .bashrc: {e}")
            
        return None
    

    def browse_zip_file(self):
        """Open a file dialog to select a ZIP file, starting in the rl_coppelia directory."""
        rl_coppelia_path = self.get_rl_coppelia_path_from_bashrc()
        
        # If rl_coppelia path was found, then it will be used as main directory for searching files
        if rl_coppelia_path and os.path.exists(rl_coppelia_path):
            start_path = rl_coppelia_path
            print(f"Starting file dialog in rl_coppelia directory: {start_path}")
        else:
            start_path = os.path.expanduser("~")
            print(f"rl_coppelia path not found, starting in home directory: {start_path}")
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select ZIP File", 
            start_path, 
            "ZIP Files (*.zip)"
        )
        
        if file_path:
            self.test_model_name_input.setText(file_path)
            print(f"Selected file: {file_path}")

    def remove_zip_extension(self, file_path):
        """Remove the .zip extension from the file name if it exists."""
        base_name, extension = os.path.splitext(file_path)
        if extension.lower() == ".zip":
            return base_name
        return file_path


    def create_train_tab(self):
        """Tab for training configuration."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Form for training parameters
        form = QFormLayout()
        self.robot_name_input = QLineEdit()
        self.scene_path_input = QLineEdit()
        self.params_file_input = QLineEdit()
        self.verbose_input = QSpinBox()
        self.verbose_input.setRange(0, 2)

        form.addRow("Robot Name:", self.robot_name_input)
        form.addRow("Scene Path:", self.scene_path_input)
        form.addRow("Params File:", self.params_file_input)
        form.addRow("Verbose Level:", self.verbose_input)

        # Buttons
        browse_button = QPushButton("Browse Scene")
        browse_button.clicked.connect(self.browse_scene)
        form.addRow("", browse_button)

        train_button = QPushButton("Start Training")
        train_button.clicked.connect(self.start_training)
        layout.addLayout(form)
        layout.addWidget(train_button)

        tab.setLayout(layout)
        return tab

    def create_test_tab(self):
        """Tab for testing configuration."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Form for testing parameters
        form = QFormLayout()

        # Model name (required)
        self.test_model_name_input = QLineEdit()
        self.test_model_name_input.setPlaceholderText("Select a ZIP file...")
        browse_zip_button = QPushButton("Browse ZIP")
        browse_zip_button.clicked.connect(self.browse_zip_file)
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
        self.test_iterations_input.setValue(5)

        # Verbose level (optional, default: 3)
        self.verbose_input = QSpinBox()
        self.verbose_input.setRange(-1, 4)
        self.verbose_input.setValue(3)

       

        

        # Add fields to the form
        form.addRow("Model ZIP File (required):", self.test_model_name_input)
        form.addRow("", browse_zip_button)
        form.addRow("Robot Name (optional):", self.test_robot_name_input)
        form.addRow("Scene Path (optional):", self.test_scene_path_input)
        form.addRow("", self.save_scene_checkbox)
        form.addRow("", self.save_traj_checkbox)
        form.addRow("", self.dis_parallel_mode_checkbox)
        form.addRow("", self.no_gui_checkbox)
        form.addRow("Params File (optional):", self.test_params_file_input)
        form.addRow("Iterations (default: 400):", self.test_iterations_input)
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

    




    def create_plot_tab(self):
        """Tab for plot configuration."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Form for plot parameters
        form = QFormLayout()
        self.plot_robot_name_input = QComboBox()
        self.plot_robot_name_input.setEditable(False)
        self.populate_robot_names()
        self.plot_robot_name_input.currentIndexChanged.connect(self.update_model_ids_for_selected_robot)
        self.plot_model_ids_input = QListWidget()
        self.plot_model_ids_input.setFixedHeight(200)
        self.plot_model_ids_input.setSelectionMode(QAbstractItemView.NoSelection)
        self.plot_types_checkboxes = []

        grid_widget = QWidget()
        grid_layout = QGridLayout()
        grid_widget.setLayout(grid_layout)

        cols = 2
        for index, plot_type in enumerate(PLOT_TYPES):
            checkbox = QCheckBox(plot_type)
            row = index // cols
            col = index % cols
            grid_layout.addWidget(checkbox, row, col)
            self.plot_types_checkboxes.append(checkbox)

        form.addRow("Robot name (required):", self.plot_robot_name_input)
        form.addRow("Model IDs (required):", self.plot_model_ids_input)
        form.addRow("Plot types:", grid_widget)

        self.update_model_ids_placeholder("Select a robot first")

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
    

    def update_model_ids_placeholder(self, text):
        """Show a placeholder item in the model IDs list."""
        self.plot_model_ids_input.clear()
        placeholder = QListWidgetItem(text)
        placeholder.setFlags(Qt.NoItemFlags)
        placeholder.setForeground(Qt.gray)
        self.plot_model_ids_input.addItem(placeholder)


    def update_model_ids_for_selected_robot(self):
        """Update the avaliable models accordingly to the selected robot."""
        robot_name = self.plot_robot_name_input.currentText()
        if robot_name.startswith("Select"):
            self.update_model_ids_placeholder("Select a robot first")
            return

        model_dir = os.path.join(self.base_path, "robots", robot_name, "models")
        if not os.path.isdir(model_dir):
            self.update_model_ids_placeholder("No models found for this robot")
            return

        # Load action times from train_records.csv file
        action_times = {}
        csv_path = os.path.join(self.base_path, "robots", robot_name, "training_metrics", "train_records.csv")
        if os.path.isfile(csv_path):
            with open(csv_path, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    model_name = row.get("Exp_id") 
                    action_time = row.get("Action time (s)")
                    if model_name and action_time:
                        action_times[model_name.strip()] = action_time.strip()

        # Search valid models
        model_ids = []
        for entry in os.listdir(model_dir):
            subdir_path = os.path.join(model_dir, entry)
            if os.path.isdir(subdir_path):
                match = re.match(rf"{robot_name}_model_(\d+)", entry)
                if match:
                    model_id = match.group(1)
                    expected_file = f"{robot_name}_model_{model_id}_last.zip"
                    expected_path = os.path.join(subdir_path, expected_file)
                    if os.path.isfile(expected_path):
                        model_ids.append(model_id)

        if not model_ids:
            self.update_model_ids_placeholder("No valid models found")
            return

        # Show checkboxes with models and their action times
        self.plot_model_ids_input.clear()
        for model_id in sorted(model_ids, key=int):
            full_model_name = f"{robot_name}_model_{model_id}"
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

            self.plot_model_ids_input.addItem(item)
            self.plot_model_ids_input.setItemWidget(item, widget)



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





    def create_auto_training_tab(self):
        """Tab for auto training configuration."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Form for auto training parameters
        form = QFormLayout()
        self.auto_train_session_name_input = QLineEdit()
        self.auto_train_robot_name_input = QLineEdit()
        self.auto_train_workers_input = QSpinBox()
        self.auto_train_workers_input.setRange(1, 10)

        form.addRow("Session Name:", self.auto_train_session_name_input)
        form.addRow("Robot Name:", self.auto_train_robot_name_input)
        form.addRow("Max Workers:", self.auto_train_workers_input)

        # Buttons
        auto_train_button = QPushButton("Start Auto Training")
        auto_train_button.clicked.connect(self.start_auto_training)
        layout.addLayout(form)
        layout.addWidget(auto_train_button)

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

    def browse_scene(self):
        """Open a file dialog to select a scene file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Scene File", "", "Scene Files (*.ttt)")
        if file_path:
            self.scene_path_input.setText(file_path)

    def start_training(self):
        """Start the training process."""
        args = {
            "robot_name": self.robot_name_input.text(),
            "scene_path": self.scene_path_input.text(),
            "params_file": self.params_file_input.text(),
            "verbose": self.verbose_input.value(),
        }
        logging.info(f"Starting training with args: {args}")
        train.main(args)


    def stop_specific_test(self, test_thread, process_widget):
        """Stop an individual test process and remove its widget."""
        if test_thread.isRunning():
            test_thread.stop()
            logging.info("Stopping test thread...")

        if hasattr(test_thread, 'terminal_title'):
            logging.info(f"Closing terminal: {test_thread.terminal_title}")
            subprocess.run(['wmctrl', '-c', test_thread.terminal_title], check=False)

        # Log parada manual
        process_type = getattr(process_widget, 'process_type', 'Unknown')
        timestamp = getattr(process_widget, 'timestamp', 'Unknown')
        model_name = getattr(process_widget, 'model_name', 'UnknownModel')
        stop_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        message = f"<span style='color:orange;'>⏹️ Process <b>{process_type}</b> <i>{timestamp}</i> of <code>{model_name}</code> was manually stopped at <b>{stop_time}</b>.</span>"
        self.logs_text.append(message)

        process_widget.setParent(None)
        self.update_processes_label()


    def on_process_finished(self, process_widget):
        """Handle successful process completion and log it to the GUI."""
        process_type = getattr(process_widget, 'process_type', 'Unknown')
        timestamp = getattr(process_widget, 'timestamp', 'Unknown')
        model_name = getattr(process_widget, 'model_name', 'UnknownModel')
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        message = f"<span style='color:green;'>Success: </span> Process <b>{process_type}</b> <i>{timestamp}</i> of <code>{model_name}</code> finished successfully at <b>{end_time}</b>."
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
            f"<span style='color:red;'>❌ Process <b>{process_type}</b> <i>{timestamp}</i> of <code>{model_name}</code> "
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
            warning_msg = "<span style='color:orange;'>⚠️ Warning: please select a valid model name.</span>"
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
        test_thread = TestThread(args)
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
        stop_button.clicked.connect(lambda: self.stop_specific_test(test_thread, process_widget))

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



    def start_plot(self):
        """Generate plots."""
        # Check if robot name was selected
        robot_name = self.plot_robot_name_input.currentText()
        if robot_name == "Select a robot...":   # It means that the user did not pick any robot
            self.logs_text.append("<span style='color:orange;'>⚠️ Please select a robot.</span>")
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
            self.logs_text.append("<span style='color:orange;'>⚠️ Please select at least one model ID.</span>")
            return

        if not selected_types:
            self.logs_text.append("<span style='color:orange;'>⚠️ Please select at least one plot type.</span>")
            return

        args = [
            "plot",
            "--robot_name", robot_name,
            "--model_ids", *map(str, selected_ids),
            "--plot_types", *selected_types,
            "--verbose", str(10)
        ]
        logging.info(f"Generating plots with args: {args}")
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                cli.main(args)
            except Exception as e:
                self.logs_text.append(f"<span style='color:red;'>❌ Exception during plotting: {str(e)}</span>")
                return

        content = output.getvalue()

        # Search lines with "ERROR"
        errors = [line for line in content.splitlines() if " - ERROR - " in line]

        if errors:
            self.logs_text.append("<span style='color:red;'>❌ Errors detected during plotting:</span>")
            for err in errors:
                self.logs_text.append(f"<pre>{err}</pre>")
        else:
            self.logs_text.append(
                f"<span style='color:green;'> --> </span> Plots generated for models {selected_ids} with types: {', '.join(selected_types)}."
            )


    def start_auto_training(self):
        """Start auto training."""
        args = {
            "session_name": self.auto_train_session_name_input.text(),
            "robot_name": self.auto_train_robot_name_input.text(),
            "max_workers": self.auto_train_workers_input.value(),
        }
        logging.info(f"Starting auto training with args: {args}")
        auto_training.main(args)

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