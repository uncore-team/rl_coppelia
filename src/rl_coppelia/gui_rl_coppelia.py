import contextlib
import csv
from datetime import datetime
import io
import json
import os
import re
import subprocess
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QLabel, QCheckBox, QSpinBox, QFileDialog, QLabel,
    QHBoxLayout, QToolTip, QDialog, QGroupBox, QMessageBox, QComboBox
)
from PyQt5.QtCore import Qt
import logging
from common import utils
from rl_coppelia import cli, train, test, plot, auto_training, auto_testing, retrain, sat_training
from pathlib import Path
from PyQt5.QtCore import QTimer, QThread, pyqtSignal,QSize,QPoint,QEvent
from PyQt5.QtWidgets import QProgressBar, QGridLayout, QHBoxLayout, QTextEdit, QSizePolicy, QScrollArea, QAbstractItemView,QListWidgetItem, QListWidget, QComboBox
from PyQt5.QtGui import QIcon, QPixmap, QIntValidator, QDoubleValidator
import pkg_resources


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

tooltip_text = """<b>Available Plot Types:</b><ul>
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
    params_saved = pyqtSignal(str)  # Signal to emit when parameters are saved

    def __init__(self, base_path, robot_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual parameter definition")
        self.base_path = base_path
        self.robot_name = robot_name

        self.default_path = os.path.join(self.base_path, "configs", "params_default_file.json")
        self.target_dir = os.path.join(self.base_path, "configs")
        os.makedirs(self.target_dir, exist_ok=True)

        with open(self.default_path, "r") as f:
            self.default_data = json.load(f)

        self.field_widgets = {}  # (section, key) -> widget

        layout = QVBoxLayout()
        # ["Parameters - Environment", "Parameters - Training", "Parameters - Testing"]:
        for section in ["params_env", "params_train"]:
            box = self.build_section(section, self.default_data.get(section, {}))
            layout.addWidget(box)

        self.file_name_input = QLineEdit()
        self.file_name_input.setPlaceholderText("Enter name for new param file")
        layout.addWidget(QLabel("New file name (without .json):"))
        layout.addWidget(self.file_name_input)

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: red;")
        layout.addWidget(self.warning_label)


        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_changes)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)  # Close the dialog without saving

        apply_button.setAutoDefault(False)
        cancel_button.setAutoDefault(False)
        apply_button.setDefault(False)
        cancel_button.setDefault(False)

        buttons_layout.addWidget(cancel_button)
        buttons_layout.addWidget(apply_button)

        layout.addLayout(buttons_layout)

        self.setLayout(layout)

    def build_section(self, section_name, fields):
        if section_name == "params_env":
            section_name = "Parameters - Environment"
        elif section_name == "params_train":
            section_name = "Parameters - Training"
        group = QGroupBox(section_name)
        layout = QGridLayout()
        layout.setHorizontalSpacing(25)  # Add horizontal spacing between columns

        tooltips = self.get_param_tooltips()

        row = 0
        col = 0
        for key, value in fields.items():
            label = QLabel(key)
            label.setToolTip(tooltips.get(key, ""))

            if isinstance(value, bool):
                field = QCheckBox()
                field.setChecked(value)
            else:
                field = QLineEdit(str(value))
                if isinstance(value, int):
                    field.setValidator(QIntValidator())
                elif isinstance(value, float):
                    field.setValidator(QDoubleValidator())

            self.field_widgets[(section_name, key)] = field

            layout.addWidget(label, row, col * 2)
            layout.addWidget(field, row, col * 2 + 1)

            col += 1
            if col == 2:
                col = 0
                row += 1

        group.setLayout(layout)
        return group

    def apply_changes(self):
        new_data = json.loads(json.dumps(self.default_data)) 
        for (section, key), widget in self.field_widgets.items():
            if section == "Parameters - Environment":
                section = "params_env"
            elif section == "Parameters - Training":
                section = "params_train"
            if isinstance(widget, QCheckBox):
                value = widget.isChecked()
            else:
                text = widget.text()
                if text.strip() == "":
                    continue
                original = self.default_data[section][key]
                try:
                    if isinstance(original, int):
                        value = int(text)
                    elif isinstance(original, float):
                        value = float(text)
                    else:
                        value = text
                except ValueError:
                    value = text

            new_data[section][key] = value

        name = self.file_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing name", "Please provide a name for the new parameters file.")
            return

        file_path = os.path.join(self.target_dir, f"{name}.json")
        if os.path.exists(file_path):
            self.warning_label.setText(f"❌ File '{file_path}' already exists.")
            return
        else:
            self.warning_label.setText("")  # Limpiar cualquier error anterior
        try:
            with open(file_path, "w") as f:
                json.dump(new_data, f, indent=4)
            self.params_saved.emit(f"{name}.json")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Could not save file: {e}")

    def get_param_tooltips(self):
        return {
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
                # logging.debug(line.strip())  # Console debug
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


        self.plot_tooltip_label = QLabel(self)
        self.plot_tooltip_label.setText(tooltip_text)
        self.plot_tooltip_label.setWindowFlags(Qt.ToolTip)
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
                self.populate_input_names(self.scene_to_load_folder_input, category="scene_trajs")



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
        self.tabs.addTab(self.create_plot_tab(), "Plot")
        self.tabs.addTab(self.create_auto_training_tab(), "Auto Training")
        self.tabs.addTab(self.create_auto_testing_tab(), "Auto Testing")
        self.tabs.addTab(self.create_retrain_tab(), "Retrain")

        # === Conexión cambio de pestaña ===
        self.tabs.currentChanged.connect(self.on_tab_changed)

        # === Ajustar altura del botón tras renderizado ===
        QTimer.singleShot(100, self.sync_refresh_button_height)



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
        self.dis_parallel_mode_checkbox = QCheckBox("Disable Parallel Mode")

        # Disable GUI (optional)
        self.no_gui_checkbox = QCheckBox("Disable GUI")

        self.train_verbose_input = QSpinBox()
        self.train_verbose_input.setRange(-1, 4)
        self.train_verbose_input.setValue(3)

        form.addRow("Robot Name (required):", robot_row)
        form.addRow("Scene Path (optional):", self.train_scene_path_input)
        form.addRow("Params File:", params_file_row)
        form.addRow("Options: ", self.dis_parallel_mode_checkbox)
        form.addRow("", self.no_gui_checkbox)
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
        form.addRow("Options: ", self.save_scene_checkbox)
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

    

    def handle_plot_type_change(self):
        selected_types = [cb.text() for cb in self.plot_types_checkboxes if cb.isChecked()]
        show_scene_input = "plot_scene_trajs" in selected_types
        logging.info(f"Selected types: {selected_types}")

        if show_scene_input:
            robot_name = self.plot_robot_name_input.currentText()
            if robot_name != "Select a robot...":
                self.robot_name = robot_name  
                self.populate_input_names(self.scene_to_load_folder_input, category="scene_trajs")
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

        info_button = self.create_info_button_with_tooltip(tooltip_text)

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
        self.scene_to_load_folder_input = QComboBox()
        scene_form_layout.addRow("Scene Folder:", self.scene_to_load_folder_input)

        self.scene_folder_row.hide()

        


        form.addRow("Robot name (required):", self.plot_robot_name_input)
        form.addRow("Model IDs (required):", self.plot_model_ids_input)
        form.addRow(plot_label_with_icon, grid_widget)
        form.addRow(self.scene_folder_row)

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

    def update_scene_folder_options(self):
        folder_path = os.path.join(self.base_path, "robots", self.robot_name, "scene_configs")
        self.scene_to_load_folder_input.clear()

        if not os.path.isdir(folder_path):
            self.scene_to_load_folder_input.addItem("No folders found")
            self.scene_to_load_folder_input.setEnabled(False)
            return

        folders = sorted([f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))])
        if not folders:
            self.scene_to_load_folder_input.addItem("No folders found")
            self.scene_to_load_folder_input.setEnabled(False)
        else:
            self.scene_to_load_folder_input.addItem("Select a folder...")
            self.scene_to_load_folder_input.addItems(folders)
            self.scene_to_load_folder_input.setEnabled(True)


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

        self.handle_plot_type_change()


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
        elif category == "scene_trajs":
            if not hasattr(self, "robot_name") or not self.robot_name:
                logging.warning("Robot name not set when loading scene_trajs.")
                return
            search_dir = os.path.join(self.base_path, "robots", self.robot_name, "scene_configs")
            default_text = "Select a scene folder to load..."
            warning_text = "Scene configs directory not found at: "
        elif category == "params_file":
            search_dir = os.path.join(self.base_path, "configs")
            default_text = "Select a parameters file..."
            warning_text = "Configs directory not found at: "
            

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
        else:
            logging.warning(warning_text + search_dir)

        if preserve_manual:
            input_widget.insertSeparator(input_widget.count())
            input_widget.addItem("Manual parameters")
            input_widget.setCurrentIndex(0)


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


    def start_train(self):

        """Start the training process."""
        if self.train_robot_name_combo.currentText() == "Create a new one":
            robot_name = self.train_new_robot_name_input.text().strip()
        else:
            robot_name = self.train_robot_name_combo.currentText()

        args = [
            "rl_coppelia", "train",
            "--robot_name", robot_name,
            "--iterations", str(self.test_iterations_input.value()),
            "--verbose", str(self.verbose_input.value())
        ]
        
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
            scene_folder = self.scene_to_load_folder_input.currentText()
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