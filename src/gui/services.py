from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
import re
from typing import List
import io
import logging
import contextlib

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QCheckBox, QListWidget, QListWidgetItem, QLabel, QComboBox
)

from PyQt5.QtCore import Qt, pyqtSignal,QSize



def capture_cli_output(callable_fn, argv=None):
    """Run a CLI-style callable, capturing stdout, stderr and logging output.

    Args:
        callable_fn (function): A callable to run (e.g., cli.main).
        argv (list, optional): Argument list to simulate CLI input.

    Returns:
        tuple: (combined_output: str, errors: list[str], success: bool)
    """
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


def remove_zip_extension(self, file_path):
        """Remove the .zip extension from the file name if it exists."""
        base_name, extension = os.path.splitext(file_path)
        if extension.lower() == ".zip":
            return base_name
        return file_path


def list_dirs(path: str) -> list[str]:
    """Return a sorted list of directory names under path (non-recursive)."""
    if not os.path.isdir(path):
        return []
    return sorted([n for n in os.listdir(path) if os.path.isdir(os.path.join(path, n))])


def list_json_files(path: str) -> list[str]:
    """Return a sorted list of json files under path."""
    if not os.path.isdir(path):
        return []
    return sorted([n for n in os.listdir(path) if n.endswith(".json")])


def get_rl_coppelia_path_from_bashrc(base_path):
    """
    Retrieve the rl_coppelia path from ~/.bashrc.
    Looks for rl_coppelia in PYTHONPATH or PATH environment variables.
    If found, returns the parent directory of rl_coppelia.
    Args:
        base_path (str): The expected base path to compare against.
    Returns:
        str or None: The base path if found, else None.
    """
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
                        if base_path != base_path:
                            base_path = base_path  # Update the base path
                            logging.warning(f"Found rl_coppelia path in PYTHONPATH: {base_path}, but it does not match the expected base path: {base_path}")
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
                        if base_path != base_path:
                            base_path = base_path
                            logging.warning(f"Found rl_coppelia path in PYTHONPATH: {base_path}, but it does not match the expected base path: {base_path}")
                        return base_path
                        
    except Exception as e:
        logging.error(f"Error reading .bashrc: {e}")
        
    return None
    

def get_browse_zip_file_path(self):
        """Open a file dialog to select a ZIP file, starting in the rl_coppelia directory."""
        rl_coppelia_path = self.get_rl_coppelia_path_from_bashrc()
        
        # If rl_coppelia path was found, then it will be used as main directory for searching files
        if rl_coppelia_path and os.path.exists(rl_coppelia_path):
            start_path = rl_coppelia_path
            logging.info(f"Starting file dialog in rl_coppelia directory: {start_path}")
        else:
            start_path = os.path.expanduser("~")
            logging.warning(f"rl_coppelia path not found, starting in home directory: {start_path}")
        return start_path


def _get_action_times(self, robot: str) -> dict:
    """Load action times from train_records.csv file."""
    action_times: dict = {}
    base = self._get_base_path()
    csv_path = os.path.join(base, "robots", robot, "training_metrics", "train_records.csv")
    if os.path.isfile(csv_path):
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                model_name = row.get("Exp_id") 
                action_time = row.get("Action time (s)")
                if model_name and action_time:
                    action_times[model_name.strip()] = action_time.strip()
    return action_times

def _get_last_models(self, robot: str) -> List[str]:
    """Return list of last model ZIP files for the robot."""
    models: List[str] = []
    base = self._get_base_path()
    model_dir = os.path.join(base, "robots", robot, "models")
    if not os.path.isdir(model_dir):
        return models

    try:
        for entry in os.listdir(model_dir):
            subdir_path = os.path.join(model_dir, entry)
            if os.path.isdir(subdir_path):
                m = re.match(rf"{re.escape(robot)}_model_(\d+)", entry)
                if not m:
                    continue
                model_id = m.group(1)
                expected_file = os.path.join(subdir_path, f"{robot}_model_{model_id}_last.zip")
                if os.path.isfile(expected_file):
                    models.append(model_id)
    except Exception as exc:
        logging.warning(f"Error listing models: {exc}")

    return models


def refresh_model_ids(self, robot: str) -> None:
    """Populate model list with checkboxes for the selected robot."""
    self.model_ids_input.clear()
    if not self.robot or self.robot.startswith("Select"):
        set_list_placeholder(self.model_ids_input, "Select a robot first")
        return

    base_path = self._get_base_path()
    model_dir = os.path.join(base_path, "robots", robot, "models")
    if not os.path.isdir(model_dir):
        set_list_placeholder(self.model_ids_input, "No models found for this robot")
        return

    # Load action times from train_records.csv file
    action_times = _get_action_times(self, robot)

    # Collect valid models: <robot>_model_<id>/<robot>_model_<id>_last.zip
    model_ids = _get_last_models(self, robot)

    if not model_ids:
        set_list_placeholder(self.model_ids_input, "No valid models found")
        return

    # Build items with checkboxes
    for model_id in sorted(model_ids, key=int):
        full_model_name = f"{robot}_model_{model_id}"
        time_str = action_times.get(full_model_name, "n/a")
        item = QListWidgetItem()
        item.setSizeHint(QSize(0, 20))

        # Build widget with checkbox and optional info
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        checkbox = QCheckBox()
        checkbox.setProperty("model_id", model_id)
        checkbox.setText(model_id)
        layout.addWidget(checkbox)

        # Optional secondary info (e.g., action time):
        label = QLabel(f"<span style='color:gray;'>Action time: {time_str}s</span>")
        label.setTextFormat(Qt.RichText)
        layout.addWidget(label)
        layout.addStretch()

        self.model_ids_input.addItem(item)
        self.model_ids_input.setItemWidget(item, widget)


def set_list_placeholder(w, text):
    """Show a disabled placeholder item in a QListWidget or a QComboBox.
    Args:
        w (QListWidget or QComboBox): The widget to update.
        text (str): The placeholder text to display.
    """

    if isinstance(w, QListWidget):
        w.clear()
        placeholder = QListWidgetItem(text)
        placeholder.setFlags(Qt.NoItemFlags)
        placeholder.setForeground(Qt.gray)
        w.addItem(placeholder)

    elif isinstance(w, QComboBox):
        w.clear()
        w.addItem(text)
        w.setEnabled(False)
        w.model().item(0).setEnabled(False)

    else:
        logging.warning(f"Unsupported widget type: {type(w)}")


def refresh_lists(self, input_widget, category):
    """Load available <category> names into a dropdown menu."""
    request_log = pyqtSignal(str)

    if category == "robot":
        search_dir = os.path.join(self._get_base_path(), "robots")
        default_text = "Select a robot..."
        warning_text = "Robots directory not found at: "
    elif category == "scene_configs":
        if not hasattr(self, "robot") or not self.robot:
            self.request_log.emit("<span style='color:red;'>DEBUG: robot_name no estaba definido al cargar escena.</span>")
            logging.warning("Robot name not set when loading scene_configs.")
            return
        search_dir = os.path.join(self._get_base_path(), "robots", self.robot, "scene_configs")
        default_text = "Select a scene folder to load..."
        warning_text = "Scene configs directory not found at: "
    elif category == "params_file":
        search_dir = os.path.join(self.base_path, "configs")
        default_text = "Select a parameters file..."
        warning_text = "Configs directory not found at: "

    elif category == "session_folders":
        if not hasattr(self, "robot") or not self.robot:
            self.request_log.emit("<span style='color:red;'>DEBUG: robot_name no estaba definido al cargar session_folders.</span>")
            logging.warning("Robot name not set when loading session_folders.")
            return
        search_dir = os.path.join(self._get_base_path(), "robots", self.robot, "auto_trainings")
        default_text = "Select a session folder for auto training..."
        warning_text = "Session folders for auto training directory not found at: "
        
    else:
        logging.warning(f"Unknown input category: {category}")
        return
    
    # Save manual entry if required
    preserve_manual = category == "params_file" and "Manual parameters" in [input_widget.itemText(i) for i in range(input_widget.count())]

    # Clear and set placeholder
    input_widget.clear()
    input_widget.addItem(default_text)
    input_widget.model().item(0).setEnabled(False)

    # Populate with available names
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
                    set_list_placeholder(input_widget, "No scene configs found")
                elif category == "robot":
                    set_list_placeholder(input_widget, "No robots found")
                elif category == "session_folders":
                    set_list_placeholder(input_widget, "No session folders found")
            else:
                input_widget.setEnabled(True)

    else:
        logging.warning(warning_text + search_dir)
        set_list_placeholder(input_widget, "Scene configs directory not found")
    
    # Restore manual entry if it was present
    if preserve_manual:
        input_widget.insertSeparator(input_widget.count())
        input_widget.addItem("Manual parameters")
        input_widget.setCurrentIndex(0)

    # Add custom option for scene configs
    if category == "scene_configs":
        input_widget.insertSeparator(input_widget.count())
        input_widget.addItem("Custom your scene")


def on_tab_changed(self, index):
    """Trigger refresh of file-based inputs when switching tabs."""
    current_tab = self.tabs.tabText(index)

    if current_tab == "Train":
        refresh_lists(self.train_params_file_input, category="params_file")

    elif current_tab == "Plot":
        robot_name = self.plot_robot_name_input.currentText()
        if robot_name and not robot_name.startswith("Select"):
            self.update_model_ids_for_selected_robot()
            refresh_lists(self, self.plot_scene_to_load_folder_input, category="scene_configs")

    elif current_tab == "Test scene":
        robot_name = self.test_scene_robot_name_input.currentText()
        if robot_name and not robot_name.startswith("Select"):
            self.update_model_ids_for_selected_robot()
            refresh_lists(self,self.test_scene_scene_to_load_folder_input, category="scene_configs")
    elif current_tab == "Auto Training":
        robot_name = self.auto_train_robot_name_input.currentText()
        if robot_name and not robot_name.startswith("Select"):
            refresh_lists(self, self.auto_train_session_name_input, category="session_folders")