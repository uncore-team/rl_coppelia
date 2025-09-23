import logging
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QLabel, QCheckBox, QSpinBox, QFileDialog, QLabel,
    QHBoxLayout, QToolTip, QDialog, QGroupBox, QMessageBox, QComboBox, QDoubleSpinBox,
    QTableWidgetItem, QTableWidget
)
from PyQt5.QtGui import QIcon
import pkg_resources


def create_icon_button(tip_text: str, icon_path: str, on_click: callable) -> QPushButton:
    """
    Create a button with an icon and tooltip.
    Args:
        tip_text (str): The tooltip text to display on hover.
        icon_path (str): The path to the icon image.
        on_click (callable): The function to call when the button is clicked.
    Returns:
        QPushButton: The created button with the specified icon and tooltip.
    """
    button = QPushButton()
    button.setIcon(QIcon(pkg_resources.resource_filename("rl_coppelia", "../gui/assets/edit_icon.png")))
    button.setToolTip(tip_text)
    button.setFixedSize(24, 24)
    button.setVisible(False)
    button.clicked.connect(on_click)

    return button


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
    

def refresh_lists(self, input_widget, category):
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


def on_tab_changed(self, index):
    """Trigger refresh of file-based inputs when switching tabs."""
    current_tab = self.tabs.tabText(index)

    if current_tab == "Train":
        refresh_lists(self.train_params_file_input, category="params_file")

    elif current_tab == "Plot":
        robot_name = self.plot_robot_name_input.currentText()
        if robot_name and not robot_name.startswith("Select"):
            self.update_model_ids_for_selected_robot()
            refresh_lists(self.plot_scene_to_load_folder_input, category="scene_configs")

    elif current_tab == "Test scene":
        robot_name = self.test_scene_robot_name_input.currentText()
        if robot_name and not robot_name.startswith("Select"):
            self.update_model_ids_for_selected_robot()
            refresh_lists(self.test_scene_scene_to_load_folder_input, category="scene_configs")
    elif current_tab == "Auto Training":
        robot_name = self.auto_train_robot_name_input.currentText()
        if robot_name and not robot_name.startswith("Select"):
            refresh_lists(self.auto_train_session_name_input, category="session_folders")