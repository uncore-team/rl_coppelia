"""Manage tab: encapsulated UI and logic to manage our robot/model files."""

from __future__ import annotations
import logging
import os
import re
import shutil
from typing import Callable
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox, QListWidget,
    QAbstractItemView, QLabel
)
from PyQt5.QtGui import QIcon
from common import utils


class ManageTab(QWidget):
    """Encapsulated Train tab.
    
    Exposes:
        - signal_start(args, process_type, model_name): emitted when user clicks Start.
        - request_log(html): emitted to append messages in main log panel.
    """
    signal_start = pyqtSignal(list, str, str)  # args, process_type, model_name
    request_log = pyqtSignal(str)

    def __init__(self, base_path_getter: Callable[[], str], parent=None):
        """Build the Manage tab.

        Args:
            base_path_getter: Callable returning the current project base path.
        """
        super().__init__(parent)
        self._get_base_path = base_path_getter        

        # Main horizontal layout
        self.main_layout = QHBoxLayout()

        # --- Left: List of robots ---
        self.robot_panel = QVBoxLayout()
        self.robot_label = QLabel("Available Robots:")
        self.robot_list = QListWidget()
        self.robot_list.itemSelectionChanged.connect(self._on_robot_selection)

        # Delete robot button
        self.delete_robot_btn = QPushButton("Delete Robot")
        self.delete_robot_btn.setIcon(QIcon.fromTheme("edit-delete"))
        self.delete_robot_btn.clicked.connect(self._handle_delete_robot)
        self.delete_robot_btn.setEnabled(False)

        self.robot_panel.addWidget(self.robot_label)
        self.robot_panel.addWidget(self.robot_list)
        self.robot_panel.addWidget(self.delete_robot_btn)

        # --- Right: Models for selected robot ---
        self.model_panel = QVBoxLayout()
        self.model_label = QLabel("Models for selected robot:")
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.delete_model_btn = QPushButton("Delete Selected Model(s)")
        self.delete_model_btn.setIcon(QIcon.fromTheme("edit-delete"))
        self.delete_model_btn.clicked.connect(self._handle_delete_models)
        self.delete_model_btn.setEnabled(False)

        self.model_panel.addWidget(self.model_label)
        self.model_panel.addWidget(self.model_list)
        self.model_panel.addWidget(self.delete_model_btn)

        # Add both panels to the main layout
        self.main_layout.addLayout(self.robot_panel, 1)
        self.main_layout.addLayout(self.model_panel, 2)

        # Add main layout
        layout = QVBoxLayout()
        layout.addLayout(self.main_layout)
        self.setLayout(layout)

        # initial load
        self._refresh_robot_list()


    # -------------------------------------------------------------------------
    # ---------- Handlers for UI actions ----------
    # -------------------------------------------------------------------------
    def _refresh_robot_list(self) -> None:
        """Populate robot combo from robots/ directory."""
        self.robot_list.clear()
        robots_path = os.path.join(self._get_base_path(), "robots")
        if not os.path.exists(robots_path):
            return
        for robot in sorted(os.listdir(robots_path)):
            robot_dir = os.path.join(robots_path, robot)
            if os.path.isdir(robot_dir):
                self.robot_list.addItem(robot)


    def _on_robot_selection(self):
        """When a robot is selected, populate its models."""
        selected_robots = self.robot_list.selectedItems()
        self.model_list.clear()
        self.delete_robot_btn.setEnabled(bool(selected_robots))
        self.delete_model_btn.setEnabled(False)

        if not selected_robots:
            return

        self.robot = selected_robots[0].text()
        models_dir = os.path.join(self._get_base_path(), "robots", self.robot, "models")

        if not os.path.exists(models_dir):
            return

        model_files = []
        for root, _, files in os.walk(models_dir):
            for f in files:
                if f.endswith(".zip"):
                    rel_path = os.path.relpath(os.path.join(root, f), models_dir)
                    model_files.append(rel_path)
        
        def _model_sort_key(rel_path: str):
            m = re.search(rf"{re.escape(self.robot)}_model_(\d+)", rel_path)
            if m:
                return (int(m.group(1)), rel_path.lower())  # orden numérico y luego estable por nombre
            return (float("inf"), rel_path.lower())

        model_files.sort(key=_model_sort_key)

        # Sort alphabetically
        for rel_path in model_files:
            self.model_list.addItem(rel_path)

        self.model_list.itemSelectionChanged.connect(
            lambda: self.delete_model_btn.setEnabled(bool(self.model_list.selectedItems()))
        )


    def _handle_delete_robot(self):
        """Handle deletion of the selected robot and its models."""
        selected_robots = self.robot_list.selectedItems()
        if not selected_robots:
            return

        self.robot = selected_robots[0].text()
        confirm = QMessageBox.question(self, "Confirm Deletion", f"Are you sure you want to delete robot '{self.robot}' and all its models?", QMessageBox.Yes | QMessageBox.No)
        if confirm != QMessageBox.Yes:
            return

        robot_dir = os.path.join(self._get_base_path(), "robots", self.robot)
        try:
            shutil.rmtree(robot_dir)
            logging.info(f"Deleted robot directory: {robot_dir}")
            robot_plugin = os.path.join(self._get_base_path(), "src", "rl_coppelia", "robot_plugins", f"{self.robot}.py")
            os.remove(robot_plugin)
            logging.info(f"Deleted robot plugin: {robot_plugin}")
            self.request_log.emit(f"<span style='color:green;'>Robot '{self.robot}' and all its data have been deleted.</span>")
            self._refresh_robot_list()
            self.model_list.clear()
        except Exception as e:
            logging.error(f"Error deleting robot directory: {e}")
            QMessageBox.critical(self, "Error", f"Failed to delete robot '{self.robot}': {e}")
        self.delete_robot_btn.setEnabled(False)
        self.delete_model_btn.setEnabled(False)


    def _handle_delete_models(self): 
        """Handle deletion of the selected models."""
        selected_models = self.model_list.selectedItems()
        selected_robots = self.robot_list.selectedItems()
        if not selected_models or not selected_robots:
            return

        robot_name = selected_robots[0].text()
        model_names = [item.text() for item in selected_models]
        confirm = QMessageBox.question(self, "Confirm Deletion", f"Are you sure you want to delete the selected model(s) for robot '{robot_name}'?", QMessageBox.Yes | QMessageBox.No)
        if confirm != QMessageBox.Yes:
            return

        models_dir = os.path.join(self._get_base_path(), "robots", robot_name, "models")
        errors = []
        for model_name in model_names:
            model_path = os.path.join(models_dir, model_name)
            try:
                # Remove zip file
                os.remove(model_path)
                logging.info(f"Deleted model file: {model_path}")

                # Remove parent directory if empty
                model_folder = os.path.dirname(model_path)
                if os.path.isdir(model_folder) and not os.listdir(model_folder):
                    os.rmdir(model_folder) 
                logging.info(f"Deleted model folder as it was empty: {model_folder}")

                # Remove train record file for this model
                robot, model_id = utils.extract_robot_and_model_id(model_name)
                csv_path = utils.find_model_record_csv(self._get_base_path(), robot, model_id)

                if csv_path and os.path.exists(csv_path):
                    os.remove(csv_path)
                    logging.info(f"Deleted training record file: {csv_path}")

                # Remove entries from train_records.csv
                records_file = os.path.join(self._get_base_path(), "robots", robot_name, "training_metrics", "train_records.csv")
                first_col_key = f"{robot}_model_{model_id}"
                if records_file and os.path.exists(records_file):
                    removed = utils.remove_row_where_first_col_equals(records_file, first_col_key)
                    print("removed:", removed)
                    if removed:
                        logging.info(f"Removed entries for model '{model_name}' from {records_file}")
                    else:
                        logging.info(f"Failed to remove entries for model '{model_name}' from {records_file}")
                
                # Remove callbacks directory for this model
                callbacks_dir = os.path.join(self._get_base_path(), "robots", robot_name, "callbacks", f"{robot_name}_callbacks_{model_id}")
                if os.path.isdir(callbacks_dir):
                    shutil.rmtree(callbacks_dir)
                    logging.info(f"Deleted callbacks directory: {callbacks_dir}")


                self.request_log.emit(f"<span style='color:red;'> --- Deleted {model_path} model(s) from '{robot_name}'.</span>")
            except Exception as e:
                logging.error(f"Error deleting model file '{model_path}': {e}")
                errors.append(f"Failed to delete model '{model_name}': {e}")

        self._on_robot_selection()