"""Auto train tab: encapsulated UI and logic to start auto training."""

from __future__ import annotations
from datetime import datetime
import glob
import logging
import os
from typing import Callable, Optional
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QComboBox,
    QHBoxLayout, QCheckBox, QMessageBox,
    QListWidget, QAbstractItemView, QGridLayout, QLabel, QSpinBox
)
from gui import dialogs  
from gui.services import refresh_lists
from gui.common_ui import create_styled_button
from rl_coppelia import cli  


class AutoTrainTab(QWidget):
    """Encapsulated AutoTrain tab.
    
    Exposes:
        - signal_start(args, process_type, model_name): emitted when user clicks Start.
        - request_log(html): emitted to append messages in main log panel.
    """
    signal_start = pyqtSignal(list, str, str)  # args, process_type, model_name
    request_log = pyqtSignal(str)

    def __init__(self, base_path_getter: Callable[[], str], parent=None):
        """Build the AutoTrain tab.

        Args:
            base_path_getter: Callable returning the current project base path.
        """
        super().__init__(parent)
        self._get_base_path = base_path_getter

        # Robot selection
        self.robot_combo = QComboBox()
        self.robot_combo.addItem("Select a robot...")
        self.robot_combo.model().item(0).setEnabled(False)
        self.robot_combo.currentIndexChanged.connect(self._on_robot_changed)
        
        # Session Name (required)
        self.session_name_input = QComboBox()
        self.session_name_input.currentTextChanged.connect(self._handle_session_name_change)

        # Disable parallel mode (optional)
        self.disable_parallel_cb = QCheckBox("Disable Parallel Mode")

        # Max workers (optional, only relevant if parallel mode is active)
        self.max_workers_input = QSpinBox()
        self.max_workers_input.setRange(1, 10)
        self.max_workers_input.setValue(3)

        # Verbose
        self.verbose = QSpinBox()
        self.verbose.setRange(-1, 4)
        self.verbose.setValue(3)

        # Build the main layout
        form = QFormLayout()
        form.addRow("Robot Name (required):", self.robot_combo)
        form.addRow("Session Name (required):", self.session_name_input)
        form.addRow("Options: ", self.disable_parallel_cb)
        form.addRow("Max Workers (default: 3):", self.max_workers_input)
        form.addRow("Verbose Level (default: 1):", self.verbose)
        
        # Start button
        self.start_btn = create_styled_button(self,"Start Auto Training", self._start_auto_train_clicked)

        # Create main layout
        layout = QVBoxLayout(self)
        
        # Centered (horizontally and vertically) button layout        
        button_layout = QVBoxLayout()
        button_layout.addStretch()

        centered_h = QHBoxLayout()
        centered_h.addStretch()
        centered_h.addWidget(self.start_btn)
        centered_h.addStretch()

        button_layout.addLayout(centered_h)
        button_layout.addStretch()

        # Add everything to the main layout
        layout.addLayout(form)
        layout.addLayout(button_layout)

        # initial load
        self._refresh_robot_list()

    # -------------------------------------------------------------------------
    # ---------- UI components ----------
    # -------------------------------------------------------------------------

    def _refresh_robot_list(self) -> None:
        """Populate robot combo from robots/ directory."""
        refresh_lists(self, self.robot_combo, category="robot")


    def _on_robot_changed(self) -> None:
        """Handle robot change: refresh models and scene folders."""
        self.robot = self._current_robot()
        if self.robot and not self.robot.startswith("Select"):
            refresh_lists(self, self.session_name_input, category="session_folders")
    
    def _handle_session_name_change(self):
        """Check if a session folder for auto training exists, and if it contains json files."""
        if not (self.session_name_input.currentText().strip().startswith("Select") or self.session_name_input.currentText().strip().startswith("Scene")):
            print("hey")
            session_name = self.session_name_input.currentText()

            # Get the directory containing the parameter files for the session.
            session_dir = os.path.join(self._get_base_path(), "robots", self.robot, "auto_trainings", session_name)
                
            # Create the directory if it doesn't exist
            os.makedirs(session_dir, exist_ok=True)

            # Check if the directory is empty
            if not os.listdir(session_dir):
                warning = f"ERROR: The directory {session_dir} is empty. Please add the desired param.json files for training."
                logging.critical(warning)
                self.request_log.emit(f"<span style='color:red;'> --- {warning}</span>")
                return
            else:
                # Search all the json files inside the provided folder.
                param_files = glob.glob(os.path.join(session_dir, "*.json"))
                message = f"Found {len(param_files)} parameter files in {session_dir}."
                logging.info(message)
                self.request_log.emit(f"<span style='color:green;'> --- </span>{message}")
    

    # ---------- internals ----------
    def _current_robot(self) -> Optional[str]:
        """Return current robot name or None."""
        txt = self.robot_combo.currentText()
        if not txt or txt.startswith("Select"):
            return None
        return txt


    def _start_auto_train_clicked(self) -> None:
        """Build CLI args and emit start signal."""
        self.robot = self._current_robot()
        if not self.robot:
            QMessageBox.warning(self, "Missing robot", "Please select a robot name.")
            return

        session_name = self.session_name_input.currentText().strip()
        if not session_name:
            self.request_log.emit("<span style='color:orange;'>⚠️ Please provide a session name.</span>")
            return

        dis_parallel = self.disable_parallel_cb.isChecked()
        max_workers = self.max_workers_input.value()
        verbose = self.verbose.value()

        process_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Build args
        args = [
            "uncore_rl", "auto_training",
            "--session_name", session_name,
            "--robot_name", self.robot,
            "--max_workers", str(max_workers),
            "--timestamp", str(process_timestamp),
            "--verbose", str(verbose)
        ]
        if dis_parallel:
            args.append("--dis_parallel_mode")

        logging.info(f"Starting auto training with args: {args}")
        self.signal_start.emit(args, "AutoTrain", self.robot)