"""Test tab: encapsulated UI and logic to start testing a trained agent."""

from __future__ import annotations
import re
from typing import Callable
import os
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit,
    QHBoxLayout, QCheckBox, QSpinBox, QPushButton, QMessageBox,
    QFileDialog
)

from gui.services import get_rl_coppelia_path_from_bashrc, remove_zip_extension
from gui.common_ui import create_styled_button  # uses your existing styled button


class TestTab(QWidget):
    """Encapsulated Test tab.

    Exposes:
        - signal_start(args, process_type, model_name)
        - request_log(html)
    """
    signal_start = pyqtSignal(list, str, str)
    request_log = pyqtSignal(str)

    def __init__(self, base_path_getter: Callable[[], str], parent=None):
        super().__init__(parent)
        self._get_base_path = base_path_getter

        self.model_zip_input = QLineEdit()
        self.model_zip_input.setPlaceholderText("Select a ZIP file...")
        self.model_zip_input.textChanged.connect(self._on_model_name_changed)

        self.browse_zip = QPushButton("Browse ZIP")
        self.browse_zip.clicked.connect(self._browse_zip_file)

        zip_row = QWidget()
        zip_l = QHBoxLayout(zip_row)
        zip_l.setContentsMargins(0, 0, 0, 0)
        zip_l.addWidget(self.model_zip_input)
        zip_l.addWidget(self.browse_zip)

        self.robot_input = QLineEdit()
        self.robot_input.setPlaceholderText("Enter robot name (optional)")

        self.scene_input = QLineEdit()
        self.scene_input.setPlaceholderText("Enter scene path (optional)")

        self.save_scene = QCheckBox("Save Scene")
        self.save_traj = QCheckBox("Save Trajectory")
        self.dis_parallel = QCheckBox("Disable Parallel Mode")
        self.no_gui = QCheckBox("Disable GUI")

        self.params_file_input = QLineEdit()
        self.params_file_input.setPlaceholderText("Enter params file path (optional)")

        self.iterations = QSpinBox()
        self.iterations.setRange(1, 1000)
        self.iterations.setValue(50)

        self.verbose = QSpinBox()
        self.verbose.setRange(-1, 4)
        self.verbose.setValue(3)

        form = QFormLayout()
        form.addRow("Model ZIP File (required):", zip_row)
        form.addRow("Robot Name (optional):", self.robot_input)
        form.addRow("Scene Path (optional):", self.scene_input)
        form.addRow("Options:", self.save_scene)
        form.addRow("", self.save_traj)
        form.addRow("", self.dis_parallel)
        form.addRow("", self.no_gui)
        form.addRow("Params File (optional):", self.params_file_input)
        form.addRow("Iterations (default: 50):", self.iterations)
        form.addRow("Verbose Level (default: 1):", self.verbose)

        self.start_btn = create_styled_button(self,"Start Testing", self._start_test_clicked)

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


    # ---------- internals ----------
    def _browse_zip_file(self):
        """Open a file dialog to select a ZIP file."""
        start_path = get_rl_coppelia_path_from_bashrc(self._get_base_path)
        file_path, _ = QFileDialog.getOpenFileName(self, "Select ZIP File", start_path, "ZIP Files (*.zip)")
        if file_path:
            self.model_zip_input.setText(file_path)

    def _on_model_name_changed(self, text: str) -> None:
        """Infer robot and params path from model zip naming."""
        if not text:
            self.robot_input.clear()
            self.scene_input.clear()
            self.params_file_input.clear()
            return

        # Expect basename like: <robot>_model_<id>_last.zip
        bn = os.path.basename(text)
        m = re.match(r"(?P<robot>[^_]+)_model_(?P<exp>\d+)_last\.zip", bn)
        if not m:
            return

        robot = m.group("robot")
        self.robot_input.setText(robot)

        # scene
        base = self._get_base_path()
        scene = os.path.join(base, "scenes", f"{robot}_scene.ttt")
        self.scene_input.setText(scene)

        # params file (best effort)
        params_dir = os.path.join(base, "robots", robot, "parameters_used")
        # no lookup here to keep it light; user can fill manually

    def _start_test_clicked(self):
        """Build args and emit start."""
        model_zip = self.model_zip_input.text().strip()
        if not model_zip:
            QMessageBox.warning(self, "Missing model", "Please select a model ZIP.")
            return

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        args = [
            "uncore_rl", "test",
            "--model_name", remove_zip_extension(self, model_zip),
            "--iterations", str(self.iterations.value()),
            "--timestamp", str(timestamp),
            "--verbose", str(self.verbose.value()),
        ]
        if self.save_traj.isChecked():
            args.append("--save_traj")
        if self.params_file_input.text():
            args += ["--params_file", self.params_file_input.text().strip()]
        if self.robot_input.text():
            args += ["--robot_name", self.robot_input.text().strip()]
        if self.scene_input.text():
            args += ["--scene_path", self.scene_input.text().strip()]
        if self.dis_parallel.isChecked():
            args.append("--dis_parallel_mode")
        if self.no_gui.isChecked():
            args.append("--no_gui")

        model_name = os.path.basename(model_zip)
        self.signal_start.emit(args, "Test", model_name)