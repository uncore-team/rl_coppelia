"""Train tab: encapsulated UI and logic to start training."""

from __future__ import annotations
from typing import Callable, Optional
import os
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QComboBox, QLineEdit,
    QHBoxLayout, QSizePolicy, QCheckBox, QSpinBox, QPushButton, QMessageBox, QListWidget,
    QAbstractItemView,QLabel
)
from PyQt5.QtGui import QIcon
import pkg_resources

from gui import dialogs, robot_generator  # uses your existing modules
from gui.services import list_dirs, list_json_files
from gui.common_ui import create_styled_button, create_icon_button  # uses your existing styled button


class TestSceneTab(QWidget):
    """Encapsulated Train tab.
    
    Exposes:
        - signal_start(args, process_type, model_name): emitted when user clicks Start.
        - request_log(html): emitted to append messages in main log panel.
    """

    signal_start = pyqtSignal(list, str, str)  # args, process_type, model_name
    request_log = pyqtSignal(str)

    def __init__(self, base_path_getter: Callable[[], str], parent=None):
        """Build the Train tab.

        Args:
            base_path_getter: Callable returning the current project base path.
        """
        super().__init__(parent)
        self._get_base_path = base_path_getter

        self.robot_combo = QComboBox()
        self.robot_combo.addItem("Select a robot...")
        self.robot_combo.model().item(0).setEnabled(False)
        # self.refresh_lists(self, self.test_scene_robot_name_input, category="robot")

        self.robot_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.robot_combo.currentTextChanged.connect(self._update_model_ids)

        # Model IDs (checkbox list)
        self.model_ids_input = QListWidget()
        self.model_ids_input.setFixedHeight(200)
        self.model_ids_input.setSelectionMode(QAbstractItemView.NoSelection)

        # Scene folder
        self.scene_to_load = QComboBox()
        self.scene_to_load.currentTextChanged.connect(self._handle_scene_folder_change) 
        
        # Actions for scene folder
        self.edit_scene_btn = create_icon_button("Edit selected scene", "../gui/assets/edit_icon.png", self.handle_edit_scene)
        self.delete_scene_button = create_icon_button("Delete selected scene", "../gui/assets/delete_icon.png", self.handle_delete_scene)

        # Horizontal row for scene folder + actions
        self.scene_folder_row = QHBoxLayout()
        self.scene_folder_row.addWidget(self.scene_to_load)
        self.scene_folder_row.addWidget(self.edit_scene_btn)
        self.scene_folder_row.addWidget(self.delete_scene_button)

        # Label for scene summary
        self.scene_info_label = QLabel()
        self.scene_info_label.hide()  # Hidden by default

        # Button to show scene preview
        self.view_scene_button = QPushButton("Check scene!")
        self.view_scene_button.setToolTip("Show scene preview")
        self.view_scene_button.clicked.connect(self.handle_show_scene_preview)
        self.view_scene_button.hide()

        # Horizontal row for scene info + button
        self.scene_info = QWidget()
        self.scene_info_layout = QHBoxLayout()
        self.scene_info_layout.setContentsMargins(0, 0, 0, 0)
        self.scene_info.setLayout(self.scene_info_layout)
        self.scene_info_layout.addWidget(self.scene_info_label)
        self.scene_info_layout.addStretch()
        self.scene_info_layout.addWidget(self.view_scene_button)
        self.scene_info.hide() 

        # Iterations per model
        self.iters_per_model = QSpinBox()
        self.iters_per_model.setRange(1, 9999)
        self.iters_per_model.setValue(10)

        # Other options
        self.no_gui = QCheckBox("Disable GUI")
        self.verbose = QSpinBox()
        self.verbose.setRange(-1, 4)
        self.verbose.setValue(3)

        # Build layout
        form = QFormLayout()
        form.addRow("Robot Name (required):", self.robot_combo)
        form.addRow("Model IDs (required):", self.model_ids_input)
        form.addRow("Scene Folder:", self.scene_folder_row)
        form.addRow("", self.scene_info)
        form.addRow("Iterations per model (default 10):", self.iters_per_model)
        form.addRow("Options:", self.no_gui)
        form.addRow("Verbose Level (default: 3):", self.verbose)

        self.start_btn = create_styled_button(self,"Start Testing Scene", self._start_test_scene_clicked)

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
        self.refresh_lists()

    # ---------- public API ----------
    def refresh_lists(self) -> None:
        """Refresh robots and parameter files."""
        base = self._get_base_path()
        # robots
        robots_dir = os.path.join(base, "robots")
        robots = list_dirs(robots_dir)
        self._replace_combo_items(self.robot_combo, robots, keep_special=True)

        # params
        configs_dir = os.path.join(base, "configs")
        files = list_json_files(configs_dir)
        self._replace_combo_items(self.params_combo, files, keep_manual=True)

    # ---------- internals ----------
    def _replace_combo_items(self, combo: QComboBox, items: list[str], keep_special=False, keep_manual=False):
        """Replace items keeping first special options if requested."""
        first = combo.itemText(0) if combo.count() else "Select..."
        specials = []
        if keep_special:
            specials = [combo.itemText(0), combo.itemText(1)] if combo.count() >= 2 else ["Select a robot...", "Create a new one!"]
        if keep_manual:
            # index 0 = Select..., index 1 = Manual parameters
            specials = [combo.itemText(0), combo.itemText(1)] if combo.count() >= 2 else ["Select a configuration file...", "Manual parameters"]

        combo.blockSignals(True)
        combo.clear()
        if keep_special or keep_manual:
            for s in specials:
                combo.addItem(s)
                if s == specials[0]:
                    combo.model().item(0).setEnabled(False)
        else:
            combo.addItem(first)
            combo.model().item(0).setEnabled(False)
        for it in items:
            combo.addItem(it)
        combo.blockSignals(False)

    def _handle_robot_selection(self, text: str) -> None:
        """Show/hide new robot input and autofill scene path."""
        is_custom = text == "Create a new one!"
        self.new_robot_label.setVisible(is_custom)
        if not is_custom:
            self._update_scene_from_robot()

    def _update_scene_from_robot(self) -> None:
        """Auto-fill scene path for selected robot."""
        robot = self._current_robot_name()
        if not robot:
            self.scene_input.clear()
            self.scene_input.setStyleSheet("")
            return
        base = self._get_base_path()
        scene_path = os.path.join(base, "scenes", f"{robot}_scene.ttt")
        self.scene_input.setText(scene_path)
        self._validate_scene()

    def _validate_scene(self) -> None:
        """Keep a light validation feedback for scene path."""
        path = self.scene_input.text().strip()
        if not path:
            self.scene_input.setStyleSheet("")
            self.scene_input.setToolTip("")
            return
        if not os.path.isfile(path):
            self.scene_input.setStyleSheet("background-color: #fff8c4;")
            self.scene_input.setToolTip("Scene file does not exist.")
            self.request_log.emit(f"<span style='color:orange;'> --- ⚠️ Scene file not found: {path}</span>")
        else:
            self.scene_input.setStyleSheet("")
            self.scene_input.setToolTip("")
            self.request_log.emit(f"<span style='color:green;'> --- </span>Scene file found: {path}")

    def _handle_params_selection(self, text: str) -> None:
        """React to selection; show edit button for concrete file."""
        if text == "Manual parameters":
            # keep your manual params dialog in dialogs module, if any
            self.request_log.emit("<span style='color:gray;'> --- Manual parameters requested (dialog not implemented here).</span>")
            self.edit_params_btn.setVisible(False)
            return

        # show edit if is a file (not placeholder)
        visible = bool(text and not text.startswith("Select"))
        self.edit_params_btn.setVisible(visible)

    def _open_edit_params_dialog(self) -> None:
        """Open your existing EditParamsDialog (if available)."""
        text = self.params_combo.currentText()
        if not text or text in ("Select a configuration file...", "Manual parameters"):
            return
        # Aquí puedes invocar tu dialogs.EditParamsDialog si ya lo tienes construido.

    def _start_test_scene_clicked(self) -> None:
        """Build CLI args and emit start signal."""
        robot = self._current_robot_name()
        if not robot:
            QMessageBox.warning(self, "Missing robot", "Please select (or create) a robot name.")
            return

        # If creating new one, scaffold env + plugin
        if self.robot_combo.currentText() == "Create a new one!":
            name = self.new_robot_label.text().strip()
            if not name:
                QMessageBox.warning(self, "Missing name", "Please enter the new robot name.")
                return
            # NewEnvDialog to gather obs space
            dlg = dialogs.NewEnvDialog(self, robot_name=name)
            exec_fn = getattr(dlg, "exec", None) or getattr(dlg, "exec_", None)
            if exec_fn() != dlg.Accepted:
                return
            try:
                spec = dlg.get_spec()
                env_path, plugin_path = robot_generator.create_robot_env_and_plugin(self._get_base_path(), name, spec)
                self.request_log.emit(
                    f"<span style='color:green;'> --- </span>Env created: <code>{env_path}</code><br>"
                    f"<span style='color:green;'> --- </span>Plugin created: <code>{plugin_path}</code>"
                )
                # refresh robots and select it
                self.refresh_lists()
                idx = self.robot_combo.findText(name)
                if idx >= 0:
                    self.robot_combo.setCurrentIndex(idx)
                self._update_scene_from_robot()
                robot = name
            except Exception as exc:
                QMessageBox.critical(self, "Generation error", f"Failed to create env/plugin: {exc}")
                return

        # Build args
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        args = [
            "rl_coppelia", "train",
            "--robot_name", robot,
            "--timestamp", str(timestamp),
            "--verbose", str(self.verbose.value()),
        ]

        params_filename = self.params_combo.currentText().split()[0] if self.params_combo.currentText() else ""
        if params_filename and not params_filename.startswith("Select"):
            args += ["--params_file", os.path.join(self._get_base_path(), "configs", params_filename)]
        if self.dis_parallel.isChecked():
            args.append("--dis_parallel_mode")
        if self.no_gui.isChecked():
            args.append("--no_gui")

        self.signal_start.emit(args, "Train", robot)

    def _current_robot_name(self) -> Optional[str]:
        """Return current robot or None."""
        text = self.robot_combo.currentText()
        if text == "Create a new one!" or text.startswith("Select"):
            # new robot name input
            name = self.new_robot_label.text().strip()
            return name if name else None
        return text if text else None