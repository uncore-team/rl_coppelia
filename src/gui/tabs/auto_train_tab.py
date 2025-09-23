"""Train tab: encapsulated UI and logic to start training."""

from __future__ import annotations
from typing import Callable, Optional
import os
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QComboBox, QLineEdit,
    QHBoxLayout, QSizePolicy, QCheckBox, QSpinBox, QPushButton, QMessageBox
)
from PyQt5.QtGui import QIcon
import pkg_resources

from gui import dialogs, robot_generator  # uses your existing modules
from gui.services import list_dirs, list_json_files
from gui.common_ui import create_styled_button  # uses your existing styled button


class AutoTrainTab(QWidget):
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
        self.robot_combo.addItem("Create a new one!")
        self.robot_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.robot_combo.currentTextChanged.connect(self._handle_robot_selection)

        self.new_robot_label = QLineEdit()
        self.new_robot_label.setPlaceholderText("Introduce name")
        self.new_robot_label.setMaximumWidth(220)
        self.new_robot_label.hide()

        row = QWidget()
        row_l = QHBoxLayout(row)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.addWidget(self.robot_combo)
        row_l.addWidget(self.new_robot_label)
        row_l.addStretch()

        self.scene_input = QLineEdit()
        self.scene_input.setPlaceholderText("Enter scene path (optional)")
        self.scene_input.editingFinished.connect(self._validate_scene)

        self.params_combo = QComboBox()
        self.params_combo.addItem("Select a configuration file...")
        self.params_combo.model().item(0).setEnabled(False)
        self.params_combo.addItem("Manual parameters")
        self.params_combo.currentTextChanged.connect(self._handle_params_selection)

        self.edit_params_btn = QPushButton()
        self.edit_params_btn.setIcon(QIcon(pkg_resources.resource_filename("rl_coppelia", "assets/gear_icon.png")))
        self.edit_params_btn.setFixedSize(24, 24)
        self.edit_params_btn.setToolTip("Edit selected parameter file")
        self.edit_params_btn.setVisible(False)
        self.edit_params_btn.clicked.connect(self._open_edit_params_dialog)

        params_row = QWidget()
        pr_l = QHBoxLayout(params_row)
        pr_l.setContentsMargins(0, 0, 0, 0)
        pr_l.addWidget(self.params_combo)
        pr_l.addWidget(self.edit_params_btn)

        self.dis_parallel = QCheckBox("Disable Parallel Mode")
        self.no_gui = QCheckBox("Disable GUI")

        self.verbose = QSpinBox()
        self.verbose.setRange(-1, 4)
        self.verbose.setValue(3)

        form = QFormLayout()
        form.addRow("Robot Name (required):", row)
        form.addRow("Scene Path (optional):", self.scene_input)
        form.addRow("Params File:", params_row)
        form.addRow("Options:", self.dis_parallel)
        form.addRow("", self.no_gui)
        form.addRow("Verbose Level (default: 1):", self.verbose)

        self.start_btn = create_styled_button(self,"Start Training", self._start_train_clicked)

        layout = QVBoxLayout(self)
        layout.addLayout(form)

        v = QVBoxLayout()
        from PyQt5.QtWidgets import QHBoxLayout as H
        v.addStretch()
        h = H(); h.addStretch(); h.addWidget(self.start_btn); h.addStretch()
        v.addLayout(h); v.addStretch()
        layout.addLayout(v)

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

    def _start_train_clicked(self) -> None:
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