"""Plot tab: encapsulated UI and logic to start plotting graphs."""

from __future__ import annotations
import logging
from typing import Callable, Optional
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QComboBox,
    QHBoxLayout, QCheckBox, QMessageBox,
    QListWidget, QAbstractItemView, QGridLayout, QLabel
)
from gui import dialogs  
from gui.services import capture_cli_output, refresh_model_ids, refresh_lists
from gui.common_ui import create_info_button, create_styled_button
from rl_coppelia import cli  


class PlotTab(QWidget):
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

        # Form for plot parameters
        form = QFormLayout()

        # Robot selection
        self.robot_combo = QComboBox()
        self.robot_combo.setEditable(False)
        self.robot_combo.currentIndexChanged.connect(self._on_robot_changed)
        
        # Model IDs selection
        self.model_ids_input = QListWidget()
        self.model_ids_input.setFixedHeight(200)
        self.model_ids_input.setSelectionMode(QAbstractItemView.NoSelection)
        self.plot_types_checkboxes = []

        # Grid for plot types
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout()
        self.grid_widget.setLayout(self.grid_layout)

        # Add checkboxes in a grid for selecting plot types
        cols = 2
        for index, plot_type in enumerate(dialogs.PLOT_TYPES):
            checkbox = QCheckBox(plot_type)
            row = index // cols
            col = index % cols
            self.grid_layout.addWidget(checkbox, row, col)
            self.plot_types_checkboxes.append(checkbox)

        for checkbox in self.plot_types_checkboxes:
            checkbox.stateChanged.connect(self.handle_plot_type_change)

        # Plot label and icon
        self.label_with_icon = QWidget()
        self.label_layout = QHBoxLayout()
        self.label_layout.setContentsMargins(0, 0, 0, 0)
        self.label_with_icon.setLayout(self.label_layout)

        self.plot_label = QLabel("Plot Types:")
        self.info_button = create_info_button(self, dialogs.PLOT_TOOLTIPS)

        self.label_layout.addWidget(self.plot_label)
        self.label_layout.addWidget(self.info_button)
        self.label_layout.addStretch()

        
        # Scene to load selection (it will appear just for plot_trajs case)
        self.scene_folder_row = QWidget()
        self.scene_form_layout = QFormLayout()
        self.scene_form_layout.setContentsMargins(0, 0, 0, 0)
        self.scene_form_layout.setHorizontalSpacing(60)
        self.scene_folder_row.setLayout(self.scene_form_layout)

        self.scene_to_load_folder = QComboBox()
        self.scene_form_layout.addRow("Scene Folder:", self.scene_to_load_folder)
        self.scene_folder_row.hide()

        # Build the main layout
        form = QFormLayout()
        form.addRow("Robot Name (required):", self.robot_combo)
        form.addRow("Model IDs (required):", self.model_ids_input)
        form.addRow(self.label_with_icon, self.grid_widget)
        form.addRow(self.scene_folder_row)
        
        # Start button
        self.start_btn = create_styled_button(self,"Generate Plots", self._start_plots_clicked)

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
        self.refresh_all_lists()

    # -------------------------------------------------------------------------
    # Public refresh API
    # -------------------------------------------------------------------------
    def refresh_all_lists(self) -> None:
        """Public hook to refresh robots and scene folders for the selected robot."""
        self._refresh_robot_list()
        self._refresh_scene_folders(self._current_robot())


    def _refresh_robot_list(self) -> None:
        """Populate robot combo from robots/ directory."""
        refresh_lists(self, self.robot_combo, category="robot")


    def _refresh_scene_folders(self, robot: Optional[str]) -> None:
        """Populate scene folders combo for given robot."""
        if self.robot and not self.robot.startswith("Select"):
            refresh_lists(self, self.scene_to_load_folder, category="scene_configs")
        self._current_scene_csv_path = None


    def _on_robot_changed(self) -> None:
        """Handle robot change: refresh models and scene folders."""
        self.robot = self._current_robot()
        refresh_model_ids(self, self.robot)
        self._refresh_scene_folders(self.robot)


    def handle_plot_type_change(self):
        selected_types = [cb.text() for cb in self.plot_types_checkboxes if cb.isChecked()]
        show_scene_input = "plot_scene_trajs" in selected_types
        logging.info(f"Selected types: {selected_types}")

        if show_scene_input:
            self.robot = self.robot_combo.currentText()
            if self.robot != "Select a robot...":
                refresh_lists(self, self.scene_to_load_folder, category="scene_configs")
            self.scene_folder_row.show()
        else:
            self.scene_folder_row.hide()


    # ---------- internals ----------
    def _current_robot(self) -> Optional[str]:
        """Return current robot name or None."""
        txt = self.robot_combo.currentText()
        if not txt or txt.startswith("Select"):
            return None
        return txt

    def _check_selected_models(self) -> list[str]:
        """Return list of selected model IDs."""
        selected = []
        for index in range(self.model_ids_input.count()):
            item = self.model_ids_input.item(index)
            widget = self.model_ids_input.itemWidget(item)
            if widget:
                checkbox = widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    model_id = checkbox.property("model_id")
                    selected.append(model_id)
        return selected
    

    def _check_selected_plot_types(self) -> list[str]:
        """Return list of selected plot types."""
        return [cb.text() for cb in self.plot_types_checkboxes if cb.isChecked()]


    def _start_plots_clicked(self) -> None:
        """Build CLI args and emit start signal."""
        self.robot = self._current_robot()
        if not self.robot:
            QMessageBox.warning(self, "Missing robot", "Please select (or create) a robot name.")
            return
        
        plot_types = self._check_selected_plot_types()
        if not plot_types:
            self.request_log.emit("<span style='color:orange;'>⚠️ Please select at least a plot type.</span>")
            return
    
        model_ids = self._check_selected_models()
        if not model_ids:
            if "plot_scene_trajs" not in plot_types:
                self.request_log.emit("<span style='color:orange;'>⚠️ Please select at least one model ID.</span>")
                return
            else:
                model_ids.append(9999)  # plot_trajs option doesn't need a model id

        # Build args
        args = [
            "plot",
            "--robot_name", self.robot,
            "--model_ids", *map(str, model_ids),
            "--plot_types", *plot_types,
            "--verbose", str(10)
        ]

        if "plot_scene_trajs" in plot_types:  
            scene_folder = self.scene_to_load_folder.currentText()
            if scene_folder and scene_folder != "Select a folder...":
                args.extend(["--scene_to_load_folder", scene_folder])

        logging.info(f"Generating plots with args: {args}")
        output, errors, success = capture_cli_output(cli.main, argv=args)

        if errors:
            self.request_log.emit("<span style='color:red;'> --- ❌ Errors detected during plotting:</span>")
            for err in errors:
                self.request_log.emit(f"<pre>{err}</pre>")
        elif not success:
            self.request_log.emit("<span style='color:red;'> --- ❌ Exception occurred during plotting.</span>")
        else:
            if 9999 in model_ids:
                success_text = f"<span style='color:green;'> --- Plots generated successfully</span>: scene folder {scene_folder}, plot type: {', '.join(plot_types)}."
            else:
                success_text = f"<span style='color:green;'> --- Plots generated successfully</span>: models {model_ids} with plot types: {', '.join(plot_types)}."
            self.request_log.emit(success_text)

            self.signal_start.emit(args, "Train", self.robot)