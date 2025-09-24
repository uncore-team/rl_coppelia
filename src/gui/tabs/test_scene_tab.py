"""Test scene tab: encapsulated UI and logic to start testing a custom scene."""

from __future__ import annotations
import logging
import shutil
from typing import Callable, Optional, List, Tuple
import os
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QComboBox, QHBoxLayout, 
    QSizePolicy, QCheckBox, QSpinBox, QPushButton, QMessageBox, QListWidget,
    QAbstractItemView,QLabel, QListWidgetItem, QDialog
)
from PyQt5.QtGui import QPixmap
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd


from gui import dialogs
from gui.services import refresh_model_ids, refresh_lists
from gui.common_ui import create_styled_button, create_icon_button


class TestSceneTab(QWidget):
    """Encapsulated TestScene tab.
    
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
        self.robot = ""
        self._get_base_path = base_path_getter

        self.robot_combo = QComboBox()
        self.robot_combo.addItem("Select a robot...")
        self.robot_combo.model().item(0).setEnabled(False)
        self.robot_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.robot_combo.currentTextChanged.connect(self._on_robot_changed)

        # Model IDs (checkbox list)
        self.model_ids_input = QListWidget()
        self.model_ids_input.setFixedHeight(200)
        self.model_ids_input.setSelectionMode(QAbstractItemView.NoSelection)

        # Scene folder
        self.scene_to_load = QComboBox()
        self.scene_to_load.currentTextChanged.connect(self._on_scene_folder_changed) 
        
        # Actions for scene folder
        self.edit_scene_btn = create_icon_button("Edit selected scene", "../gui/assets/edit_icon.png", self.handle_edit_scene)
        self.delete_scene_btn = create_icon_button("Delete selected scene", "../gui/assets/delete_icon.png", self.handle_delete_scene)

        # Horizontal row for scene folder + actions
        self.scene_folder_row = QHBoxLayout()
        self.scene_folder_row.addWidget(self.scene_to_load)
        self.scene_folder_row.addWidget(self.edit_scene_btn)
        self.scene_folder_row.addWidget(self.delete_scene_btn)

        # Label for scene summary
        self.scene_info_label = QLabel()
        self.scene_info_label.hide()  # Hidden by default

        # Button to show scene preview
        self.view_scene_button = QPushButton("Check scene!")
        self.view_scene_button.setToolTip("Show scene preview")
        self.view_scene_button.clicked.connect(self._handle_show_scene_preview)
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
        self.refresh_all_lists()

    # -------------------------------------------------------------------------
    # Public refresh API
    # -------------------------------------------------------------------------
    def refresh_all_lists(self) -> None:
        """Public hook to refresh robots and scene folders for the selected robot."""
        self._refresh_robot_list()
        self._refresh_scene_folders(self._current_robot())

    # -------------------------------------------------------------------------
    # Internal: population & helpers
    # -------------------------------------------------------------------------
    def _refresh_robot_list(self) -> None:
        refresh_lists(self, self.robot_combo, category="robot")


    def _refresh_scene_folders(self, robot: Optional[str]) -> None:
        """Populate scene folders combo for given robot."""
        if self.robot and not self.robot.startswith("Select"):
            refresh_lists(self, self.scene_to_load, category="scene_configs")
        self._current_scene_csv_path = None


    def _current_robot(self) -> Optional[str]:
        """Return current robot name or None."""
        txt = self.robot_combo.currentText()
        if not txt or txt.startswith("Select"):
            return None
        return txt


    # -------------------------------------------------------------------------
    # Slots
    # -------------------------------------------------------------------------
    def _on_robot_changed(self) -> None:
        """Handle robot change: refresh models and scene folders."""
        self.robot = self._current_robot()
        refresh_model_ids(self, self.robot)
        self._refresh_scene_folders(self.robot)

    def _on_scene_folder_changed(self, folder_name: str) -> None:
        """When scene folder changes, try to read CSV and show summary/preview."""
        if folder_name == "Custom your scene":
            self.robot = self._current_robot()
            if not self.robot or self.robot.startswith("Select"):
                self.request_log.emit("❌ Please select a robot before choosing a scene folder.")
                return

            dialog = dialogs.CustomSceneDialog(self._get_base_path(), self.robot, parent = self)
            if dialog.exec_() == QDialog.Accepted:
                created_folder = dialog.selected_scene_folder
                refresh_lists(self, self.scene_to_load, "scene_configs")
                index = self.scene_to_load.findText(created_folder)
                if index >= 0:
                    self.scene_to_load.setCurrentIndex(index)
            return


        if folder_name in ["Select a scene folder to load...", "Custom scene", "Scene configs directory not found", "No scene configs found"] or not folder_name: 
            self.scene_info_label.hide()
            self.view_scene_button.hide()
            self.scene_info.hide()
            self.edit_scene_btn.hide()
            self.delete_scene_btn.hide()
            self._current_scene_csv_path = None
            return

        base = self._get_base_path()
        scene_dir = os.path.join(base, "robots", self.robot, "scene_configs", folder_name)
        csv_path = self._find_scene_csv(scene_dir)

        if not csv_path:
            self.scene_info_label.setText("❌ No CSV found in scene folder.")
            self.scene_info_label.show()
            self.view_scene_button.hide()
            self._current_scene_csv_path = None
            return

        # Read summary
        try:
            num_targets, num_obstacles = self._count_targets_obstacles(csv_path)
            self.scene_info_label.setText(f"Scene contains: {num_targets} targets, {num_obstacles} obstacles")
            self.scene_info_label.show()
            self.view_scene_button.show()
            self.scene_info.show()
            self._current_scene_csv_path = csv_path

            # Warn if trajs exists
            trajs_path = os.path.join(scene_dir, "trajs")
            if os.path.isdir(trajs_path) and os.listdir(trajs_path):
                self.request_log.emit(
                    f"<span style='color:orange;'> --- ⚠️ {folder_name} contains a 'trajs' dir. Test will overwrite existing trajectories.</span>"
                )
            else:
                self.request_log.emit(
                    f"<span style='color:green;'> --- </span>{folder_name} is ready for testing."
                )
        except Exception as exc:
            logging.warning(f"Error loading scene info: {exc}")
            self.scene_info_label.setText("❌ Error loading scene data")
            self.scene_info_label.show()
            self.view_scene_button.hide()
            self._current_scene_csv_path = None

        # Show/hide edit and delete buttons
        if folder_name and not folder_name.startswith("Select") and folder_name != "Custom scene":
            self.edit_scene_btn.setVisible(True)
            self.delete_scene_btn.setVisible(True)
        else:
            self.edit_scene_btn.setVisible(False)
            self.delete_scene_btn.setVisible(False)

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------
    def _start_test_scene_clicked(self) -> None:
        """Collect parameters and emit signal_start with proper args."""
        self.robot = self._current_robot()
        if not self.robot:
            self.request_log.emit("<span style='color:orange;'>⚠️ Please select a robot name.</span>")
            return

        model_ids = self._selected_model_ids()
        if not model_ids:
            self.request_log.emit("<span style='color:orange;'>⚠️ Please select at least one model ID.</span>")
            return

        scene_folder = self.scene_folder_row.currentText().strip()
        if not scene_folder or scene_folder.startswith("Select"):
            self.request_log.emit("<span style='color:orange;'>⚠️ Please select a scene folder.</span>")
            return

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        args = [
            "uncore_rl", "test_scene",
            "--robot_name", self.robot,
            "--model_ids", *model_ids,
            "--scene_to_load_folder", scene_folder,
            "--iters_per_model", str(self.iters_per_model.value()),
            "--timestamp", str(timestamp),
            "--verbose", str(self.verbose.value()),
        ]
        if self.no_gui.isChecked():
            args.append("--no_gui")

        self.signal_start.emit(args, "Test Scene", self.robot)

    def _selected_model_ids(self) -> List[str]:
        """Return list of checked model IDs from the list widget."""
        ids: List[str] = []
        for i in range(self.model_ids_input.count()):
            item = self.model_ids_input.item(i)
            w = self.model_ids_input.itemWidget(item)
            if not w:
                continue
            from PyQt5.QtWidgets import QCheckBox
            cb = w.findChild(QCheckBox)
            if cb and cb.isChecked():
                ids.append(cb.text().strip())
        return ids

    # -------------------------------------------------------------------------
    # Scene preview helpers
    # -------------------------------------------------------------------------
    def _find_scene_csv(self, scene_dir: str) -> Optional[str]:
        """Find a CSV file that likely contains the scene definition."""
        if not os.path.isdir(scene_dir):
            return None
        for f in os.listdir(scene_dir):
            if f.lower().endswith(".csv") and "scene" in f.lower():
                return os.path.join(scene_dir, f)
        # fallback: first CSV
        for f in os.listdir(scene_dir):
            if f.lower().endswith(".csv"):
                return os.path.join(scene_dir, f)
        return None
    

    def handle_edit_scene(self):
        '''
        Edit the selected scene folder using the CustomSceneDialog.
        '''
        folder_name = self.scene_to_load.currentText().strip()
        if not folder_name or folder_name in ["Select a scene folder to load...", "Custom scene", "Scene configs directory not found", "No scene configs found"]:
            return

        dialog = dialogs.CustomSceneDialog(self._get_base_path(), self.robot, self, edit_mode=True)
        dialog.load_scene(folder_name)  
        if dialog.exec_() == QDialog.Accepted:
            if self.robot and not self.robot.startswith("Select"):
                refresh_lists(self, self.scene_to_load, category="scene_configs")
            self._current_scene_csv_path = None
            idx = self.scene_to_load.findText(folder_name)
            if idx >= 0:
                self.scene_to_load.setCurrentIndex(idx)


    def handle_delete_scene(self):
        '''
        Delete the selected scene folder after user confirmation.
        '''
        folder_name = self.scene_to_load.currentText().strip()
        if not folder_name or folder_name in ["Select a scene folder to load...", "Custom scene"]:
            return

        reply = QMessageBox.question(self, "Delete Scene", f"Are you sure you want to delete scene '{folder_name}'?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            scene_path = os.path.join(self._get_base_path(), "robots", self.robot, "scene_configs", folder_name)
            try:
                shutil.rmtree(scene_path)
                if self.robot and not self.robot.startswith("Select"):
                    refresh_lists(self, self.scene_to_load, category="scene_configs")
                self._current_scene_csv_path = None
                self.scene_to_load.setCurrentIndex(0)
                message = f"Scene folder '{folder_name}' deleted successfully."
                logging.info(message)
                self.request_log.emit(f"<span style='color:green;'> --- {message}</span>")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not delete scene: {e}")


    def _count_targets_obstacles(self, csv_path: str) -> Tuple[int, int]:
        """Count targets and obstacles from a scene CSV."""
        num_targets = 0
        num_obstacles = 0
        
        df = pd.read_csv(csv_path)
        num_targets = (df["type"].str.lower() == "target").sum()
        num_obstacles = (df["type"].str.lower() == "obstacle").sum()
        return num_targets, num_obstacles


    def _handle_show_scene_preview(self) -> None:
        """Generate and display a preview of the scene CSV."""
        if not self._current_scene_csv_path or not os.path.isfile(self._current_scene_csv_path):
            self.request_log.emit("<span style='color:red;'>⚠️ No scene CSV to preview.</span>")
            return
        try:
            img_path = self._plot_scene(self._current_scene_csv_path)
            self._show_image_dialog(img_path, title="Scene Preview")
        except Exception as exc:
            logging.error(f"Could not show scene preview: {exc}")
            self.request_log.emit(f"<span style='color:red;'>⚠️ Failed to show scene preview: {exc}</span>")


    def _plot_scene(self, csv_file: str) -> str:
        """Generate a scene visualization from a CSV and return the image path.

        Args:
            csv_file: Path to the scene CSV file.

        Returns:
            Path to the saved PNG image.
        """
        import pandas as pd

        df = pd.read_csv(csv_file)

        fig, ax = plt.subplots(figsize=(6, 6))
        robot_df = df[df["type"].str.lower() == "robot"]
        targets = df[df["type"].str.lower() == "target"]
        obstacles = df[df["type"].str.lower() == "obstacle"]

        # Obstacles
        for _, row in obstacles.iterrows():
            circle = plt.Circle((row["x"], row["y"]), 0.25 / 2, color="gray", label="Obstacle")
            ax.add_patch(circle)

        # Targets (bullseye-like)
        for label_idx, (_, row) in enumerate(targets.iterrows()):
            x, y = row["x"], row["y"]
            idx = chr(65 + label_idx)  # A, B, C...
            ax.add_patch(Circle((x, y), 0.25, color="blue", alpha=0.3))     # Outer
            ax.add_patch(Circle((x, y), 0.125, color="red", alpha=0.5))     # Middle
            ax.add_patch(Circle((x, y), 0.015, color="yellow", alpha=0.8))  # Inner
            ax.text(x, y - 0.1, idx, fontsize=12, fontweight="bold", ha="center")

        # Robot
        for _, row in robot_df.iterrows():
            circle = plt.Circle((row["x"], row["y"]), 0.35 / 2, color="black", label="Robot", zorder=4)
            ax.add_patch(circle)

            # Orientation triangle if 'theta' present
            theta = row["theta"] if "theta" in row else None
            if theta is not None:
                front_length = 0.15
                side_offset = 0.08
                front = (row["x"] + front_length * np.cos(theta), row["y"] + front_length * np.sin(theta))
                left = (row["x"] + side_offset * np.cos(theta + 2.5), row["y"] + side_offset * np.sin(theta + 2.5))
                right = (row["x"] + side_offset * np.cos(theta - 2.5), row["y"] + side_offset * np.sin(theta - 2.5))
                triangle = plt.Polygon([front, left, right], color="white", zorder=4)
                ax.add_patch(triangle)

        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        # Unique legend
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            ax.legend(unique.values(), unique.keys(), loc="upper right", labelspacing=1.2)
        ax.set_xlim(2.5, -2.5)
        ax.set_ylim(2.5, -2.5)
        plt.tight_layout()

        out_path = os.path.join("/tmp", "scene_preview.png")
        plt.savefig(out_path)
        plt.close(fig)
        return out_path


    def _show_image_dialog(self, img_path: str, title: str = "Preview") -> None:
        """Show a simple modal dialog with an image."""
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        v = QVBoxLayout(dlg)
        lbl = QLabel()
        pix = QPixmap(img_path).scaledToWidth(600, Qt.SmoothTransformation)
        lbl.setPixmap(pix)
        v.addWidget(lbl)
        dlg.exec_()