from datetime import datetime
import os
import subprocess
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QPushButton, 
    QLabel, QLabel, QHBoxLayout, QStackedWidget,QTextEdit, QScrollArea,QProgressBar
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import logging

from common import utils
from pathlib import Path

import pkg_resources
from gui.workers import ProcessThread

from gui.tabs.train_tab import TrainTab
from gui.tabs.test_tab import TestTab
from gui.tabs.plot_tab import PlotTab
from gui.tabs.test_scene_tab import TestSceneTab
from gui.tabs.auto_train_tab import AutoTrainTab
from gui.tabs.manage_tab import ManageTab

from gui.screens.welcome import WelcomeScreen


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.robot = ""
        self.setWindowTitle("UNCORE RL Manager")
        self.setGeometry(200, 200, 500, 300)

        # Get base path
        self._base_path = self._compute_base_path()

        # Central stacked widget for welcome + main UI
        self._stack = QStackedWidget()  
        self.setCentralWidget(self._stack)

        # Page 0: welcome
        self._welcome = WelcomeScreen(self)
        self._welcome.continue_clicked.connect(self._load_main_interface)
        self._stack.addWidget(self._welcome)  # index 0


         # ---------- base path ----------
    def _compute_base_path(self) -> str:
        """Compute project base path (two levels above this file)."""
        expanded_path = os.path.abspath(__file__)
        return str(Path(expanded_path).parents[2])

    def get_base_path(self) -> str:
        """Public getter passed to tabs."""
        return self._base_path


    # ---------- UI scaffold for main window ----------
    def _load_main_interface(self) -> None:
        """Create main layout: tabs + side panel."""
        self.resize(1000, 600)
        page = QWidget()
        main = QHBoxLayout(page)

        # Left: tabs
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        main.addWidget(left_container, stretch=3)

        # Tabs
        self.tabs = QTabWidget()
        left_layout.addWidget(self.tabs)

        # Right: side (logs + processes)
        right_container = QWidget()
        right_container.setLayout(self._build_logs_panel())
        main.addWidget(right_container, stretch=1)

        # Build tabs instances and connect signals
        self.train_tab = TrainTab(self.get_base_path, self)
        self.train_tab.signal_start.connect(self._on_start_requested)
        self.train_tab.request_log.connect(self._append_log)

        self.test_tab = TestTab(self.get_base_path, self)
        self.test_tab.signal_start.connect(self._on_start_requested)
        self.test_tab.request_log.connect(self._append_log)

        self.plot_tab = PlotTab(self.get_base_path, self)
        self.plot_tab.request_log.connect(self._append_log)

        self.test_scene_tab = TestSceneTab(self.get_base_path, self)
        self.test_scene_tab.signal_start.connect(self._on_start_requested)
        self.test_scene_tab.request_log.connect(self._append_log)

        self.auto_train_tab = AutoTrainTab(self.get_base_path, self)
        self.auto_train_tab.signal_start.connect(self._on_start_requested)
        self.auto_train_tab.request_log.connect(self._append_log)

        self.manage_tab = ManageTab(self.get_base_path, self)

        # Add tabs
        self.tabs.addTab(self.train_tab, "Train")
        self.tabs.addTab(self.test_tab, "Test")
        self.tabs.addTab(self.test_scene_tab, "Test scene")
        self.tabs.addTab(self.plot_tab, "Plot")
        self.tabs.addTab(self.auto_train_tab, "Auto Training")
        self.tabs.addTab(self.manage_tab, "Manage")

        # Put page into stack and switch
        self._stack.addWidget(page)  # index 1
        self._stack.setCurrentWidget(page)


    def _build_logs_panel(self):
        """Create logs and processes panel."""
        layout = QVBoxLayout()

        # Logs header
        hdr = QWidget()
        hdr_l = QHBoxLayout(hdr); 
        hdr_l.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Logs"); 
        title.setStyleSheet("font-weight: bold;")
        clear_btn = QPushButton("Clean logs")
        clear_btn.setFixedHeight(22)
        clear_btn.clicked.connect(lambda: self.logs_text.clear())
        hdr_l.addWidget(title); 
        hdr_l.addStretch(); 
        hdr_l.addWidget(clear_btn)
        layout.addWidget(hdr)

        # Logs area
        self.logs_text = QTextEdit(); 
        self.logs_text.setReadOnly(True)
        logs_scroll = QScrollArea(); 
        logs_scroll.setWidgetResizable(True)
        logs_scroll.setWidget(self.logs_text); 
        logs_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(logs_scroll, stretch=1)

        # Processes
        self.processes_label = QLabel("No processes yet")
        layout.addWidget(self.processes_label)

        processes_container_widget = QWidget()
        self.processes_container = QVBoxLayout(processes_container_widget)
        proc_scroll = QScrollArea(); 
        proc_scroll.setWidgetResizable(True)
        proc_scroll.setWidget(processes_container_widget)
        layout.addWidget(proc_scroll, stretch=1)

        return layout

    # ---------- logging ----------
    def _append_log(self, html: str) -> None:
        """Append HTML text to the logs panel."""
        self.logs_text.append(html)


    def _update_processes_label(self) -> None:
        """Update text of the process box according to items count."""
        count = self.processes_container.count()
        self.processes_label.setText("No processes yet" if count == 0 else "Current processes:")


    # ---------- process orchestration ----------
    def _on_start_requested(self, args: list[str], process_type: str, model_name: str) -> None:
        """Handle a tab asking to start a background process."""
        timestamp = args[args.index("--timestamp") + 1]
        title = f"{process_type} - {timestamp}"
        self._run_process(args, process_type, model_name, title)


    def _run_process(self, args: list[str], process_type: str, model_name: str, title: str) -> None:
        """Create ProcessThread, widget, and wire up signals."""
        thread = ProcessThread(args) 
        thread.terminal_title = title

        print(thread.terminal_title)

        # Visual block
        proc_widget = QWidget()
        
        v_layout = QVBoxLayout(proc_widget)
        info_label = QLabel(f"<b>{title}</b>")
        progress_bar = QProgressBar(); 
        progress_bar.setRange(0, 100); 
        progress_bar.setValue(0)
        stop_button = QPushButton("Stop")
        stop_button.clicked.connect(lambda: self._stop_specific_process(thread, proc_widget))
        v_layout.addWidget(info_label); 
        v_layout.addWidget(progress_bar); 
        v_layout.addWidget(stop_button)

        # Save metadata on widget (lightweight)
        proc_widget.process_type = process_type
        proc_widget.model_name = model_name
        proc_widget.timestamp = title

        # Place in UI
        self.processes_container.addWidget(proc_widget)
        self._update_processes_label()

        # Connect signals
        thread.progress_signal.connect(progress_bar.setValue)
        thread.finished_signal.connect(lambda: self._on_process_finished(proc_widget))
        thread.error_signal.connect(lambda msg: self._on_process_error(msg, proc_widget))

        # Start
        thread.start()
        self._append_log(f"<span style='color:green;'> --- </span>{process_type} started with args: <code>{' '.join(args)}</code>")


    def _stop_specific_process(self, process_thread: ProcessThread, process_widget: QWidget) -> None:
        """Stop a specific background process and remove its widget."""
        if process_thread.isRunning():
            process_thread.stop()
            logging.info("Stopping thread...")
        # Try closing terminal if any (best effort, optional)
        try:
            subprocess.run(["wmctrl", "-c", getattr(process_thread, "terminal_title", "")], check=False)
        except Exception:
            pass

        stop_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._append_log(
            f"<span style='color:orange;'> --- ⏹️ Process <b>{getattr(process_widget, 'process_type','?')}</b> "
            f"of <code>{getattr(process_widget, 'model_name','?')}</code> stopped at <b>{stop_time}</b>.</span>"
        )
        process_widget.setParent(None)
        self._update_processes_label()


    def _on_process_finished(self, process_widget: QWidget) -> None:
        """Handle successful process completion."""
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._append_log(
            f"<span style='color:green;'> --- Success: </span> Process <b>{getattr(process_widget,'process_type','?')}</b> "
            f"of <code>{getattr(process_widget,'model_name','?')}</code> finished at <b>{end_time}</b>."
        )
        process_widget.setParent(None)
        self._update_processes_label()


    def _on_process_error(self, error_message: str, process_widget: QWidget) -> None:
        """Handle error during process execution."""
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._append_log(
            f"<span style='color:red;'> --- ❌ Process <b>{getattr(process_widget,'process_type','?')}</b> "
            f"of <code>{getattr(process_widget,'model_name','?')}</code> failed at <b>{end_time}</b>:<br>{error_message}</span>"
        )
        process_widget.setParent(None)
        self._update_processes_label()


def main():
    """Main entry point."""
    utils.logging_config_gui()
    app = QApplication(sys.argv)
    logo_path = pkg_resources.resource_filename("rl_coppelia", "../gui/assets/uncore.png")
    app.setWindowIcon(QIcon(logo_path))

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())