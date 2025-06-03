from datetime import datetime
import os
import re
import subprocess
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QLabel, QCheckBox, QSpinBox, QFileDialog
)
from PyQt5.QtCore import Qt
import logging
from common import utils
from rl_coppelia import train, test, plot, auto_training, auto_testing, retrain, sat_training
from pathlib import Path
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QProgressBar, QHBoxLayout, QTextEdit, QSizePolicy, QScrollArea
from PyQt5.QtGui import QIcon


class TestThread(QThread):
    progress_signal = pyqtSignal(int)  # Señal para actualizar la barra de progreso
    finished_signal = pyqtSignal()    # Señal para indicar que el test ha terminado
    error_signal = pyqtSignal(str)    # Señal para manejar errores

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.terminal_title = ""
        self.was_stopped_manually = False

    def stop(self):
        self.was_stopped_manually = True
        if hasattr(self, 'process') and self.process and self.process.poll() is None:
            self.process.terminate()  # O self.process.kill() si quieres forzar

    def run(self):
        """Ejecuta el test en un hilo separado."""
        try:
            self.was_stopped_manually = False  # Reset flag

            # Ejecutar el comando usando subprocess
            self.process = subprocess.Popen(
                self.args,
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True
            )

            # Leer la salida en tiempo real para actualizar la barra de progreso
            for line in self.process.stdout:
                print(line.strip())  # Para depuración en consola
                if "Testing Episodes" in line:
                    try:
                        match = re.search(r"(\d+)%", line)
                        if match:
                            progress = int(match.group(1))
                            self.progress_signal.emit(progress)
                        else:
                            logging.warning(f"Could not parse progress from line: {line.strip()}")
                    except ValueError:
                        logging.warning(f"Could not parse progress from line: {line.strip()}")

            self.process.wait()

            # Evitar emitir errores si fue parada manual
            if self.was_stopped_manually:
                return

            if self.process.returncode != 0:
                self.error_signal.emit("Process returned non-zero exit code.")
            else:
                self.finished_signal.emit()

        except Exception as e:
            self.error_signal.emit(str(e))



class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL Coppelia Manager")
        self.setGeometry(100, 100, 1000, 600)

        expanded_path = os.path.abspath(__file__)
        self.base_path = str(Path(expanded_path).parents[2])
        logging.info(f"Base path configured as: {self.base_path}")
        self.model_name = ""
        self.robot_name = ""
        self.experiment_id = 0

        logo_path = os.path.join(self.base_path, "rl_coppelia", "assets", "uncore.png")
        self.setWindowIcon(QIcon(logo_path))

        # Widget central con layout horizontal
        main_layout = QHBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Sección izquierda: tabs existentes
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_train_tab(), "Train")
        self.tabs.addTab(self.create_test_tab(), "Test")
        self.tabs.addTab(self.create_plot_tab(), "Plot")
        self.tabs.addTab(self.create_auto_training_tab(), "Auto Training")
        self.tabs.addTab(self.create_auto_testing_tab(), "Auto Testing")
        self.tabs.addTab(self.create_retrain_tab(), "Retrain")
        main_layout.addWidget(self.tabs, stretch=3)

        # Sección derecha: panel lateral con logs y procesos activos
        self.side_panel = self.create_side_panel()
        main_layout.addLayout(self.side_panel, stretch=1)


    def update_processes_label(self):
        """Actualizar el texto de procesos según cuántos haya."""
        count = self.processes_container.count()
        if count == 0:
            self.processes_label.setText("No processes yet")
        else:
            self.processes_label.setText("Current processes:")


    def create_side_panel(self):
        """Creates the side panel with scrollable logs and a scrollable process list."""
        from PyQt5.QtWidgets import QScrollArea

        layout = QVBoxLayout()

        # Logs
        layout.addWidget(QLabel("Logs"))

        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setLineWrapMode(QTextEdit.NoWrap)

        logs_scroll = QScrollArea()
        logs_scroll.setWidgetResizable(True)
        logs_scroll.setWidget(self.logs_text)
        logs_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        layout.addWidget(logs_scroll, stretch=1)  # Ocupa la mitad superior

        # Contenedor de procesos
        self.processes_label = QLabel("No processes yet")  # Inicia sin procesos
        layout.addWidget(self.processes_label)

        process_scroll_content = QWidget()
        self.processes_container = QVBoxLayout()
        process_scroll_content.setLayout(self.processes_container)

        process_scroll = QScrollArea()
        process_scroll.setWidgetResizable(True)
        process_scroll.setWidget(process_scroll_content)

        layout.addWidget(process_scroll, stretch=1)  # Ocupa la otra mitad

        return layout



    def update_progress_bar(self, value):
        """Actualizar la barra de progreso."""
        self.progress_bar.setValue(value)

    def update_test_inputs_when_model_name_introduced(self):
        """Update the robot name based on the selected model name."""
        self.model_name = self.test_model_name_input.text()
        if self.model_name:
            # Extract the robot name from the model name
            parts = self.model_name.split("/")
            parts = parts[-1].split("_")  # Split by "_model_" to get the robot name
            self.robot_name = parts[0] 
            self.experiment_id = parts[2]

            if len(parts) > 1:
                # Write robot name to the input field
                robot_name = parts[0]  # The part before "_model_"
                self.test_robot_name_input.setText(robot_name)

                # Construct the scene file path
                scene_path = os.path.join(self.base_path, "scenes", f"{parts[0]}_scene.ttt")
                # Check if the scene file exists
                if not os.path.exists(scene_path):
                    warning_message = f"WARNING: {scene_path} does not exist. Please check the model name."
                    logging.warning(warning_message)
                    self.test_scene_path_input.setText(warning_message)
                else:
                    logging.info(f"Scene file {scene_path} found for robot {robot_name}.")
                    self.test_scene_path_input.setText(scene_path)

                # Construct the params file path
                params_file_path = os.path.join(self.base_path, "robots", robot_name, "parameters_used", f"params_file_model_{self.experiment_id}.json")
                # Check if the params file exists
                if not os.path.exists(params_file_path):
                    warning_message = f"WARNING: {params_file_path} does not exist. Please check the model name."
                    logging.warning(warning_message)
                    self.test_params_file_input.setText(warning_message)
                else:
                    logging.info(f"Params file {params_file_path} found for robot {robot_name}.")
                    self.test_params_file_input.setText(params_file_path)
                
            else:
                self.test_robot_name_input.setText("")  # Clear if format is invalid
                self.test_scene_path_input.setText("")
                self.test_params_file_input.setText("")
        else:
            self.test_robot_name_input.setText("")  # Clear if no model name is provided
            self.test_scene_path_input.setText("")
            self.test_params_file_input.setText("")


    def get_rl_coppelia_path_from_bashrc(self):
        """Retrieve the rl_coppelia path from ~/.bashrc."""
        bashrc_path = os.path.expanduser("~/.bashrc")
        if not os.path.exists(bashrc_path):
            return None
        
        try:
            with open(bashrc_path, "r") as bashrc:
                content = bashrc.read()
                
            # Buscar en PYTHONPATH
            pythonpath_matches = re.findall(r'(?:export\s+)?PYTHONPATH[^=]*=(.+)', content)
            for match in pythonpath_matches:
                # Limpiar la línea (remover comillas, espacios, etc.)
                clean_match = match.strip().strip('"').strip("'")
                paths = clean_match.split(":")
                for path in paths:
                    path = path.strip()
                    if path and "rl_coppelia" in path:
                        # Expandir variables de entorno si las hay
                        expanded_path = os.path.expandvars(path)
                        if os.path.exists(expanded_path):
                            base_path = str(Path(expanded_path).parents[1])  # Get the parent directory of rl_coppelia
                            if base_path != self.base_path:
                                self.base_path = base_path  # Update the base path
                                logging.warning(f"Found rl_coppelia path in PYTHONPATH: {base_path}, but it does not match the expected base path: {self.base_path}")
                            return base_path  
            
            # Buscar en PATH si no se encontró en PYTHONPATH
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
                            if base_path != self.base_path:
                                self.base_path = base_path
                                logging.warning(f"Found rl_coppelia path in PYTHONPATH: {base_path}, but it does not match the expected base path: {self.base_path}")
                            return base_path
                            
        except Exception as e:
            print(f"Error reading .bashrc: {e}")
            
        return None
    

    def browse_zip_file(self):
        """Open a file dialog to select a ZIP file, starting in the rl_coppelia directory."""
        rl_coppelia_path = self.get_rl_coppelia_path_from_bashrc()
        
        # Si encontramos la ruta de rl_coppelia, usarla como directorio inicial
        if rl_coppelia_path and os.path.exists(rl_coppelia_path):
            start_path = rl_coppelia_path
            print(f"Starting file dialog in rl_coppelia directory: {start_path}")
        else:
            start_path = os.path.expanduser("~")
            print(f"rl_coppelia path not found, starting in home directory: {start_path}")
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select ZIP File", 
            start_path, 
            "ZIP Files (*.zip)"
        )
        
        if file_path:
            self.test_model_name_input.setText(file_path)
            print(f"Selected file: {file_path}")

    def remove_zip_extension(self, file_path):
        """Remove the .zip extension from the file name if it exists."""
        base_name, extension = os.path.splitext(file_path)
        if extension.lower() == ".zip":
            return base_name
        return file_path


    def create_train_tab(self):
        """Tab for training configuration."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Form for training parameters
        form = QFormLayout()
        self.robot_name_input = QLineEdit()
        self.scene_path_input = QLineEdit()
        self.params_file_input = QLineEdit()
        self.verbose_input = QSpinBox()
        self.verbose_input.setRange(0, 2)

        form.addRow("Robot Name:", self.robot_name_input)
        form.addRow("Scene Path:", self.scene_path_input)
        form.addRow("Params File:", self.params_file_input)
        form.addRow("Verbose Level:", self.verbose_input)

        # Buttons
        browse_button = QPushButton("Browse Scene")
        browse_button.clicked.connect(self.browse_scene)
        form.addRow("", browse_button)

        train_button = QPushButton("Start Training")
        train_button.clicked.connect(self.start_training)
        layout.addLayout(form)
        layout.addWidget(train_button)

        tab.setLayout(layout)
        return tab

    def create_test_tab(self):
        """Tab for testing configuration."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Form for testing parameters
        form = QFormLayout()

        # Model name (required)
        self.test_model_name_input = QLineEdit()
        self.test_model_name_input.setPlaceholderText("Select a ZIP file...")
        browse_zip_button = QPushButton("Browse ZIP")
        browse_zip_button.clicked.connect(self.browse_zip_file)
        self.test_model_name_input.textChanged.connect(self.update_test_inputs_when_model_name_introduced)  # Connect text change to update_robot_name


        # Robot name (optional)
        self.test_robot_name_input = QLineEdit()
        self.test_robot_name_input.setPlaceholderText("Enter robot name (default: burgerBot)")

        # Scene path (optional)
        self.test_scene_path_input = QLineEdit()
        self.test_scene_path_input.setPlaceholderText("Enter scene path (optional)")

        # Save scene (optional)
        self.save_scene_checkbox = QCheckBox("Save Scene")

        # Save trajectory (optional)
        self.save_traj_checkbox = QCheckBox("Save Trajectory")

        # Disable parallel mode (optional)
        self.dis_parallel_mode_checkbox = QCheckBox("Disable Parallel Mode")

        # Disable GUI (optional)
        self.no_gui_checkbox = QCheckBox("Disable GUI")

        # Params file (optional)
        self.test_params_file_input = QLineEdit()
        self.test_params_file_input.setPlaceholderText("Enter path to params file (optional)")

        # Iterations (optional, default: 50)
        self.test_iterations_input = QSpinBox()
        self.test_iterations_input.setRange(1, 1000)
        self.test_iterations_input.setValue(5)

        # Verbose level (optional, default: 3)
        self.verbose_input = QSpinBox()
        self.verbose_input.setRange(-1, 4)
        self.verbose_input.setValue(3)

       

        

        # Add fields to the form
        form.addRow("Model ZIP File (required):", self.test_model_name_input)
        form.addRow("", browse_zip_button)
        form.addRow("Robot Name (optional):", self.test_robot_name_input)
        form.addRow("Scene Path (optional):", self.test_scene_path_input)
        form.addRow("", self.save_scene_checkbox)
        form.addRow("", self.save_traj_checkbox)
        form.addRow("", self.dis_parallel_mode_checkbox)
        form.addRow("", self.no_gui_checkbox)
        form.addRow("Params File (optional):", self.test_params_file_input)
        form.addRow("Iterations (default: 400):", self.test_iterations_input)
        form.addRow("Verbose Level (default: 1):", self.verbose_input)

        # Buttons
        test_button = QPushButton("Start Testing")
        test_button.clicked.connect(self.start_testing)
        test_button.setFixedHeight(40)  # Altura mayor
        test_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # No se expande horizontalmente
        test_button.setStyleSheet("""
            QPushButton {
                padding: 6px 16px;
                font-size: 14px;
                border-radius: 8px;
                background-color: #007ACC;
                color: white;
            }
            QPushButton:disabled {
                background-color: #888;
                color: #ccc;
            }
        """)
        # Layout centrado (horizontal + vertical)
        center_button_layout = QVBoxLayout()
        center_button_layout.addStretch()

        h_center = QHBoxLayout()
        h_center.addStretch()
        h_center.addWidget(test_button)
        h_center.addStretch()

        center_button_layout.addLayout(h_center)
        center_button_layout.addStretch()

        # Ensamblar la pestaña
        layout.addLayout(form)
        layout.addLayout(center_button_layout)

        tab.setLayout(layout)
        return tab

    

    def create_plot_tab(self):
        """Tab for plot configuration."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Form for plot parameters
        form = QFormLayout()
        self.plot_robot_name_input = QLineEdit()
        self.plot_model_ids_input = QLineEdit()
        self.plot_types_input = QLineEdit()

        form.addRow("Robot Name:", self.plot_robot_name_input)
        form.addRow("Model IDs (comma-separated):", self.plot_model_ids_input)
        form.addRow("Plot Types (comma-separated):", self.plot_types_input)

        # Buttons
        plot_button = QPushButton("Generate Plots")
        plot_button.clicked.connect(self.generate_plots)
        layout.addLayout(form)
        layout.addWidget(plot_button)

        tab.setLayout(layout)
        return tab

    def create_auto_training_tab(self):
        """Tab for auto training configuration."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Form for auto training parameters
        form = QFormLayout()
        self.auto_train_session_name_input = QLineEdit()
        self.auto_train_robot_name_input = QLineEdit()
        self.auto_train_workers_input = QSpinBox()
        self.auto_train_workers_input.setRange(1, 10)

        form.addRow("Session Name:", self.auto_train_session_name_input)
        form.addRow("Robot Name:", self.auto_train_robot_name_input)
        form.addRow("Max Workers:", self.auto_train_workers_input)

        # Buttons
        auto_train_button = QPushButton("Start Auto Training")
        auto_train_button.clicked.connect(self.start_auto_training)
        layout.addLayout(form)
        layout.addWidget(auto_train_button)

        tab.setLayout(layout)
        return tab

    def create_auto_testing_tab(self):
        """Tab for auto testing configuration."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Form for auto testing parameters
        form = QFormLayout()
        self.auto_test_robot_name_input = QLineEdit()
        self.auto_test_model_ids_input = QLineEdit()
        self.auto_test_iterations_input = QSpinBox()
        self.auto_test_iterations_input.setRange(1, 1000)

        form.addRow("Robot Name:", self.auto_test_robot_name_input)
        form.addRow("Model IDs (comma-separated):", self.auto_test_model_ids_input)
        form.addRow("Iterations:", self.auto_test_iterations_input)

        # Buttons
        auto_test_button = QPushButton("Start Auto Testing")
        auto_test_button.clicked.connect(self.start_auto_testing)
        layout.addLayout(form)
        layout.addWidget(auto_test_button)

        tab.setLayout(layout)
        return tab

    def create_retrain_tab(self):
        """Tab for retraining configuration."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Form for retraining parameters
        form = QFormLayout()
        self.retrain_model_name_input = QLineEdit()
        self.retrain_steps_input = QSpinBox()
        self.retrain_steps_input.setRange(1, 1000000)

        form.addRow("Model Name:", self.retrain_model_name_input)
        form.addRow("Retrain Steps:", self.retrain_steps_input)

        # Buttons
        retrain_button = QPushButton("Start Retraining")
        retrain_button.clicked.connect(self.start_retraining)
        layout.addLayout(form)
        layout.addWidget(retrain_button)

        tab.setLayout(layout)
        return tab

    def browse_scene(self):
        """Open a file dialog to select a scene file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Scene File", "", "Scene Files (*.ttt)")
        if file_path:
            self.scene_path_input.setText(file_path)

    def start_training(self):
        """Start the training process."""
        args = {
            "robot_name": self.robot_name_input.text(),
            "scene_path": self.scene_path_input.text(),
            "params_file": self.params_file_input.text(),
            "verbose": self.verbose_input.value(),
        }
        logging.info(f"Starting training with args: {args}")
        train.main(args)


    def stop_specific_test(self, test_thread, process_widget):
        """Stop an individual test process and remove its widget."""
        if test_thread.isRunning():
            test_thread.stop()
            logging.info("Stopping test thread...")

        if hasattr(test_thread, 'terminal_title'):
            logging.info(f"Closing terminal: {test_thread.terminal_title}")
            subprocess.run(['wmctrl', '-c', test_thread.terminal_title], check=False)

        # Log parada manual
        process_type = getattr(process_widget, 'process_type', 'Unknown')
        timestamp = getattr(process_widget, 'timestamp', 'Unknown')
        model_name = getattr(process_widget, 'model_name', 'UnknownModel')
        stop_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        message = f"<span style='color:orange;'>⏹️ Process <b>{process_type}</b> <i>{timestamp}</i> of <code>{model_name}</code> was manually stopped at <b>{stop_time}</b>.</span>"
        self.logs_text.append(message)

        process_widget.setParent(None)
        self.update_processes_label()


    def on_process_finished(self, process_widget):
        """Handle successful process completion and log it to the GUI."""
        process_type = getattr(process_widget, 'process_type', 'Unknown')
        timestamp = getattr(process_widget, 'timestamp', 'Unknown')
        model_name = getattr(process_widget, 'model_name', 'UnknownModel')
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        message = f"<span style='color:green;'>Success: </span> Process <b>{process_type}</b> <i>{timestamp}</i> of <code>{model_name}</code> finished successfully at <b>{end_time}</b>."
        self.logs_text.append(message)

        process_widget.setParent(None)
        self.update_processes_label()


    def on_process_error(self, error_message, process_widget):
        """Handle error during process execution and log it."""
        process_type = getattr(process_widget, 'process_type', 'Unknown')
        timestamp = getattr(process_widget, 'timestamp', 'Unknown')
        model_name = getattr(process_widget, 'model_name', 'UnknownModel')
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        message = (
            f"<span style='color:red;'>❌ Process <b>{process_type}</b> <i>{timestamp}</i> of <code>{model_name}</code> "
            f"failed at <b>{end_time}</b> with error:<br>{error_message}</span>"
        )
        self.logs_text.append(message)

        process_widget.setParent(None)
        self.update_processes_label()


    def disable_button_with_countdown(self, button, seconds=8):
        """Disable a button and show a countdown in its text."""
        original_text = button.text()
        button.setEnabled(False)

        def update_text():
            nonlocal seconds
            if seconds > 0:
                button.setText(f"Wait... ({seconds})")
                seconds -= 1
            else:
                self.timer.stop()
                button.setText(original_text)
                button.setEnabled(True)

        self.timer = QTimer(self)
        self.timer.timeout.connect(update_text)
        self.timer.start(1000)  # 1 segundo


    def start_testing(self):
        """Start the testing process (multi-threaded and UI-tracked)."""
        self.model_name = self.test_model_name_input.text()
        if not self.model_name:
            warning_msg = "<span style='color:orange;'>⚠️ Warning: please select a valid model name.</span>"
            self.logs_text.append(warning_msg)
            logging.warning("Please select a valid model name.")
            return
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        terminal_title = f"CoppeliaTerminal_{timestamp}"

        args = [
            "rl_coppelia", "test",
            "--model_name", self.remove_zip_extension(self.test_model_name_input.text()),
            "--iterations", str(self.test_iterations_input.value()),
            "--timestamp", str(timestamp),
            "--verbose", str(self.verbose_input.value())
        ]

        # Parámetros opcionales
        if self.save_traj_checkbox.isChecked():
            args.append("--save_traj")
        if self.test_params_file_input.text():
            args += ["--params_file", self.test_params_file_input.text()]
        if self.test_robot_name_input.text():
            args += ["--robot_name", self.test_robot_name_input.text()]
        if self.test_scene_path_input.text():
            args += ["--scene_path", self.test_scene_path_input.text()]
        if self.dis_parallel_mode_checkbox.isChecked():
            args.append("--dis_parallel_mode")
        if self.no_gui_checkbox.isChecked():
            args.append("--no_gui")

        logging.info(f"Starting testing with args: {args}")

        # Crear hilo de test
        test_thread = TestThread(args)
        test_thread.terminal_title = terminal_title

        # Crear widget visual del proceso
        process_widget = QWidget()
        process_layout = QVBoxLayout()
        process_widget.setLayout(process_layout)

        # Save metadata
        process_widget.process_type = "Test"
        process_widget.timestamp = timestamp
        process_widget.model_name = self.model_name or "UnknownModel"

        info_label = QLabel(f"<b>Test</b> — {timestamp}")
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        stop_button = QPushButton("Stop")
        stop_button.clicked.connect(lambda: self.stop_specific_test(test_thread, process_widget))

        process_layout.addWidget(info_label)
        process_layout.addWidget(progress_bar)
        process_layout.addWidget(stop_button)

        self.processes_container.addWidget(process_widget)
        self.update_processes_label()

        # Conectar señales con esta barra específica
        test_thread.progress_signal.connect(progress_bar.setValue)
        test_thread.finished_signal.connect(lambda: self.on_process_finished(process_widget))
        test_thread.error_signal.connect(lambda msg: self.on_process_error(msg, process_widget))

        test_thread.start()

        # Desactivar botón de Start Testing temporalmente
        button = self.sender()
        if isinstance(button, QPushButton):
            self.disable_button_with_countdown(button, seconds=8)



    def generate_plots(self):
        """Generate plots."""
        args = {
            "robot_name": self.plot_robot_name_input.text(),
            "model_ids": [int(x) for x in self.plot_model_ids_input.text().split(",")],
            "plot_types": self.plot_types_input.text().split(","),
        }
        logging.info(f"Generating plots with args: {args}")
        plot.main(args)

    def start_auto_training(self):
        """Start auto training."""
        args = {
            "session_name": self.auto_train_session_name_input.text(),
            "robot_name": self.auto_train_robot_name_input.text(),
            "max_workers": self.auto_train_workers_input.value(),
        }
        logging.info(f"Starting auto training with args: {args}")
        auto_training.main(args)

    def start_auto_testing(self):
        """Start auto testing."""
        args = {
            "robot_name": self.auto_test_robot_name_input.text(),
            "model_ids": [int(x) for x in self.auto_test_model_ids_input.text().split(",")],
            "iterations": self.auto_test_iterations_input.value(),
        }
        logging.info(f"Starting auto testing with args: {args}")
        auto_testing.main(args)

    def start_retraining(self):
        """Start retraining."""
        args = {
            "model_name": self.retrain_model_name_input.text(),
            "retrain_steps": self.retrain_steps_input.value(),
        }
        logging.info(f"Starting retraining with args: {args}")
        retrain.main(args)


def main():
    utils.logging_config_gui()
    app = QApplication(sys.argv)
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "uncore.png")
    print(logo_path)
    app.setWindowIcon(QIcon(logo_path))
    window = MainApp()
    window.show()
    sys.exit(app.exec_())