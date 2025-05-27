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
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QProgressBar


class TestThread(QThread):
    progress_signal = pyqtSignal(int)  # Señal para actualizar la barra de progreso
    finished_signal = pyqtSignal()    # Señal para indicar que el test ha terminado
    error_signal = pyqtSignal(str)    # Señal para manejar errores

    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self):
        """Ejecuta el test en un hilo separado."""
        try:
            # Ejecutar el comando usando subprocess
            process = subprocess.Popen(
                self.args,
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,  # Para recibir la salida como texto en lugar de bytes
                universal_newlines=True  # Para manejar la salida de texto correctamente
            )

            # Leer la salida en tiempo real para actualizar la barra de progreso
            for line in process.stdout:
                print(line.strip())  # Depuración: imprimir cada línea de salida
                if "Testing Episodes" in line:  # Buscar la línea de progreso de tqdm
                    try:
                        # Buscar el porcentaje en la línea usando una expresión regular
                        match = re.search(r"(\d+)%", line)
                        if match:
                            progress = int(match.group(1))  # Extraer el porcentaje como entero
                            self.progress_signal.emit(progress)  # Emitir señal para actualizar la barra
                        else:
                            logging.warning(f"Could not parse progress from line: {line.strip()}")
                    except ValueError:
                        logging.warning(f"Could not parse progress from line: {line.strip()}")

            process.wait()  # Esperar a que el proceso termine
            if process.returncode != 0:
                error_message = process.stderr.read()
                self.error_signal.emit(error_message)
            else:
                self.finished_signal.emit()  # Emitir señal de finalización
        except Exception as e:
            self.error_signal.emit(str(e))


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL Coppelia Manager")
        self.setGeometry(100, 100, 800, 600)

        expanded_path = os.path.abspath(__file__)
        self.base_path = str(Path(expanded_path).parents[2])
        logging.info(f"Base path configured as: {self.base_path}")
        self.model_name = ""
        self.robot_name = ""
        self.experiment_id = 0

        # Tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Add tabs
        self.tabs.addTab(self.create_train_tab(), "Train")
        self.tabs.addTab(self.create_test_tab(), "Test")
        self.tabs.addTab(self.create_plot_tab(), "Plot")
        self.tabs.addTab(self.create_auto_training_tab(), "Auto Training")
        self.tabs.addTab(self.create_auto_testing_tab(), "Auto Testing")
        self.tabs.addTab(self.create_retrain_tab(), "Retrain")

    def update_progress_bar(self, value):
        """Actualizar la barra de progreso."""
        self.progress_bar.setValue(value)

    def on_test_finished(self):
        """Manejar la finalización del test."""
        logging.info("Test completed successfully.")
        self.progress_bar.setValue(100)  # Asegurarse de que la barra esté llena
        self.tabs.setTabEnabled(self.tabs.indexOf(self.create_test_tab()), True)  # Reactivar la pestaña de Test

    def on_test_error(self, error_message):
        """Manejar errores durante el test."""
        logging.error(f"Error during test: {error_message}")
        self.tabs.setTabEnabled(self.tabs.indexOf(self.create_test_tab()), True)  # Reactivar la pestaña de Test


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
        self.test_iterations_input.setValue(50)

        # Verbose level (optional, default: 3)
        self.verbose_input = QSpinBox()
        self.verbose_input.setRange(-1, 4)
        self.verbose_input.setValue(3)

        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

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
        form.addRow("Iterations (default: 50):", self.test_iterations_input)
        form.addRow("Verbose Level (default: 0):", self.verbose_input)

        # Buttons
        test_button = QPushButton("Start Testing")
        test_button.clicked.connect(self.start_testing)
        layout.addLayout(form)
        layout.addWidget(test_button)

        # Añadir la barra de progreso al diseño
        layout.addWidget(self.progress_bar)

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

    def start_testing(self):
        """Start the testing process."""
        args = [
            "rl_coppelia", "test",
            "--model_name", self.remove_zip_extension(self.model_name),
            "--iterations", str(self.test_iterations_input.value()),
            "--verbose", str(self.verbose_input.value())
        ]

        # Add optional parameters if they are provided
        if self.save_traj_checkbox.isChecked():
            args.append("--save_traj")
        if self.test_params_file_input.text():
            args.extend(["--params_file", self.test_params_file_input.text()])
        if self.test_robot_name_input.text():
            args.extend(["--robot_name", self.test_robot_name_input.text()])
        if self.test_scene_path_input.text():
            args.extend(["--scene_path", self.test_scene_path_input.text()])
        if self.dis_parallel_mode_checkbox.isChecked():
            args.append("--dis_parallel_mode")
        if self.no_gui_checkbox.isChecked():
            args.append("--no_gui")

        logging.info(f"Starting testing with args: {args}")

        # Deshabilitar la pestaña de Test
        # self.tabs.setTabEnabled(self.tabs.indexOf(self.create_test_tab()), False)

        # Crear y configurar el hilo
        self.test_thread = TestThread(args)
        self.test_thread.progress_signal.connect(self.update_progress_bar)
        self.test_thread.finished_signal.connect(self.on_test_finished)
        self.test_thread.error_signal.connect(self.on_test_error)

        # Iniciar el hilo
        self.test_thread.start()


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


if __name__ == "__main__":
    utils.logging_config_gui()
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())