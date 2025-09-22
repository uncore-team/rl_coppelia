import logging
import re
import subprocess
from PyQt5.QtCore import QThread, pyqtSignal
from typing import Optional


def parse_progress(line: str) -> Optional[int]:
    """Extract percentage from a line.
    
    Supports "... 42%" anywhere in the line.
    """
    m = re.search(r"(\d+)%", line)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None
    

class ProcessThread(QThread):
    progress_signal = pyqtSignal(int)  # Signal for updating progress bar
    finished_signal = pyqtSignal()    # Signal for indicating the process has finished
    error_signal = pyqtSignal(str)    # Signal for managing errors

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.terminal_title = ""
        self.was_stopped_manually = False
        self.process: Optional[subprocess.Popen] = None

    def stop(self):
        self.was_stopped_manually = True
        if hasattr(self, 'process') and self.process and self.process.poll() is None:
            self.process.terminate()

    def run(self):
        """Execute each train/test in a separate thread, so the user can run multiple processes simultanously."""
        try:
            self.was_stopped_manually = False  # Reset flag

            # Execute the command using subprocess
            self.process = subprocess.Popen(
                self.args,
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True
            )
            assert self.process.stdout is not None

            # Read the output of the process for updating the progress bar in real time
            for line in self.process.stdout:
                logging.debug(line.strip())  # Console debug
                pct = parse_progress(line)  # Get percentage
                if pct is not None:
                    self.progress_signal.emit(pct)  # Emit progress signal

            self.process.wait()

            # Avoid error indicators if the process was manually stopped
            if self.was_stopped_manually:
                return

            if self.process.returncode != 0:
                self.error_signal.emit("Process returned non-zero exit code.")
            else:
                self.finished_signal.emit()

        except Exception as e:
            self.error_signal.emit(str(e))