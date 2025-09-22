from __future__ import annotations

import contextlib
import io
import logging
import os
from pathlib import Path
import re

def capture_cli_output(self, callable_fn, argv=None):
        """Run a CLI-style callable, capturing stdout, stderr and logging output.

        Args:
            callable_fn (function): A callable to run (e.g., cli.main).
            argv (list, optional): Argument list to simulate CLI input.

        Returns:
            tuple: (combined_output: str, errors: list[str], success: bool)
        """
        import io
        import logging
        import contextlib

        stdout_capture = io.StringIO()
        log_capture = io.StringIO()

        log_handler = logging.StreamHandler(log_capture)
        log_handler.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.addHandler(log_handler)

        success = True

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stdout_capture):
            try:
                callable_fn(argv)
            except Exception:
                success = False
            finally:
                logger.removeHandler(log_handler)

        combined_output = stdout_capture.getvalue() + "\n" + log_capture.getvalue()
        errors = [line for line in combined_output.splitlines() if " - ERROR - " in line]

        return combined_output.strip(), errors, success


def remove_zip_extension(self, file_path):
        """Remove the .zip extension from the file name if it exists."""
        base_name, extension = os.path.splitext(file_path)
        if extension.lower() == ".zip":
            return base_name
        return file_path


def list_dirs(path: str) -> list[str]:
    """Return a sorted list of directory names under path (non-recursive)."""
    if not os.path.isdir(path):
        return []
    return sorted([n for n in os.listdir(path) if os.path.isdir(os.path.join(path, n))])


def list_json_files(path: str) -> list[str]:
    """Return a sorted list of json files under path."""
    if not os.path.isdir(path):
        return []
    return sorted([n for n in os.listdir(path) if n.endswith(".json")])


def get_rl_coppelia_path_from_bashrc(self):
    """Retrieve the rl_coppelia path from ~/.bashrc."""
    bashrc_path = os.path.expanduser("~/.bashrc")
    if not os.path.exists(bashrc_path):
        return None
    
    try:
        with open(bashrc_path, "r") as bashrc:
            content = bashrc.read()
            
        # Search in PYTHONPATH
        pythonpath_matches = re.findall(r'(?:export\s+)?PYTHONPATH[^=]*=(.+)', content)
        for match in pythonpath_matches:
            # Clean the result (remove spaces and "")
            clean_match = match.strip().strip('"').strip("'")
            paths = clean_match.split(":")
            for path in paths:
                path = path.strip()
                if path and "rl_coppelia" in path:
                    # Expand environment variables (if so)
                    expanded_path = os.path.expandvars(path)
                    if os.path.exists(expanded_path):
                        base_path = str(Path(expanded_path).parents[1])  # Get the parent directory of rl_coppelia
                        if base_path != self.base_path:
                            self.base_path = base_path  # Update the base path
                            logging.warning(f"Found rl_coppelia path in PYTHONPATH: {base_path}, but it does not match the expected base path: {self.base_path}")
                        return base_path  
        
        # Search in PATH it the previous search was unsuccessful
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
        logging.error(f"Error reading .bashrc: {e}")
        
    return None
    

def get_browse_zip_file_path(self):
        """Open a file dialog to select a ZIP file, starting in the rl_coppelia directory."""
        rl_coppelia_path = self.get_rl_coppelia_path_from_bashrc()
        
        # If rl_coppelia path was found, then it will be used as main directory for searching files
        if rl_coppelia_path and os.path.exists(rl_coppelia_path):
            start_path = rl_coppelia_path
            logging.info(f"Starting file dialog in rl_coppelia directory: {start_path}")
        else:
            start_path = os.path.expanduser("~")
            logging.warning(f"rl_coppelia path not found, starting in home directory: {start_path}")
        return start_path


