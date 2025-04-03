#!/bin/bash

# Modify PYTHONPATH for adding the 'rl_spin_decoupler' dependency
BASE_DIR="$(pwd)"
DEPS_DIR="$BASE_DIR/dependencies/rl_spin_decoupler"

# Make the PYTHONPATH change permanent by adding it to the shell's configuration
SHELL_CONFIG="$HOME/.bashrc"  # Default for bash shell; change to ~/.zshrc for zsh users

# Add the export line to the config file only if it is not already there
if ! grep -q "export PYTHONPATH=\"$DEPS_DIR" "$SHELL_CONFIG"; then
  echo "export PYTHONPATH=\"$DEPS_DIR:\$PYTHONPATH\"" >> "$SHELL_CONFIG"
  echo "PYTHONPATH has been added to $SHELL_CONFIG"
else
  echo "PYTHONPATH is already set in $SHELL_CONFIG"
fi

# Reload the shell configuration so the change takes effect immediately
source "$SHELL_CONFIG"

# Install all the required packages (including 'rl_coppelia' package in editable mode
pip install -r requirements.txt
pip install -e .
