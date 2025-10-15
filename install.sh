#!/bin/bash

# Modify PYTHONPATH for adding the 'rl_spin_decoupler' dependency
BASE_DIR="$(pwd)"
DEPS_DIR="$BASE_DIR/dependencies/rl_spin_decoupler"
RL_COPPELIA_DIR="$BASE_DIR/src/rl_coppelia"
SRC_DIR = "$BASE_DIR/src"

# Make the PYTHONPATH change permanent by adding it to the shell's configuration
SHELL_CONFIG="$HOME/.bashrc"  # Default for bash shell; change to ~/.zshrc for zsh users

PATH_UPDATED=false

# Add the export lines to the config file only if they are not already there
if ! grep -q "export PYTHONPATH=\"$DEPS_DIR" "$SHELL_CONFIG"; then
  echo "export PYTHONPATH=\"$DEPS_DIR:\$PYTHONPATH\"" >> "$SHELL_CONFIG"
  echo "export PYTHONPATH=\"$SRC_DIR:\$PYTHONPATH\"" >> "$SHELL_CONFIG"
  echo "rl_spin_decoupler has been added to PYTHONPATH in $SHELL_CONFIG"
  echo "The added lines are: PYTHONPATH=\"$DEPS_DIR:\$PYTHONPATH\ and PYTHONPATH=\"$SRC_DIR:\$PYTHONPATH\""
  PATH_UPDATED=true
else
  echo "rl_spin_decoupler is alreaedy configured in PYTHONPATH in $SHELL_CONFIG"
fi

if ! grep -q "$RL_COPPELIA_DIR" "$SHELL_CONFIG"; then
    echo "export PATH=\$PATH:$RL_COPPELIA_DIR" >> "$SHELL_CONFIG"
    echo "rl_coppelia has been added to PATH in $SHELL_CONFIG"
    echo "The added line is: PATH=\$PATH:$RL_COPPELIA_DIR"
    PATH_UPDATED=true
else
    echo "rl_coppelia is already configured in PATH in $SHELL_CONFIG"
fi

if [ "$PATH_UPDATED" = true ]; then
  echo "IMPORTANT: Please execute 'source $SHELL_CONFIG' for applying the changes."
fi

# Install all the required packages (including 'rl_coppelia' package in editable mode
pip install -r requirements.txt
pip install -e .

# Reload the shell configuration so the change takes effect immediately
source "$SHELL_CONFIG"