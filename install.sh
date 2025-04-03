#!/bin/bash

# Modify PYTHONPATH for adding the 'rl_spin_decoupler' dependency
BASE_DIR="$(pwd)"
DEPS_DIR="$BASE_DIR/dependencies/rl_spin_decoupler"
export PYTHONPATH="$DEPS_DIR:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

# Install all the required packages (including 'rl_coppelia' package in editable mode
pip install -r requirements.txt