#!/bin/bash

# Modify PYTHONPATH for adding the 'rl_spin_decoupler' dependency
export PYTHONPATH="/home/adrian/devel/rl_coppelia/dependencies/rl_spin_decoupler:$PYTHONPATH"

# Install all the required packages (including 'rl_coppelia' package in editable mode
pip install -r requirements.txt