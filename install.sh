#!/bin/bash

# Modify PYTHONPATH for adding the 'rl_spin_decoupler' dependency
BASE_DIR="$(pwd)"
DEPS_DIR="$BASE_DIR/dependencies/rl_spin_decoupler"
RL_COPPELIA_DIR="$BASE_DIR/src/rl_coppelia"
SRC_DIR="$BASE_DIR/src"

# Make the PYTHONPATH change permanent by adding it to the shell's configuration
SHELL_CONFIG="$HOME/.bashrc"  # Default for bash shell; change to ~/.zshrc for zsh users

expand_path() {
  local input="$1"
  eval echo "$input"
}

# --- Create and activate environment ---
echo ""
read -r -p "Do you have a folder with your virtual environments (venvs)? [y/N]: " HAS_CENTRAL
HAS_CENTRAL="${HAS_CENTRAL:-N}"

if [[ "$HAS_CENTRAL" =~ ^[Yy]$ ]]; then
  # Ask for base path of venvs
  read -r -p "Introduce venv folder path (ej. ~/.venvs): " VENV_BASE_INPUT
  VENV_BASE_INPUT="${VENV_BASE_INPUT:-$HOME/.venvs}"
  VENV_BASE_DIR="$(expand_path "$VENV_BASE_INPUT")"

  # Name for the venv
  read -r -p "Name for the venv (default: uncore_rl_venv): " VENV_NAME
  VENV_NAME="${VENV_NAME:-uncore_rl_venv}"
  VENV_DIR="$VENV_BASE_DIR/$VENV_NAME"
  
else
  # Creating venv folder inside HOME
  VENV_BASE_DIR="$HOME/.venvs"
  VENV_NAME="uncore_rl_venv"
  VENV_DIR="$VENV_BASE_DIR/$VENV_NAME"
fi
echo "➡️  Venv to be used: $VENV_DIR"

# Create the virtual environment folder if it does not exist
mkdir -p "$(dirname "$VENV_DIR")" 2>/dev/null

# Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip inside the venv
echo "Upgrading pip..."
pip install --upgrade pip

# --- Update shell configuration ---
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
  
  if [ "$0" != "$BASH_SOURCE" ]; then
    echo "🔁 Applying environment changes..."
    source "$SHELL_CONFIG"
    echo "✅ Environment variables updated!"
  else
    echo "IMPORTANT: Please execute 'source $SHELL_CONFIG' for applying the changes."
  fi
fi

# --- Install python dependencies ---
echo "Installing Python dependencies..."
pip install -r requirements.txt
pip install -e .