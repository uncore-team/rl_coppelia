#!/usr/bin/env bash
set -Eeuo pipefail

# Debug trap: prints failing line and keeps window open if KEEP_OPEN=1
trap 'rc=$?; echo "❌ Error (exit $rc) at line $LINENO: ${BASH_COMMAND}"; 
      if [[ "${KEEP_OPEN:-0}" == "1" ]]; then read -rp "Press Enter to close..."; fi; exit $rc' ERR

# Enable xtrace if DEBUG=1
if [[ "${DEBUG:-0}" == "1" ]]; then set -x; fi

# Check if the script is being sourced or executed
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "Please 'source' this script, do not execute it."
  exit 1
fi


# Modify PYTHONPATH for adding the 'rl_spin_decoupler' dependency
BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" # Directory where this script is located
DEPS_DIR="$BASE_DIR/dependencies/rl_spin_decoupler"
SRC_DIR="$BASE_DIR/src"
RL_COPPELIA_DIR="$SRC_DIR/rl_coppelia"

# Detect user's shell rc file (fallback to bashrc)
SHELL_NAME="${SHELL##*/}"
case "${SHELL_NAME}" in
  zsh)  SHELL_CONFIG="${HOME}/.zshrc" ;;
  bash) SHELL_CONFIG="${HOME}/.bashrc" ;;
  *)    SHELL_CONFIG="${HOME}/.bashrc" ;; # default
esac


# ---- helpers ----
die() {
  echo "❌ $*" >&2
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 1
  else
    exit 1
  fi
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

expand_to_abs() {
  local p="$1"
  "$PY" - "$p" <<'PYCODE'
import os, sys, pathlib

if len(sys.argv) < 2:
    print(os.getcwd())
    sys.exit(0)

s = sys.argv[1]

# Trim outer quotes if any (common when user pastes with quotes)
if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
    s = s[1:-1]

# Expand ~ and $VARS
s = os.path.expandvars(os.path.expanduser(s))

# Make absolute if needed
p = pathlib.Path(s)
if not p.is_absolute():
    p = pathlib.Path(os.getcwd()) / p

# Normalize but don't require existence
print(str(p.resolve(strict=False)))
PYCODE
}






# Choose python interpreter
choose_python() {
  if command_exists python3; then
    echo "python3"
  elif command_exists python; then
    echo "python"
  else
    die "Python is not installed. Please install Python 3.8+ first."
  fi
}

PY="$(choose_python)"

# --- Start installation ---
echo
echo "== UnCoRe RL installer =="
echo "Repo root: ${BASE_DIR}"
echo "Source dir: ${SRC_DIR}"
echo


# --- Create and activate environment ---
echo ""
read -r -p "Do you have a folder with your virtual environments (venvs)? [y/N]: " HAS_CENTRAL
HAS_CENTRAL="${HAS_CENTRAL:-N}"

if [[ "$HAS_CENTRAL" =~ ^[Yy]$ ]]; then
  read -r -p "Enter the venv base path (e.g., ~/.venvs): " VENV_BASE_INPUT
  VENV_BASE_INPUT="${VENV_BASE_INPUT:-$HOME/.venvs}"
  # sanitize trailing/leading spaces and CR
  VENV_BASE_INPUT="$(printf '%s' "$VENV_BASE_INPUT" | tr -d '\r' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  VENV_BASE_DIR="$(expand_to_abs "$VENV_BASE_INPUT")"

  read -r -p "Name for the venv (default: uncore_rl_venv): " VENV_NAME
  VENV_NAME="${VENV_NAME:-uncore_rl_venv}"
  
  VENV_DIR="${VENV_BASE_DIR}/${VENV_NAME}"
else
  VENV_BASE_DIR="${HOME}/.venvs"
  VENV_NAME="uncore_rl_venv"
  VENV_DIR="${VENV_BASE_DIR}/${VENV_NAME}"
fi

echo "➡️  Virtualenv path: ${VENV_DIR}"
mkdir -p -- "${VENV_BASE_DIR}"

# --- Create venv if missing ---
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtualenv..."
  "${PY}" -m venv "${VENV_DIR}" || die "Failed to create venv at ${VENV_DIR}"
else
  echo "Venv already exists."
fi


echo "Activating venv..."
source "${VENV_DIR}/bin/activate" || die "Failed to activate venv ${VENV_DIR}"

# Upgrade pip inside the venv
echo "Upgrading pip..."
pip install --upgrade pip

# --- Environment exports (prefer wrapper over touching shell rc) ---
# We set PATH for rl_coppelia CLI tools and PYTHONPATH for src + dependency.
# We will generate a wrapper 'activate_rl' that users can source.
ACTIVATE_WRAPPER="${VENV_DIR}/bin/activate_rl"
cat > "${ACTIVATE_WRAPPER}" <<'EOF'
# Usage: source /path/to/venv/bin/activate_rl
# This wrapper activates the venv and exports PATH/PYTHONPATH temporarily for this shell.
# Do not execute directly; source it.

# Activate the venv itself
# shellcheck disable=SC1091
source "$(dirname -- "${BASH_SOURCE[0]}")/activate"

# Project-specific exports (filled below by installer)
# __RL_COPPELIA_PATH__ and __DEPS_PATH__ are placeholders to be replaced at install time.
export PATH="${PATH}:__RL_COPPELIA_PATH__"
# Prepend both deps and project src to PYTHONPATH
export PYTHONPATH="__DEPS_PATH__:${PYTHONPATH:-}"
export PYTHONPATH="__SRC_PATH__:${PYTHONPATH}"
echo "✅ rl_coppelia environment ready (temporary)."
EOF

# Replace placeholders with actual absolute paths
# (Use printf %q if you need to escape spaces; here we assume typical no-space paths.)
tmp_file="${ACTIVATE_WRAPPER}.tmp"
sed \
  -e "s|__RL_COPPELIA_PATH__|${RL_COPPELIA_DIR}|g" \
  -e "s|__DEPS_PATH__|${DEPS_DIR}|g" \
  -e "s|__SRC_PATH__|${SRC_DIR}|g" \
  "${ACTIVATE_WRAPPER}" > "${tmp_file}"
mv -- "${tmp_file}" "${ACTIVATE_WRAPPER}"
chmod +x "${ACTIVATE_WRAPPER}"

echo
echo "A temporary activator was created:"
echo "  source \"${ACTIVATE_WRAPPER}\""
echo "(It sets PATH/PYTHONPATH only for the current shell.)"

# --- (Optional) Offer to update user's shell rc persistently ---
echo
read -r -p "Also add permanent PATH/PYTHONPATH exports to ${SHELL_CONFIG}? [y/N]: " WANT_RC
WANT_RC="${WANT_RC:-N}"
if [[ "$WANT_RC" =~ ^[Yy]$ ]]; then
  # Use idempotent markers to avoid duplicates
  START_MARK="# >>> rl_coppelia (installer) >>>"
  END_MARK="# <<< rl_coppelia (installer) <<<"

  # Remove old block if present
  if grep -Fq "${START_MARK}" "${SHELL_CONFIG}" 2>/dev/null; then
    awk -v start="${START_MARK}" -v end="${END_MARK}" '
      $0==start {skip=1}
      skip && $0==end {skip=0; next}
      !skip {print}
    ' "${SHELL_CONFIG}" > "${SHELL_CONFIG}.tmp" && mv -- "${SHELL_CONFIG}.tmp" "${SHELL_CONFIG}"
  fi

  {
    echo "${START_MARK}"
    echo "export PATH=\"\$PATH:${RL_COPPELIA_DIR}\""
    echo "export PYTHONPATH=\"${DEPS_DIR}:\$PYTHONPATH\""
    echo "export PYTHONPATH=\"${SRC_DIR}:\$PYTHONPATH\""
    echo "${END_MARK}"
  } >> "${SHELL_CONFIG}"

  echo "✅ Added persistent PATH/PYTHONPATH to ${SHELL_CONFIG}"
  echo "   Run: source ${SHELL_CONFIG}"
else
  echo "Skipping persistent shell config changes (safer default)."
fi

# --- Install python dependencies ---
echo
echo "Installing Python dependencies..."
pip install -r "${BASE_DIR}/requirements.txt"
pip install -e "${BASE_DIR}"

echo
echo "✅ Done."
echo "Next steps:"
echo "  1) source \"${ACTIVATE_WRAPPER}\""
echo "  2) run your uncore_rl commands"