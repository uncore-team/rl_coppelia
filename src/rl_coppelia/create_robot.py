import logging
import os
import sys
import termios
import tty
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from common import robot_generator, utils


# ---------------------------------
# ------ SPEC BUILDING LOGIC -------
# ---------------------------------

def _build_action_spec() -> Dict[str, Any]:
    """Interactively build the action space spec asking for variable names first.

    The user provides a name for each action dimension before entering its bounds.
    Names are stored in env_spec['act']['vars'].
    """
    print("\n-- Action space --")
    n_act = utils.prompt_int("Number of action variables", min_val=1)
    vars_list: List[Dict[str, Any]] = []

    for i in range(n_act):
        default_name = f"act_{i}"
        name = utils.prompt_str(f"  · Action var {i+1} name", default=default_name)
        low = utils.prompt_float(f"    Bottom limit for '{name}'")
        high = utils.prompt_float(f"    Upper limit for '{name}'")
        vars_list.append({"name": name, "low": low, "high": high})

    return {"vars": vars_list}


def _build_obs_spec() -> Dict[str, Any]:
    """Interactively build the observation space spec asking for names first.

    - Optionally includes laser observations; user can auto-name (laser_obs0..N-1)
      or provide custom names for each laser dimension.
    - Then asks for additional (non-laser) observation variables, prompting first
      for the variable name, then for the bounds.
    """
    print("\n-- Observation space --")
    vars_list: List[Dict[str, Any]] = []

    # ----- Laser observations (optional) -----
    include_laser = utils.prompt_yes_no("Include laser observations?", default=False)
    laser_count = 0
    if include_laser:
        laser_count = utils.prompt_int("  Number of laser observations", min_val=1)
        laser_low = utils.prompt_float("  Bottom limit for all laser observations", default=0.0)
        laser_high = utils.prompt_float("  Upper limit for all laser observations", default=10.0)

        custom_names = utils.prompt_yes_no("  Do you want to name each laser variable?", default=False)
        if custom_names:
            for i in range(laser_count):
                default_name = f"laser_obs{i}"
                name = utils.prompt_str(f"    · Laser var {i+1} name", default=default_name)
                vars_list.append({"name": name, "low": laser_low, "high": laser_high})
        else:
            for i in range(laser_count):
                name = f"laser_obs{i}"
                vars_list.append({"name": name, "low": laser_low, "high": laser_high})

    # ----- Additional scalar observations -----
    n_obs_extra = utils.prompt_int(
        "Number of additional (non-laser) observation variables",
        default=0, min_val=0
    )
    for i in range(n_obs_extra):
        default_name = f"obs_{i}"
        name = utils.prompt_str(f"  · Obs var {i+1} name", default=default_name)
        low = utils.prompt_float(f"    Bottom limit for '{name}'")
        high = utils.prompt_float(f"    Upper limit for '{name}'")
        vars_list.append({"name": name, "low": low, "high": high})

    # Nota: devolvemos también laser_count para rellenar params_updates["params_env"]["laser_observations"]
    return {"vars": vars_list}, laser_count


def _build_robot_data() -> Dict[str, Any]:
    """Prompt for basic robot kinematics."""
    print("\n-- Robot data --")
    dbw = utils.prompt_float("distance_between_wheels (m)")
    wr = utils.prompt_float("wheel_radius (m)")
    return {"distance_between_wheels": dbw, "wheel_radius": wr}


def _build_agent_handles() -> Dict[str, Any]:
    """Prompt for scene handles used by the Agent in Coppelia."""
    print("\n-- Scene handles (Coppelia) --")
    h_robot = utils.prompt_str("  robot (e.g., /Turtlebot2)")
    h_baselink = utils.prompt_str("  robot_baselink (e.g., /Turtlebot2/base_link_respondable)")
    use_laser = utils.prompt_yes_no("  Has laser handle?", default=True)
    h_laser = utils.prompt_str("  laser (e.g., /Turtlebot2/fastHokuyo_ROS2)", allow_empty=not use_laser) if use_laser else ""

    def _starts_with_bar(h):
        """Ensure that the string starts with a backslash ('\\')."""
        if not h.startswith("\\"):
            h = "\\" + h
        return h

    return {
        "robot": _starts_with_bar(h_robot),
        "robot_baselink": _starts_with_bar(h_baselink),
        "laser": _starts_with_bar(h_laser) or ""
    }


def _pick_scene(base_path: str) -> str:
    """Let the user pick a .ttt scene from <base_path>/scenes with an arrow menu.

    Falls back to manual input if no scenes are found or curses fails.
    Adds '.ttt' automatically if missing.
    """
    scenes_dir = os.path.join(base_path, "scenes")
    os.makedirs(scenes_dir, exist_ok=True)

    try:
        files = sorted(
            [f for f in os.listdir(scenes_dir) if f.lower().endswith(".ttt")]
        )
    except Exception as e:
        logging.warning(f"Could not list scenes in {scenes_dir}: {e}")
        files = []

    # Always include a "type manually" option
    MANUAL = "Type a custom scene name..."
    options = files + [MANUAL] if files else [MANUAL]

    # Try curses menu; fallback to manual prompt if curses fails
    chosen = None
    try:
        chosen = utils.arrow_menu(options, title=f"Select a Coppelia scene ({scenes_dir})")
    except Exception as e:
        logging.warning(f"Curses menu failed, falling back to manual input: {e}")
        chosen = MANUAL

    if chosen == MANUAL:
        typed = utils.prompt_str("Scene name (without or with .ttt)")
        return utils.ensure_ttt(typed)

    # Picked an existing file
    return utils.ensure_ttt(chosen)


# def interactive_create_specs(base_path: str) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
#     """Run an interactive session and return all specs.

#     Args:
#         base_path (str): Base directory of the project

#     Returns:
#         (robot_name, env_spec, agent_spec, params_updates)
#         - params_updates: extra params to write into params JSON (e.g., laser_observations)
#     """

#     utils.print_uncore_logo()

#     print("------ Create Robot (Env + Agent) ------")
#     robot_name = utils.prompt_str("Robot name")

#     # Spaces
#     act_spec = _build_action_spec()
#     obs_spec, laser_count = _build_obs_spec()

#     # Robot data
#     robot_data = _build_robot_data()

#     print("\n-- Scene selection --")
#     scene_name = _pick_scene(base_path)
#     print(f"Scene name: {scene_name}")

#     # Agent handles
#     handles = _build_agent_handles()

#     env_spec = {
#         "robot_data": robot_data,
#         "obs": obs_spec,
#         "act": act_spec,
#     }
#     agent_spec = {
#         "handles": handles
#     }

#     # Params JSON extra updates
#     params_updates: Dict[str, Any] = {"params_env": {}}
#     if laser_count > 0:
#         params_updates["params_env"]["laser_observations"] = laser_count

#     return robot_name, env_spec, agent_spec, params_updates


class _BackSignal(Exception):
    """Internal signal to go back one step (raised on Ctrl+B)."""
    pass

def _read_line_ctrlb(label: str, default: Optional[str] = None) -> str:
    """Lee una línea con soporte Ctrl+B y Ctrl+C (cross-platform POSIX).
    - Enter -> devuelve texto (o default si vacío y default existe)
    - Ctrl+B -> lanza _BackSignal para retroceder
    - Ctrl+C -> lanza KeyboardInterrupt (cancelación normal)
    """
    # Prompt limpio sin indentaciones raras
    prompt = f"\n{label}"
    if default not in (None, ""):
        prompt += f" [{default}]"
    prompt += ": "
    sys.stdout.write(prompt)
    sys.stdout.flush()

    buf: List[str] = []

    if os.name == "nt":
        import msvcrt
        while True:
            ch = msvcrt.getwch()
            if ch in ("\r", "\n"):               # Enter
                sys.stdout.write("\n"); sys.stdout.flush()
                text = "".join(buf)
                return text if text or default is None else str(default)
            if ch == "\x03":                     # Ctrl+C
                sys.stdout.write("\n"); sys.stdout.flush()
                raise KeyboardInterrupt
            if ch == "\x02":                     # Ctrl+B
                sys.stdout.write("\n"); sys.stdout.flush()
                raise _BackSignal()
            if ch in ("\b", "\x7f"):             # Backspace
                if buf:
                    buf.pop(); sys.stdout.write("\b \b"); sys.stdout.flush()
                continue
            if ch in ("\x00", "\xe0"):           # Teclas especiales -> consume siguiente
                _ = msvcrt.getwch()
                continue
            buf.append(ch); sys.stdout.write(ch); sys.stdout.flush()
    else:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch in ("\r", "\n"):           # Enter
                    sys.stdout.write("\n"); sys.stdout.flush()
                    text = "".join(buf)
                    return text if text or default is None else str(default)
                if ch == "\x03":                 # Ctrl+C
                    sys.stdout.write("\n"); sys.stdout.flush()
                    raise KeyboardInterrupt
                if ch == "\x02":                 # Ctrl+B
                    sys.stdout.write("\n"); sys.stdout.flush()
                    raise _BackSignal()
                if ch in ("\x7f", "\b"):         # Backspace
                    if buf:
                        buf.pop(); sys.stdout.write("\b \b"); sys.stdout.flush()
                    continue
                if ch.isprintable():
                    buf.append(ch); sys.stdout.write(ch); sys.stdout.flush()
                # ignora no imprimibles
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

def _patch_utils_prompts():
    """Devuelve (wrappers, restore_fn). Los wrappers reemplazan utils.prompt_* con soporte Ctrl+B."""

    orig = {
        "str": getattr(utils, "prompt_str", None),
        "int": getattr(utils, "prompt_int", None),
        "float": getattr(utils, "prompt_float", None),
        "choice": getattr(utils, "prompt_choice", None),
        "confirm": getattr(utils, "prompt_confirm", None),
    }

    def prompt_str(label: str, default: Optional[str] = None) -> str:
        return _read_line_ctrlb(label, default=default)

    def prompt_int(label: str, default: Optional[int] = None) -> int:
        while True:
            s = _read_line_ctrlb(label, default=None if default is None else str(default))
            if s == "" and default is not None:
                return int(default)
            try:
                return int(s)
            except ValueError:
                print("Valor inválido. Introduce un entero (Ctrl+B para volver).")

    def prompt_float(label: str, default: Optional[float] = None) -> float:
        while True:
            s = _read_line_ctrlb(label, default=None if default is None else str(default))
            if s == "" and default is not None:
                return float(default)
            try:
                return float(s)
            except ValueError:
                print("Valor inválido. Introduce un número (Ctrl+B para volver).")

    def prompt_choice(label: str, options: Iterable[str], default_idx: int = 0) -> str:
        opts = list(options)
        for i, opt in enumerate(opts):
            print(f"  [{i}] {opt}")
        idx = prompt_int(f"{label} (elige índice)", default=default_idx)
        if 0 <= idx < len(opts):
            return opts[idx]
        print("Índice fuera de rango. Usando el índice por defecto.")
        return opts[default_idx]

    def prompt_confirm(label: str, default: bool = True) -> bool:
        suf = "Y/n" if default else "y/N"
        ans = prompt_str(f"{label} ({suf})", default="y" if default else "n").strip().lower()
        return ans in ("y", "yes", "s", "si", "sí")

    # monkey-patch
    utils.prompt_str = prompt_str
    utils.prompt_int = prompt_int
    utils.prompt_float = prompt_float
    utils.prompt_choice = prompt_choice
    utils.prompt_confirm = prompt_confirm

    def restore():
        # restaura sólo si existían
        if orig["str"] is not None:    utils.prompt_str = orig["str"]
        if orig["int"] is not None:    utils.prompt_int = orig["int"]
        if orig["float"] is not None:  utils.prompt_float = orig["float"]
        if orig["choice"] is not None: utils.prompt_choice = orig["choice"]
        if orig["confirm"] is not None:utils.prompt_confirm = orig["confirm"]

    return restore

# ===================== Tu función, genérica como antes =====================

def interactive_create_specs(base_path: str) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Run an interactive session and return all specs.

    Igual que antes, pero:
      - Ctrl+B retrocede de paso.
      - Ctrl+C cancela (KeyboardInterrupt normal).
      - Sin indentaciones raras en los prompts.
    """
    utils.print_uncore_logo()
    print("------ Create Robot (Env + Agent) ------")
    print("ℹ️  Ctrl+B: volver al paso anterior · Ctrl+C: cancelar\n")

    # Activa prompts con Ctrl+B en todo el flujo (también dentro de tus builders)
    restore_prompts = _patch_utils_prompts()

    # Estado
    robot_name: str = ""
    act_spec: Dict[str, Any] = {}
    obs_spec: Dict[str, Any] = {}
    robot_data: Dict[str, Any] = {}
    scene_name: str = ""
    handles: Dict[str, Any] = {}
    laser_count: int = 0

    def step_robot_name():
        nonlocal robot_name
        robot_name = utils.prompt_str("Robot name")

    def step_spaces_action():
        nonlocal act_spec
        act_spec = _build_action_spec()  # usa utils.prompt_* (ya parcheados)

    def step_spaces_obs():
        nonlocal obs_spec, laser_count
        obs_spec, laser_count = _build_obs_spec()

    def step_robot_data():
        nonlocal robot_data
        robot_data = _build_robot_data()

    def step_scene():
        nonlocal scene_name
        print("\n-- Scene selection --")
        scene_name = _pick_scene(base_path)
        print(f"Scene name: {scene_name}")

    def step_handles():
        nonlocal handles
        handles = _build_agent_handles()

    steps: list[Callable[[], None]] = [
        step_robot_name,
        step_spaces_action,
        step_spaces_obs,
        step_robot_data,
        step_scene,
        step_handles,
    ]

    i = 0
    try:
        while 0 <= i < len(steps):
            try:
                steps[i]()
                i += 1
            except _BackSignal:
                i = max(0, i - 1)
    except KeyboardInterrupt:
        print("\n⛔ Operación cancelada por el usuario (Ctrl+C).")
        raise
    finally:
        # Restaurar los prompts originales, pase lo que pase
        restore_prompts()

    # Construcción EXACTAMENTE como antes (genérico):
    env_spec = {
        "robot_data": robot_data,
        "obs": obs_spec,
        "act": act_spec,
    }
    agent_spec = {
        "handles": handles,
        # Si ya migraste a incluir escena aquí, puedes añadir:
        # "scene_name": scene_name,
    }

    params_updates: Dict[str, Any] = {"params_env": {}}
    if laser_count > 0:
        params_updates["params_env"]["laser_observations"] = laser_count

    return robot_name, env_spec, agent_spec, params_updates


# -----------------------------
# ----------- MAIN ------------
# -----------------------------

def main():
    """Interactive entry point for creating a new robot.

    Returns:
        Process exit code (0 on success).
    """
    # Base path (project root)
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Get data from user
    robot_name, env_spec, agent_spec, params_updates = interactive_create_specs(base_path)
    
    print("\n------ Summary ------")
    print(f"Robot name: {robot_name}")
    print(f"Environment specifications: {env_spec}")
    print(f"Agent specifications: {agent_spec}")
    print(f"Params to update in params file: {params_updates}")

    # Perform scaffolding
    print("\nScaffolding files...")
    result = robot_generator.scaffold_robot(base_path, robot_name, env_spec, agent_spec, params_updates =params_updates)

    print("\nDone. Generated paths:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # Optional: Reminder about PYTHONPATH (only print if plugins directory is not importable)
    try:
        import importlib  # noqa
        __import__("plugins.envs")
        __import__("plugins.agents")
    except Exception:
        logging.error("\nReminder: ensure PYTHONPATH includes '<project>/src' so plugins load correctly. e.g., export PYTHONPATH=\"/home/adrian/devel/rl_coppelia/src")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())