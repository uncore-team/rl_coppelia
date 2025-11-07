import logging
import math
import os
import sys
import termios
import tty
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from common import robot_generator, utils


# ---------------------------------
# ------ SPEC BUILDING LOGIC -------
# ---------------------------------


# ------- ACTION SPEC -------

def _build_action_spec() -> Dict[str, Any]:
    """Interactively build the action space spec.

    Returns:
        {
          "names": List[str],
          "low":   List[float],
          "high":  List[float]
        }
    """
    print("\n***** Action space *****")

    n_act = utils.prompt_int("Number of action variables", min_val=1)

    # Ask if the user wants to name each variable
    use_names = utils.prompt_confirm("Do you want to name each action variable?", default=True)

    names: List[str] = []
    lows: List[float] = []
    highs: List[float] = []

    for i in range(n_act):
        idx = i + 1
        if use_names:
            name = utils.prompt_str(f"- Name for action {idx}", default=f"act_{idx}")
        else:
            name = f"act_{idx}"

        low = utils.prompt_float(f"-- Lower limit for '{name}'")

        high = utils.prompt_float(f"-- Upper limit for '{name}'",
                                     min_val=low if low is not None else None)

        names.append(name)
        lows.append(float(low))
        highs.append(float(high))

    return {"names": names, "low": lows, "high": highs}


# ------- OBSERVATION SPEC -------

def _build_obs_spec() -> Tuple[Dict[str, Any], int]:
    """Interactively build a generic observation space spec.

    This function does NOT assume any fixed variables. It asks:
      1) Number of non-laser observations. 
      2) For each observation: name, lower bound, upper bound.
      3) Number of laser observations (N_lasers).
      4) Whether to name each laser variable.
      5) A common lower/upper bound for all laser observations.      

    Returns:
        Tuple[dict, int]: (obs_spec, n_lasers)
            obs_spec = {
                "names": List[str],
                "low":   List[float],
                "high":  List[float],
            }
            n_lasers = number of laser observations
    """
    print("\n***** Observation space *****")

    # 1) Non-laser observations
    n_extra = utils.prompt_int("Number of non-laser observations")

    extra_names: List[str] = []
    extra_lows: List[float] = []
    extra_highs: List[float] = []

    for j in range(n_extra):
        # Name for extra variable
        nm = utils.prompt_str(f"- Name for observation {j+1}", default=f"obs{j+1}", allow_empty=False)

        low = utils.prompt_float(f"-- Lower limit for '{nm}'")

        high = utils.prompt_float(
            f"-- Upper limit for '{nm}'",
            min_val=low if low is not None else None
        )

        extra_names.append(nm)
        extra_lows.append(float(low))
        extra_highs.append(float(high))

    # 2) Number of lasers
    n_lasers = utils.prompt_int("Number of laser observations", min_val=0)

    # 3) Laser naming
    laser_names: List[str] = []
    if n_lasers > 0:
        if utils.prompt_confirm("Do you want to name each laser variable?", default=False):
            for i in range(n_lasers):
                nm = utils.prompt_str(f"- Name for laser {i}", default=f"laser_obs_{i}", allow_empty=False)
                laser_names.append(nm)
        else:
            laser_names = [f"laser_obs{i}" for i in range(n_lasers)]

    # 4) Common limits for all lasers (with local back support)
    laser_low, laser_high = None, None
    if n_lasers > 0:

        laser_low = utils.prompt_float("-- Lower limit for all laser observations", min_val=0.0)
        laser_high = utils.prompt_float(
            "-- Upper limit for all laser observations",
            min_val=laser_low if laser_low is not None else None
        )

    # Assemble final spec
    names: List[str] = []
    lows: List[float] = []
    highs: List[float] = []

    if n_extra > 0:
        names.extend(extra_names)
        lows.extend(extra_lows)
        highs.extend(extra_highs)

    if n_lasers > 0:
        names.extend(laser_names)
        lows.extend([laser_low] * n_lasers)    # type: ignore[arg-type]
        highs.extend([laser_high] * n_lasers)  # type: ignore[arg-type]

    obs_spec = {"names": names, "low": lows, "high": highs}
    return obs_spec, n_lasers


# ----------------------------- ROBOT DATA ------------------------------

def _build_robot_data() -> Dict[str, Any]:
    """Interactively build robot-specific data (generic fields)."""
    print("\n***** Robot data *****")
    # Keep it minimal/generic; extend as needed for your project
    wheel_radius = utils.prompt_float("- Wheel radius (m)", default=0.035, min_val=0.0)
    distance_between_wheels   = utils.prompt_float("- Distance between wheels (m)",   default=0.23,  min_val=0.0)

    return {
        "wheel_radius": wheel_radius,
        "distance_between_wheels": distance_between_wheels,
    }


# ------------------------------ SCENE PICK -----------------------------

def _pick_scene(base_path: str) -> str:
    """Pick a .ttt scene from <base_path>/scenes or type a new name."""
    scenes_dir = os.path.join(base_path, "scenes")
    try:
        files = sorted([f for f in os.listdir(scenes_dir) if f.lower().endswith(".ttt")])
    except Exception:
        files = []

    if files:
        for i, f in enumerate(files):
            print(f"  [{i}] {f}")
        idx = utils.prompt_int("Pick scene (index)", min_val=0, max_val=len(files) - 1)
        scene = files[idx]
    else:
        scene = utils.prompt_str("Scene name (will end with .ttt)", default="default.ttt", allow_empty=False)
        if not scene.endswith(".ttt"):
            scene += ".ttt"

    return scene


# --------------------------- AGENT HANDLES -----------------------------

def _build_agent_handles() -> Dict[str, Any]:
    """Ask for agent handles (robot, base, laser)."""

    def ensure_leading_slash(s: str) -> str:
        return s if (not s or s.startswith("/")) else "/" + s

    print("\n***** Agent handles *****")
    rh = utils.prompt_str("robot_handle", default="/Turtlebot2", allow_empty=False)
    rb = utils.prompt_str("robot_base_handle", default="/Turtlebot2/base_link_respondable", allow_empty=False)
    lh = utils.prompt_str("laser_handle", default="/Turtlebot2/fastHokuyo_ROS2", allow_empty=False)

    return {
        "robot_handle": ensure_leading_slash(rh),
        "robot_base_handle": ensure_leading_slash(rb),
        "laser_handle": ensure_leading_slash(lh),
    }


def interactive_create_specs(base_path: str) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Run an interactive session (Ctrl+B = back, Ctrl+C = cancel) and return all specs.

    Workflow:
      1) Ask robot name
      2) Build action space
      3) Build observation space
      4) Build robot data
      5) Select scene
      6) Build agent handles

    Returns:
        (robot_name, env_spec, agent_spec, params_updates)
    """
    print("------ Create Robot (Env + Agent) ------")
    print("ℹ️  Ctrl+B: go back · Ctrl+C: cancel\n")

    # Shared state
    robot_name: str = ""
    act_spec: Dict[str, Any] = {}
    obs_spec: Dict[str, Any] = {}
    robot_data: Dict[str, Any] = {}
    scene_name: str = ""
    handles: Dict[str, Any] = {}
    laser_count: int = 0

    # ---------- Wizard steps ----------
    def step_robot_name():
        nonlocal robot_name
        robot_name = utils.prompt_str("Robot name", allow_empty=False)

    def step_action_spec():
        nonlocal act_spec
        act_spec = _build_action_spec()  # usa utils.prompt_* (Ctrl+B/C integrados)

    def step_obs_spec():
        nonlocal obs_spec, laser_count
        obs_spec, laser_count = _build_obs_spec()

    def step_robot_data():
        nonlocal robot_data
        robot_data = _build_robot_data()

    def step_scene():
        nonlocal scene_name
        print("\n***** Scene selection *****")
        scene_name = _pick_scene(base_path)
        print(f"Scene name: {scene_name}")

    def step_handles():
        nonlocal handles
        handles = _build_agent_handles()

    steps: list[Callable[[], None]] = [
        step_robot_name,
        step_action_spec,
        step_obs_spec,
        step_robot_data,
        step_scene,
        step_handles,
    ]

    # ---------- Run wizard with Ctrl+B/C support ----------
    i = 0
    try:
        while 0 <= i < len(steps):
            try:
                steps[i]()
                i += 1
            except utils.BackSignal:
                # Move one step back — no “jumping” multiple steps
                i = max(0, i - 1)
    except KeyboardInterrupt:
        print("\n⛔ Operation cancelled by user (Ctrl+C).")
        raise

    # ---------- Build output ----------
    env_spec = {
        "robot_data": robot_data,
        "obs": obs_spec,
        "act": act_spec,
    }
    agent_spec = {
        "handles": handles,
        "scene_name": scene_name if scene_name.endswith(".ttt") else scene_name + ".ttt",
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