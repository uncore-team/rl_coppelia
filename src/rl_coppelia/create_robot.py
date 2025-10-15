import logging
import os
from typing import Any, Dict, List, Optional, Tuple
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
    return {
        "robot": h_robot,
        "robot_baselink": h_baselink,
        "laser": h_laser or ""
    }


def interactive_create_specs() -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Run an interactive session and return all specs.

    Returns:
        (robot_name, env_spec, agent_spec, params_updates)
        - params_updates: extra params to write into params JSON (e.g., laser_observations)
    """
    print("=== Create Robot (Env + Agent) ===")
    robot_name = utils.prompt_str("Robot name")

    # Spaces
    act_spec = _build_action_spec()
    obs_spec, laser_count = _build_obs_spec()

    # Robot data
    robot_data = _build_robot_data()

    # Agent handles
    handles = _build_agent_handles()

    env_spec = {
        "robot_data": robot_data,
        "obs": obs_spec,
        "act": act_spec,
    }
    agent_spec = {
        "handles": handles
    }

    # Params JSON extra updates
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
    robot_name, env_spec, agent_spec, params_updates = interactive_create_specs()

    # Perform scaffolding
    logging.info("\nScaffolding files...")
    result = robot_generator.scaffold_robot(base_path, robot_name, env_spec, agent_spec, params_updates =params_updates)

    logging.info("\n✅ Done. Generated paths:")
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