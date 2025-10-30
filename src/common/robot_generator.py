from __future__ import annotations
import json
import os
import shutil
import tempfile
import textwrap
import re
from typing import Dict, Any, List, Optional, Tuple
import warnings

# ------------------------------------
# ------- AUXILIARY FUNCTIONS --------
# ------------------------------------


def _snake_to_camel(name: str) -> str:
    """Convert a snake_case or kebab-case string to CamelCase.
    Examples:
        "burgerBot" -> "BurgerBot"
        "my_new-bot" -> "MyNewBot"
    """
    parts = [p for p in re.split(r"[_\- ]+", name) if p]
    return "".join(s.capitalize() for s in parts)


def _resolve_space_from_spec(spec: dict) -> tuple[int, list[float], list[float], list[str]]:
    """Resolve a Box-like space from the wizard spec format.

    Expected format:
        {
          "names": [str, ...],
          "low":   [float, ...],
          "high":  [float, ...]
        }

    Returns:
        (dim, lows, highs, names)
    """
    if not all(k in spec for k in ("names", "low", "high")):
        raise ValueError("Spec must contain 'names', 'low', and 'high' lists.")

    names = spec["names"]
    lows = spec["low"]
    highs = spec["high"]

    if not (isinstance(names, list) and isinstance(lows, list) and isinstance(highs, list)):
        raise ValueError("'names', 'low', and 'high' must be lists.")
    if not (len(names) == len(lows) == len(highs)):
        raise ValueError("Lengths of 'names', 'low', and 'high' must match.")
    if len(names) == 0:
        raise ValueError("Empty observation/action space definition.")

    # Optional sanity check
    for i, (lo, hi) in enumerate(zip(lows, highs)):
        if lo > hi:
            raise ValueError(
                f"Invalid range for '{names[i]}' (index {i}): low > high ({lo} > {hi})."
            )

    return len(names), [float(x) for x in lows], [float(x) for x in highs], [str(n) for n in names]


def _assign_handle(var: str, path: str) -> str:
    '''
    Helper to generate a line that assigns a Coppelia handle to self.var.
    If path is empty, r.
    '''
    if not path:
        warnings.warn(f"No path provided for handle '{var}'; leaving unassigned.")
        return f"# self.{var} = sim.getObject('/path/to/{var}')"
    return f"self.{var} = sim.getObject('{path}')"


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update dict `dst` with keys from `src`."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _derive_generic_env_fields_from_spec(env_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Build generic env fields (dim_* and *_limits) from the unified env_spec.

    Args:
        env_spec: Dict with:
            - "obs": named or size-based space
            - "act": named or size-based space
            - "robot_data": optional, handled elsewhere (params_scene)

    Returns:
        Dict with fields suitable for params_env:
            {
              "dim_action_space": int,
              "action_bottom_limits": List[float],
              "action_upper_limits": List[float],
              "dim_observation_space": int,
              "observation_bottom_limits": List[float],
              "observation_upper_limits": List[float]
            }
    """
    obs_def = env_spec.get("obs", {})
    act_def = env_spec.get("act", {})

    dim_obs, obs_lows, obs_highs, obs_names = _resolve_space_from_spec(obs_def)
    dim_act, act_lows, act_highs, act_names = _resolve_space_from_spec(act_def)

    return {
        "dim_action_space": dim_act,
        "action_names": act_names,
        "action_bottom_limits": act_lows,
        "action_upper_limits": act_highs,
        "dim_observation_space": dim_obs,
        "observation_names": obs_names,
        "observation_bottom_limits": obs_lows,
        "observation_upper_limits": obs_highs,
    }


def _extract_agent_fields_from_agent_spec(agent_spec: Dict[str, Any]) -> Dict[str, str]:
    """Extracts agent-related fields (handles + scene name) from agent_spec."""
    def _ensure_leading_slash(s: str) -> str:
        if not isinstance(s, str) or not s:
            return s
        return s if s.startswith("/") else "/" + s

    result: Dict[str, str] = {}

    # Get nested handles dict if present 
    handles = agent_spec.get("handles", agent_spec)

    # Handles (robot, base, laser)
    for key in ("robot_handle", "robot_base_handle", "laser_handle"):
        val = handles.get(key)
        if val:
            result[key] = _ensure_leading_slash(val)

    # Scene name
    scene_name = agent_spec.get("scene_name")
    if scene_name:
        if not scene_name.endswith(".ttt"):
            scene_name += ".ttt"
        result["scene_name"] = scene_name

    return result


def _create_generic_params_file(
    base_path: str,
    robot_name: str,
    env_spec: Dict[str, Any],
    agent_spec: Dict[str, Any],
    updates: Optional[Dict[str, Any]] = None,
    *,
    template_filename: str = "params_default_file.json"
) -> str:
    """Create a new params JSON from the default template (safe copy first).

    Steps:
      1) Copy `<base_path>/configs/<template_filename>` into
         `<base_path>/configs/params_default_file_<robot_name>.json`.
      2) Load that copy.
      3) Inject generic space fields (dim_*, *_limits) from `env_spec`.
      4) Inject `params_scene` from `env_spec["robot_data"]`.
      5) Inject `params_train` handles from `agent_spec`.
      6) Apply explicit `updates` if provided.

    Args:
        base_path: Project root.
        robot_name: Robot name (used for output filename).
        env_spec: Unified env spec (robot_data, obs, act).
        updates: Optional explicit overrides for sections.
        template_filename: Base template JSON name in configs/.

    Returns:
        Absolute path to the new JSON file.
    """
    configs_dir = os.path.join(base_path, "configs")
    template_path = os.path.join(configs_dir, template_filename)
    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")

    # --- Step 1: copy original to new file -----------------------------------
    target_name = f"params_default_file_{robot_name}.json"
    target_path = os.path.join(configs_dir, target_name)
    shutil.copyfile(template_path, target_path)

    # --- Step 2: load the new copy -------------------------------------------
    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure required sections exist
    data.setdefault("params_env", {})
    data.setdefault("params_scene", {})
    data.setdefault("params_train", {})
    data.setdefault("params_test", {})

    # --- Step 3: inject generic env fields -----------------------------------
    generic_env = _derive_generic_env_fields_from_spec(env_spec)
    _deep_update(data["params_env"], generic_env)

    # --- Step 4: inject robot_data -------------------------------------------
    robot_data = env_spec.get("robot_data")
    if isinstance(robot_data, dict) and robot_data:
        _deep_update(data["params_scene"], robot_data)

    # --- Step 5: Inject training handles (robot/laser handles)
    agent_fields = _extract_agent_fields_from_agent_spec(agent_spec)
    if agent_fields:
        _deep_update(data["params_train"], agent_fields)

    # --- Step 6: apply explicit overrides ------------------------------------
    if updates:
        _deep_update(data, updates)

    # --- Step 7: write changes atomically ------------------------------------
    fd, tmp_path = tempfile.mkstemp(prefix=f".{target_name}.", dir=configs_dir, text=True)
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as w:
            json.dump(data, w, ensure_ascii=False, indent=4)
            w.write("\n")
        os.replace(tmp_path, target_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise

    return target_path



# ------------------------------------
# --------------- ENVS ---------------
# ------------------------------------

def generate_env_code(robot_name: str, spec: Dict[str, Any]) -> str:
    """Build the Python source for the new Env class from a unified spec.

    The spec supports two shapes:
        - Named variables (recommended for readability)
        - Size + broadcastable bounds

    Spec examples:
        # A) Named variables
        spec = {
            "obs": {
                "vars": [
                    {"name": "distance", "low": 0.0, "high": 10.0},
                    {"name": "angle",    "low": -3.1416, "high": 3.1416}
                ]
            },
            "act": {
                "vars": [
                    {"name": "v",     "low": -1.0, "high": 1.0},
                    {"name": "omega", "low": -2.0, "high": 2.0}
                ]
            }
        }

        # B) Size + broadcastable bounds (scalars or lists)
        spec = {
            "obs": {"size": 3, "low": [0, -3.14, 0], "high": [10, 3.14, 100]},
            "act": {"size": 2, "low": -1.0, "high": 1.0}
        }

    Args:
        robot_name: Robot name (e.g., "burgerBot").
        spec: Dict with 'obs' and 'act' definitions (see examples).

    Returns:
        Python file content as a string.
    """
    class_name = f"{_snake_to_camel(robot_name)}Env"

    # Get obs and act info from spec dict
    obs_def = spec.get("obs", {})
    act_def = spec.get("act", {})

    # Resolve spaces
    _obs_dim, obs_lows, obs_highs, obs_names = _resolve_space_from_spec(obs_def)
    _act_dim, act_lows, act_highs, act_names = _resolve_space_from_spec(act_def)

    # Stringify for code
    obs_low_arr = ", ".join(f"{float(x):.10g}" for x in obs_lows)
    obs_high_arr = ", ".join(f"{float(x):.10g}" for x in obs_highs)
    act_low_arr = ", ".join(f"{float(x):.10g}" for x in act_lows)
    act_high_arr = ", ".join(f"{float(x):.10g}" for x in act_highs)
    obs_names_list = ", ".join(repr(n) for n in obs_names) if obs_names else ""
    act_names_list = ", ".join(repr(n) for n in act_names) if act_names else ""

    doc_obs_names = f"names = [{obs_names_list}]" if obs_names else "names = []  # unnamed (size-based)"
    doc_act_names = f"names = [{act_names_list}]" if act_names else "names = []  # unnamed (size-based)"

    return textwrap.dedent(f'''\
        """Auto-generated environment for '{robot_name}'.

        This Env inherits from CoppeliaEnv and defines a Box observation space based on GUI input.
        """

        import math
        import numpy as np
        from gymnasium import spaces

        from common.coppelia_envs import CoppeliaEnv

        class {class_name}(CoppeliaEnv):
            def __init__(self, params_scene, params_env, comms_port=49054):
                """Custom environment for '{robot_name}'.

                Args:
                    params_scene (dict): Scene parameters.
                    params_env (dict): Environment parameters.
                    comms_port (int, optional): Port for communication with the agent. Defaults to 49054.

                Notes:
                    The action and observation spaces are a Box with the variables specified at creation time:
                    Observation space {doc_obs_names}.
                    Action space {doc_act_names}.

                """
                super({class_name}, self).__init__(params_scene, params_env, comms_port)

                # Define observation space
                self.observation_space = spaces.Box(
                    low=np.array([{obs_low_arr}], dtype=np.float32),
                    high=np.array([{obs_high_arr}], dtype=np.float32),
                    dtype=np.float32
                )

                # Define action space
                self.action_space = spaces.Box(
                    low=np.array([{act_low_arr}], dtype=np.float32),
                    high=np.array([{act_high_arr}], dtype=np.float32),
                    dtype=np.float32
                )

            # NOTE: Implement other environment-specific methods if needed,
            # such as _get_observation(), _compute_reward(), step(), reset(), etc.
    ''')


def generate_env_plugin_code(robot_name: str) -> str:
    """Build the plugin source that registers the factory for this robot.

    Args:
        robot_name: Robot name.

    Returns:
        Python file content as a string.
    """
    class_name = f"{_snake_to_camel(robot_name)}Env"
    return textwrap.dedent(f'''\
        """Plugin to register '{robot_name}' robot environment.

        Loaded on the RL side. It registers a VecEnv factory on import.
        """

        from __future__ import annotations
        from common.rl_coppelia_manager import RLCoppeliaManager
        from plugins.envs import register_env
        from stable_baselines3.common.env_util import make_vec_env
        from robots.{robot_name}.envs import {class_name}

        def make_env(manager: RLCoppeliaManager):
            """Create a VecEnv instance for '{robot_name}'.

            Args:
                manager: The current RLCoppeliaManager instance.

            Returns:
                A vectorized environment (VecEnv) suitable for training/testing.
            """
            return make_vec_env(
                {class_name},
                n_envs=1,
                monitor_dir=manager.log_monitor,
                env_kwargs={{
                    "params_scene": self.params_scene,
                    "params_env": manager.params_env,
                    "comms_port": manager.free_comms_port,
                }},
            )

        # Register on module import
        register_env("{robot_name}", make_env)
    ''')


def create_robot_env_and_plugin(base_path: str, robot_name: str, spec: dict) -> tuple[str, str]:
    """Create the env module and the plugin for a new robot.

    Args:
        base_path: Project root (the parent of 'robots' and 'src').
        robot_name: New robot name (folder-friendly, e.g., 'myNewBot').
        spec: Dict as returned by NewEnvDialog.get_spec():
              {{
                "include_time": bool,
                "vars": [{{"name": str, "low": float, "high": float}}, ...]
              }}

    Returns:
        (env_file_path, plugin_file_path)
    """
    os.makedirs(os.path.join(base_path, "robots", robot_name), exist_ok=True)
    # robots/<robot>/__init__.py
    pkg_init = os.path.join(base_path, "robots", robot_name, "__init__.py")
    if not os.path.exists(pkg_init):
        with open(pkg_init, "w", encoding="utf-8") as f:
            f.write("# Package for robot: " + robot_name + "\n")

    # robots/<robot>/envs.py
    env_path = os.path.join(base_path, "robots", robot_name, "envs.py")
    env_src = generate_env_code(robot_name, spec)
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(env_src)

    # src/plugins/envs/__init__.py
    plugins_pkg = os.path.join(base_path, "src", "plugins", "envs")
    os.makedirs(plugins_pkg, exist_ok=True)
    init_plugins = os.path.join(plugins_pkg, "__init__.py")
    if not os.path.exists(init_plugins):
        with open(init_plugins, "w", encoding="utf-8") as f:
            f.write('"""Env plugins package (RL side). Modules here should register on import."""\n')

    # src/plugins/envs/<robot>.py
    plugin_path = os.path.join(plugins_pkg, f"{robot_name}.py")
    plugin_src = generate_env_plugin_code(robot_name)
    with open(plugin_path, "w", encoding="utf-8") as f:
        f.write(plugin_src)

    return env_path, plugin_path



# ------------------------------------
# -------------- AGENTS --------------
# ------------------------------------

def generate_agent_code(robot_name: str, spec: dict) -> str:
    """Build the Python source for the new Agent subclass.

    Args:
        robot_name: Folder-friendly robot name (e.g., "burgerBot").
        spec: Dict with needed keys for creating an agent:
            {
              "handles": {
                  "robot": "/Turtlebot2",
                  "robot_baselink": "/Turtlebot2/base_link_respondable",
                  "laser": "/Turtlebot2/fastHokuyo_ROS2"   # optional
              }
            }

    Returns:
        Python file content as a string.
    """
    class_name = f"{_snake_to_camel(robot_name)}Agent"

    # Extract agent values from spec
    h = (spec or {}).get("handles", {})
    robot = h.get("robot_handle", "")
    robot_bl = h.get("robot_base_handle", "")
    laser = h.get("laser_handle", "")

    robot_line = _assign_handle("robot", robot)
    robot_bl_line = _assign_handle("robot_baselink", robot_bl)
    laser_line = _assign_handle("laser", laser)

    return textwrap.dedent(f'''\
        """Auto-generated agent subclass for '{robot_name}'.

        This class wires only the robot-specific handles. All generic logic and
        additional scene objects are handled by the base CoppeliaAgent.
        """

        import logging

        from common.coppelia_agents import CoppeliaAgent

        class {class_name}(CoppeliaAgent):
            def __init__(self, sim, params_scene, params_env, paths, file_id, verbose, comms_port=49054):
                """Custom agent for {robot_name} (auto-generated).

                Args:
                    sim: Coppelia object for handling the scene's objects.
                    params_scene (dict): Robot parameters
                    params_env (dict): Environment parameters.
                    paths (dict): Project paths.
                    file_id (str): Experiment/session ID.
                    verbose (int): Verbosity.
                    comms_port (int): Port for RL-side communications. Defaults to 49054.
                """
                super({class_name}, self).__init__(sim, params_scene, params_env, paths, file_id, verbose, comms_port)

                # --- Scene handles which are specific for this robot ---
                {robot_line}
                {robot_bl_line}
                {laser_line}
                self.handle_laser_get_observation_script=sim.getScript(1,self.laser,'laser_get_observations')
                self.handle_robot_scripts = sim.getScript(1, self.robot)
                
                logging.info(f"{class_name} created successfully using port {{comms_port}}.")
    ''')



def generate_agent_plugin_code(robot_name: str) -> str:
    """Build a small factory module for the Agent signature."""

    class_name = f"{_snake_to_camel(robot_name)}Agent"
    return textwrap.dedent(f'''\
        """Factory for '{robot_name}' agent (auto-generated)."""

        from plugins.agents import register_agent
        from robots.{robot_name}.agent import {class_name}

        def make_agent(sim, params_scene, params_env, paths, file_id, verbose, comms_port=49054):
            """Return an instance of the robot-specific Agent.

            Args:
                sim: Coppelia API object.
                params_scene (dict): Robot parameters
                params_env (dict): Environment parameters.
                paths (dict): Project paths.
                file_id (str): Experiment/session identifier.
                verbose (int): Verbosity level.
                comms_port (int): RL comms port.

            Returns:
                {class_name}: Configured agent instance.
            """
            return {class_name}(sim, params_scene, params_env, paths, file_id, verbose, comms_port)
        
        register_agent("{robot_name}", make_agent)
    ''')


def create_robot_agent_and_plugin(base_path: str, robot_name: str, spec: dict) -> tuple[str, str]:
    """Create the Agent module and its plugin for a new robot (Coppelia side).

    Args:
        base_path: Project root (parent of 'src' and 'robots').
        robot_name: New robot name (e.g., 'myNewBot').
        spec: Dict for agent generator (see generate_agent_code).

    Returns:
        (agent_file_path, agent_plugin_file_path)
    """
    # robots/<robot>/agent.py
    robot_dir = os.path.join(base_path, "robots", robot_name)
    os.makedirs(robot_dir, exist_ok=True)

    pkg_init = os.path.join(robot_dir, "__init__.py")
    if not os.path.exists(pkg_init):
        with open(pkg_init, "w", encoding="utf-8") as f:
            f.write("# Package for robot: " + robot_name + "\\n")

    agent_path = os.path.join(robot_dir, "agent.py")
    agent_src = generate_agent_code(robot_name, spec)
    with open(agent_path, "w", encoding="utf-8") as f:
        f.write(agent_src)

    # src/plugins/agents/<robot>.py
    plugins_pkg = os.path.join(base_path, "src", "plugins", "agents")
    os.makedirs(plugins_pkg, exist_ok=True)
    init_plugins = os.path.join(plugins_pkg, "__init__.py")
    if not os.path.exists(init_plugins):
        with open(init_plugins, "w", encoding="utf-8") as f:
            f.write('"""Agent plugins package (Coppelia side). Registers factories on import."""\\n')

    plugin_path = os.path.join(plugins_pkg, f"{robot_name}.py")
    plugin_src = generate_agent_plugin_code(robot_name)
    with open(plugin_path, "w", encoding="utf-8") as f:
        f.write(plugin_src)

    return agent_path, plugin_path


# ------------------------------------
# --------- ONE-SHOT SCAFFOLD --------
# ------------------------------------

def scaffold_robot(
    base_path: str,
    robot_name: str,
    env_spec: dict,
    agent_spec: dict,
    *,
    params_updates: dict | None = None,
    template_filename: str = "params_default_file.json"
) -> dict:
    """Create Env+plugin, Agent+plugin, and a params JSON for the new robot.

    Args:
        base_path: Project root (parent of 'src' and 'robots').
        robot_name: New robot name.
        env_spec: Unified env spec (robot_data, obs, act).
        agent_spec: Agent spec (handles).
        params_updates: Optional overrides to merge into the params JSON (e.g., {"params_env": {"laser_observations": 8}}).
        template_filename: Template JSON in <base_path>/configs.
        drop_legacy_limits: Remove legacy per-axis limit keys from params_env.

    Returns:
        dict with generated paths.
    """
    env_path, env_plugin_path = create_robot_env_and_plugin(base_path, robot_name, env_spec)
    agent_path, agent_plugin_path = create_robot_agent_and_plugin(base_path, robot_name, agent_spec)

    # Create params file for the new robot
    params_path = _create_generic_params_file(
        base_path=base_path,
        robot_name=robot_name,
        env_spec=env_spec,
        agent_spec=agent_spec,
        updates=params_updates,
        template_filename=template_filename
    )
    return {
        "env": env_path,
        "env_plugin": env_plugin_path,
        "agent": agent_path,
        "agent_plugin": agent_plugin_path,
        "params_file": params_path,
    }