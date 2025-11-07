# Robot_Script: path helpers (ctrlPt polyline)
# --------------------------------------------
# Exposes two functions called by Agent_Script:
#   - rp_init(path_alias, step_m): sample the polyline under 'RecordedPath' dummy
#   - rp_tp(i): teleport the robot base to the i-th sampled pose (x,y,z0,yaw)

import logging
import os
import shutil
import sys
import math
import numpy as np

# Locate rl_coppelia installation and append the source folder to sys.path
project_folder = shutil.which("rl_coppelia")
common_paths = [
        os.path.expanduser("~/Documents"),  # Search in Documents folder
        os.path.expanduser("~/Downloads"),  # Search in Downloads folder
        os.path.expanduser("~/devel"),
        "/opt", "/usr/local", "/home"     # Common system directories
    ]

project_path = None

for path in common_paths:
    for root, dirs, files in os.walk(path):
        if "rl_coppelia" in dirs: 
            project_path = os.path.join(root, "rl_coppelia") 
            break  
    if project_path:
        break
sys.path.append(os.path.abspath(os.path.join(project_path,"src")))

from common import utils
from common.coppelia_agents import BurgerBotAgent, TurtleBotAgent

# Uncomment next line for using ROS, and enable simROS2_flag
# simROS2=require('simROS2')
simROS2_flag = False

# Handles
robotHandle = -1
footprintHandle = -1
robot_initial_pose_handle = -1
robotAlias = ""
robot_alias = ""        # received from params file
robot_base_alias = ""   # received from params file
laser_alias = ""        # received from params file
laserHandle = -1

# Other control variables
verbose = None

# Robot variables
distance_between_wheels = None  # meters
wheel_radius = None            # meters 
linealSpeed = -1
angularSpeed = -1


# ----------------------------------------------------------------------
# ------------- RecordedPath (Dummy + ctrlPt*) polyline helpers --------
# ----------------------------------------------------------------------

def _get_ctrlpts_polyline(sim, recorded_path_handle):
    """Collect ctrlPt children under a 'RecordedPath' dummy and return their world positions.

    Behavior:
      - Scans descendants of 'recorded_path_handle' and keeps Dummies whose alias starts with 'ctrlPt'.
      - Sorts them by numeric suffix (ctrlPt0, ctrlPt1, ...). If no suffix, keeps name order.
      - Returns a list of points [[x,y,z], ...] in world coordinates.

    Raises:
      RuntimeError if no ctrlPt children are found.
    """
    # Gather all descendant dummies
    dummies = sim.getObjectsInTree(recorded_path_handle, sim.object_dummy_type, 1) or []
    pts = []
    for h in dummies:
        alias = sim.getObjectAlias(h, 0) or ""
        if alias.startswith("ctrlPt"):
            # Extract numeric suffix if any (ctrlPt12 -> 12), else None
            suf = alias[6:]  # characters after 'ctrlPt'
            try:
                idx = int(suf) if suf else None
            except Exception:
                idx = None
            pos = sim.getObjectPosition(h, sim.handle_world)  # [x,y,z]
            pts.append((idx, alias, [float(pos[0]), float(pos[1]), float(pos[2])]))

    if not pts:
        raise RuntimeError("No ctrlPt* dummies found under the provided RecordedPath object.")

    # Sort: first by idx (if present), then by alias for stability
    pts.sort(key=lambda t: (999999 if t[0] is None else t[0], t[1]))
    return [p for _, __, p in pts]


def _sample_polyline_world_poses_by_count(points_w, n_samples: int, n_extra_poses: int = 0, delta_deg: float = 10.0):
    """Sample exactly n_samples points evenly (by arc-length) along the polyline,
    and optionally augment each point with extra yaw-only variations.

    Args:
        points_w (list[tuple[float, float, float]]): Polyline points in world frame (x, y, z).
        n_samples (int): Number of evenly spaced samples along the polyline.
        n_extra_poses (int, optional): Number of extra yaw-only variants on each side
            of the base yaw. If N>0, for each sampled pose we add +k*delta and -k*delta
            (k=1..N). Defaults to 0.
        delta_deg (float, optional): Angle step in degrees for the extra yaw variants.
            Defaults to 10.0.

    Returns:
        list[tuple[float, float, float, float]]: List of (x, y, z, yaw) world poses.

    Notes:
        - Base yaw is the direction of the local segment (atan2(dy, dx)).
        - Yaw normalization uses atan2(sin, cos) to keep angles within [-pi, pi].
        - If the polyline is degenerate, all samples repeat the first point with yaw 0.
    """
    import numpy as np

    def _normalize_angle(a: float) -> float:
        """Normalize angle to [-pi, pi]."""
        return math.atan2(math.sin(a), math.cos(a))

    n_samples = max(1, int(n_samples))
    poses = []

    # Handle short/degenerate polylines
    if len(points_w) < 2:
        x, y, z = points_w[0]
        poses = [(float(x), float(y), float(z), 0.0)] * n_samples
    else:
        # Accumulated lengths
        cum = [0.0]
        for i in range(len(points_w) - 1):
            x1, y1, _ = points_w[i]
            x2, y2, _ = points_w[i + 1]
            cum.append(cum[-1] + math.hypot(x2 - x1, y2 - y1))
        total = cum[-1]

        if total <= 1e-12:
            x, y, z = points_w[0]
            poses = [(float(x), float(y), float(z), 0.0)] * n_samples
        else:
            # Target arc-lengths (include endpoints if n_samples>=2)
            if n_samples == 1:
                targets = [total * 0.5]
            else:
                targets = [k * (total / (n_samples - 1)) for k in range(n_samples)]

            for s in targets:
                # Locate segment
                i = min(len(cum) - 2, max(0, int(np.searchsorted(cum, s, side="right") - 1)))
                s0, s1 = cum[i], cum[i + 1]
                p1, p2 = points_w[i], points_w[i + 1]
                seglen = s1 - s0

                if seglen <= 1e-12:
                    x, y, z = p1
                    yaw = 0.0
                    # Try to infer yaw from valid neighbors
                    found = False
                    for j in range(i - 1, -1, -1):
                        if (cum[j + 1] - cum[j]) > 1e-12:
                            dx = points_w[j + 1][0] - points_w[j][0]
                            dy = points_w[j + 1][1] - points_w[j][1]
                            yaw = math.atan2(dy, dx)
                            found = True
                            break
                    if not found:
                        for j in range(i + 1, len(points_w) - 1):
                            if (cum[j + 1] - cum[j]) > 1e-12:
                                dx = points_w[j + 1][0] - points_w[j][0]
                                dy = points_w[j + 1][1] - points_w[j][1]
                                yaw = math.atan2(dy, dx)
                                break
                else:
                    t = (s - s0) / seglen
                    x = p1[0] + t * (p2[0] - p1[0])
                    y = p1[1] + t * (p2[1] - p1[1])
                    z = p1[2] + t * (p2[2] - p1[2])
                    yaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

                poses.append((float(x), float(y), float(z), float(yaw)))

    # ---- Augment each pose with extra yaw-only variants --------------------
    delta_rad = math.radians(float(delta_deg))
    n_extra = max(0, int(n_extra_poses))
    augmented = []

    for (x, y, z, yaw) in poses:
        # Base
        augmented.append((x, y, z, _normalize_angle(yaw)))
        if n_extra > 0:
            # +10°, +20°, ..., +N*10°
            for k in range(1, n_extra + 1):
                augmented.append((x, y, z, _normalize_angle(yaw + k * delta_rad)))
            # -10°, -20°, ..., -N*10°
            for k in range(1, n_extra + 1):
                augmented.append((x, y, z, _normalize_angle(yaw - k * delta_rad)))

    print(f"Robot positions to be tested: {len(augmented)}")
    return augmented


def _sample_polyline_world_poses(points_w, step_m: float, keep_last: bool = True):
    """Sample a polyline (list of world XYZ points) at fixed arc-length step.

    Returns:
      list of tuples (x, y, z, yaw), where yaw comes from the local segment tangent.

    Notes:
      - If keep_last is True, the final vertex is included even if it doesn't land exactly on a step.
      - Duplicate consecutive points (zero-length segments) are skipped/handled gracefully.
    """
    if len(points_w) < 2:
        return []

    # Build cumulative arc-length along the piecewise-linear chain
    cum = [0.0]
    for i in range(len(points_w) - 1):
        x1, y1, _ = points_w[i]
        x2, y2, _ = points_w[i + 1]
        seglen = math.hypot(x2 - x1, y2 - y1)
        cum.append(cum[-1] + seglen)

    total = cum[-1]
    if total <= 1e-12:
        return []

    # Sampling positions along s = 0..total
    poses = []
    n = max(1, int(math.floor(total / step_m)))
    samples = [k * step_m for k in range(n + 1)]
    if keep_last and samples[-1] < total - 1e-9:
        samples.append(total)

    # For each target arc-length, find the segment and interpolate
    for s in samples:
        # Locate segment index such that cum[i] <= s <= cum[i+1]
        i = min(len(cum) - 2, max(0, int(np.searchsorted(cum, s, side="right") - 1)))
        s0, s1 = cum[i], cum[i + 1]
        p1 = points_w[i]
        p2 = points_w[i + 1]
        seglen = s1 - s0

        if seglen <= 1e-12:
            # Degenerate: use p1 and try to borrow a direction from neighbors
            x, y, z = p1
            yaw = 0.0
            # Search left
            found = False
            for j in range(i - 1, -1, -1):
                if (cum[j + 1] - cum[j]) > 1e-12:
                    dx = points_w[j + 1][0] - points_w[j][0]
                    dy = points_w[j + 1][1] - points_w[j][1]
                    yaw = math.atan2(dy, dx)
                    found = True
                    break
            # Or search right
            if not found:
                for j in range(i + 1, len(points_w) - 1):
                    if (cum[j + 1] - cum[j]) > 1e-12:
                        dx = points_w[j + 1][0] - points_w[j][0]
                        dy = points_w[j + 1][1] - points_w[j][1]
                        yaw = math.atan2(dy, dx)
                        break
        else:
            t = (s - s0) / seglen
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            z = p1[2] + t * (p2[2] - p1[2])
            yaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

        poses.append((float(x), float(y), float(z), float(yaw)))
    return poses


# -------------------------------
# ------------- ROS -------------
# -------------------------------

def getTransformStamped(objHandle,name,relTo,relToName):
    global sim
    # This function retrieves the stamped transform for a specific object
    #t=simROS2.getSystemTime() --> from simulatorIDE opening
    t=simROS2.getSimulationTime()  # type: ignore
    p=sim.getObjectPosition(objHandle,relTo)
    o=sim.getObjectQuaternion(objHandle,relTo)
    return {
        "header": {
            "stamp": t,
            "frame_id": relToName
        },
        "child_frame_id": name,
        "transform": {
            "translation": {"x": p[0], "y": p[1], "z": p[2]},
            "rotation": {"x": o[0], "y": o[1], "z": o[2], "w": o[3]}
        }
    }


# -------------------------------
# ------- MAIN FUNCTIONS --------
# -------------------------------

def sysCall_init():
    global sim, simROS2_flag, verbose
    global robotHandle, robotAlias, footprintHandle, laserHandle, motorLeft, motorRight
    global robot_initial_pose_handle
    global subscriber_twist, publisher_odometry, publisher_ground_truth
    global robot_alias, robot_base_alias, laser_alias

    sim = require('sim')    # type: ignore

    # HANDLES
    robotHandle = sim.getObject('..')                   # the robot
    robotAlias = sim.getObjectAlias(robotHandle,3)      # robot name
    if f"/{robotAlias}" != robot_alias:
        logging.warning(f"Warning: robot alias from params file '{robot_alias}' does not match the robot alias in the scene '/{robotAlias}'")
    robotHandle = sim.getObject(robot_base_alias) 
    footprintHandle= sim.getObject(robot_base_alias)
    laserHandle = sim.getObject(laser_alias)
    motorLeft = sim.getObject("/wheel_left_joint")
    motorRight = sim.getObject("/wheel_right_joint")
    
    # ROS2 PUBLISHERS AND SUBSCRIBERS
    if simROS2_flag:
        try:
            subscriber_twist = simROS2.createSubscription(robotAlias + '/cmd_vel', 'geometry_msgs/msg/Twist', 'cmd_vel')    # type: ignore
            publisher_odometry = simROS2.createPublisher(robotAlias + '/odom', 'geometry_msgs/msg/Pose')                    # type: ignore      
            publisher_ground_truth = simROS2.createPublisher(robotAlias + '/ground_truth', 'geometry_msgs/msg/Pose')        # type: ignore
        except:
            raise RuntimeError('[Turtle] simROS2 not available. Unable to set Publishers and Subscribers')


def sysCall_actuation():
    global sim, simROS2_flag
    global robotHandle, robotAlias, footprintHandle, laserHandle, robot_initial_pose_handle
    # publish TFs
    if simROS2_flag:
        try:
            simROS2.sendTransform(getTransformStamped(footprintHandle, 'base_footprint_' + robotAlias, -1, 'map'))  # type: ignore
            simROS2.sendTransform(getTransformStamped(footprintHandle, 'base_footprint_' + robotAlias, robot_initial_pose_handle, 'odom_' + robotAlias))    # type: ignore
            simROS2.sendTransform(getTransformStamped(robotHandle, 'base_link_' + robotAlias, footprintHandle, 'base_footprint_' + robotAlias)) # type: ignore
            # TF: robot --> Laser (static)
            simROS2.sendTransform(getTransformStamped(laserHandle, 'laser_scan_' + robotAlias, robotHandle, 'base_link_' + robotAlias)) # type: ignore# type: ignore
        except:
            raise RuntimeError('[Turtle] simROS2 not available. Unable to publish TFs')


def sysCall_sensing():
    pass


def sysCall_cleanup():
    # Do some clean-up here if needed
    pass


# --------------------------------------------
# --------------- Exposed API ----------------
# --------------------------------------------


def rp_init(n_samples, sample_step_m, n_extra_poses, path_name):
    """Initialize path sampling.    # TODO: UPdate docstring

    Args:
        inStrings[0]: path alias (e.g., "/RecordedPath")
        inInts[0]   : OPTIONAL number of samples (N). If >0, overrides step_m.
        inFloats[0] : OPTIONAL step_m (meters) used only if N is not provided.

    Returns:
        outInts[0]: number of sampled poses
    """
    print(f"Trying to sample the path using a path alias: {path_name}, with {n_samples} samples, or sampling it with a {sample_step_m}m step.")
    path_alias = path_name if path_name else "/RecordedPath"
    N = n_samples if n_samples else 0

    if sample_step_m == 0:
        step_m = 0.01
    else:
        step_m = sample_step_m

    path_handle = sim.getObject(path_alias)
    pts = _get_ctrlpts_polyline(sim, path_handle)

    if N > 0:
        pos_samples = _sample_polyline_world_poses_by_count(pts, N, n_extra_poses)
    else:
        pos_samples = _sample_polyline_world_poses(pts, step_m, keep_last=True)

    return pos_samples


def rp_tp(pose): 
    """Teleport the robot to the i-th sampled pose. #TODO update docstring

    Args:
        inInts[0]: i in [0..N-1]

    Returns:
        outInts[0]: 1 if OK, 0 otherwise
        outFloats:  [x, y, yaw] actually set
    """
    x, y, z, yaw = pose
    sim.setObjectPosition(robotHandle, sim.handle_world, [x, y, 0.06969])
    sim.setObjectOrientation(robotHandle, sim.handle_world, [0.0, 0.0, yaw])

    return float(x), float(y), float(yaw)


def cmd_vel(linear, angular):
    global linealSpeed, angularSpeed
    global sim
    # Add noise to desired commands (optional)
    error_v = 0 
    error_w = 0 
    linear = linear + error_v    
    angularSpeed = angular + error_w

    if angular != 0:
        R = linear / angular
        vLeft = angular * (R - distance_between_wheels / 2)
        vRight = angular * (R + distance_between_wheels / 2)
    else:
        vLeft = linear
        vRight = linear
    
    # Set the speed to both motors
    sim.setJointTargetVelocity(motorLeft, vLeft / wheel_radius)
    sim.setJointTargetVelocity(motorRight, vRight / wheel_radius)
    linealSpeed = linear