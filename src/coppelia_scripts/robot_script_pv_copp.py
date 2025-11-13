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

MAX_SAMPLES = 1000
MIN_SAMPLES = 10


# ----------------------------------------------------------------------
# ----------------------------Path  helpers ----------------------------
# ----------------------------------------------------------------------

def augment_base_poses(
    base_poses,
    n_extra_poses: int = 0,
    delta_deg: float = 5.0,
    default_z: float = 0.0,
    default_yaw: float = 0.0
):
    """
    Augment base poses with additional yaw variants.

    Accepts base_poses entries with shape:
      - (x, y, z, yaw)
      - (x, y, z)
      - (x, y)

    If an entry has only (x,y) or (x,y,z) the missing z/yaw are filled with
    'default_z' and 'default_yaw' respectively. Yaw is normalized to [-pi,pi].

    Returns list of (x, y, z, yaw).
    """
    n_extra = max(0, int(n_extra_poses))
    delta = math.radians(float(delta_deg))
    augmented_poses = []
    default_yaw_rads = math.radians(float(default_yaw))

    for p in base_poses:
        # Support tuples/lists of length 2,3,4
        if isinstance(p, (list, tuple)):
            if len(p) == 4:
                x, y, z, yaw = p
            elif len(p) == 3:
                x, y, z = p
                yaw = default_yaw_rads
            elif len(p) == 2:
                x, y = p
                z = default_z
                yaw = default_yaw_rads
            else:
                raise ValueError(f"Base pose must be (x,y), (x,y,z) or (x,y,z,yaw); got length {len(p)}")
        else:
            raise TypeError(f"Base pose must be sequence, got {type(p)}")

        # Ensure numeric types and normalize yaw
        try:
            x = float(x); y = float(y); z = float(z); yaw = float(yaw)
        except Exception:
            raise TypeError(f"Pose contains non-numeric values: {p}")

        yaw = normalize_angle(yaw)
        augmented_poses.append((x, y, z, yaw))  # base
        for k in range(1, n_extra + 1):
            augmented_poses.append((x, y, z, normalize_angle(yaw + k * delta)))
        for k in range(1, n_extra + 1):
            augmented_poses.append((x, y, z, normalize_angle(yaw - k * delta)))

    return augmented_poses


def normalize_angle(a):
        return math.atan2(math.sin(a), math.cos(a))


def build_world_poses_from_path_data(
    path_handle,
    n_samples: int,
    n_extra_poses: int = 0,
    delta_deg: float = 5.0
):
    """Return (x, y, z, yaw) poses uniformly sub-sampled by index from a CoppeliaSim Path.

    This reduced version:
      - Works directly on the Path's local-frame flattened arrays:
          * path_positions_flat: [x,y,z, x,y,z, ...]
          * path_quaternions_flat: [qx,qy,qz,qw, qx,qy,qz,qw, ...]
      - Picks exactly `n_samples` evenly spaced indices (uniform by index).
      - Computes yaw (Z rotation) directly from the quaternion at each sample.
      - Augments each base pose with ±k*delta_deg yaw-only variants.

    Args:
        path_positions_flat (list[float]): Flattened [x,y,z,...] in Path LOCAL frame.
        path_quaternions_flat (list[float]): Flattened [qx,qy,qz,qw,...] in Path LOCAL frame.
        n_samples (int): Desired number of samples (must be <= total poses).
        n_extra_poses (int, optional): Extra yaw variants on each side (+/-k*delta).
            If N>0, adds +k*delta and -k*delta for k=1..N. Defaults to 0.
        delta_deg (float, optional): Angle step (degrees) for yaw augmentation. Defaults to 10.0.

    Returns:
        list[tuple[float, float, float, float]]: List of (x, y, z, yaw) in Path LOCAL frame.

    Notes:
        - This function assumes n_samples <= number_of_path_poses. If n_samples > M,
          it will clamp to M and print a warning.
        - Yaw is extracted from quaternion using a standard ZYX convention:
            yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)).
        - Yaw is normalized to [-pi, pi].
    """

    # ------------- Helpers ---------------
    def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
        """Extract yaw (Z) from quaternion (ZYX convention)."""
        s = 2.0 * (qw * qz + qx * qy)
        c = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(s, c)


    # --- Get path positions and quaternions
    pathData = sim.unpackDoubleTable(sim.getBufferProperty(path_handle, 'customData.PATH'))

    m = np.array(pathData).reshape(len(pathData) // 7, 7)
    path_positions_flat = m[:, :3].flatten().tolist()
    path_quaternions_flat = m[:, 3:].flatten().tolist()

    # -------------------------- Parse & sanity checks --------------------------
    n_pos = len(path_positions_flat) // 3
    n_quat = len(path_quaternions_flat) // 4
    if n_pos != n_quat or n_pos == 0:
        raise ValueError(f"Inconsistent path arrays: {n_pos} positions vs {n_quat} quaternions")
    n_original_samples = n_pos
    indices = []

    # --- Compute uniform-by-index sample indices
    if n_samples != n_original_samples:
        if n_samples > n_original_samples:
            print(f"Resampling the path will not be neccessary as it already has {n_original_samples}")
        else:
            print(f"N original samples: {n_original_samples}, will be resampled to {n_samples}")
     
        # Evenly spread integers in [0..M-1]
        
        for k in range(n_samples):
            # Round to nearest valid index
            idx = int(round(k * (n_original_samples - 1) / (n_samples - 1)))
            if not indices or idx != indices[-1]:
                indices.append(idx)
        # Ensure exact count
        while len(indices) < n_samples and indices[-1] < n_original_samples - 1:
            indices.append(indices[-1] + 1)
    
    else:
        for k in range(n_original_samples):
            indices.append(k)

    # --- Build base (x,y,z,yaw) samples 
    base_poses = []

    for i in indices:
        x = float(path_positions_flat[3 * i + 0])
        y = float(path_positions_flat[3 * i + 1])
        z = float(path_positions_flat[3 * i + 2])

        qx = float(path_quaternions_flat[4 * i + 0])
        qy = float(path_quaternions_flat[4 * i + 1])
        qz = float(path_quaternions_flat[4 * i + 2])
        qw = float(path_quaternions_flat[4 * i + 3])

        yaw = normalize_angle(_yaw_from_quat(qx, qy, qz, qw))
        base_poses.append((x, y, z, yaw))

    # --- Yaw augmentation for extra cases
    augmented_poses = augment_base_poses(base_poses, n_extra_poses, delta_deg)

    return augmented_poses, base_poses


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


def rp_init(n_samples, n_extra_poses, path_name):
    """Initialize path sampling.    # TODO: UPdate docstring

    Args:
        inStrings[0]: path alias (e.g., "/RecordedPath")
        inInts[0]   : OPTIONAL number of samples (N). If >0, overrides step_m.
        inFloats[0] : OPTIONAL step_m (meters) used only if N is not provided.

    Returns:
        outInts[0]: number of sampled poses
    """
    if n_samples < MIN_SAMPLES:
        n_samples = 10
    elif n_samples > MAX_SAMPLES:
        n_samples = 1000
    print(f"Trying to sample the path using a path alias: {path_name}, with {n_samples} samples.")
    path_alias = path_name if path_name else "/RecordedPath"
    path_handle = sim.getObject(path_alias)
    augmented_pos_samples, base_pos_samples = build_world_poses_from_path_data(path_handle, n_samples, n_extra_poses)

    return augmented_pos_samples, base_pos_samples


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