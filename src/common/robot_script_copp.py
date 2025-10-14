import logging
import os
import shutil
import sys

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

# Robot variables
distance_between_wheels = None  # meters
wheel_radius = None            # meters 
linealSpeed = -1
angularSpeed = -1

# Handles
robotHandle = -1
footprintHandle = -1
robot_initial_pose_handle = -1
robotAlias = ""
motorLeft = -1
motorRight = -1
laserHandle = -1
curve1Handle = -1
curve2Handle = -1

# Curve handles:
graph_vel = -1
graph_lat = -1

# Color dictionary for path: each action has a different color
color_dict = [
    [1, 0, 0],        # Red
    [0.4, 0.0, 0.3],  # Purple
    [0, 1, 0],        # Green
    [0.6, 0.3, 0.0],  # Brown
    [0, 0, 1],        # Blue
    [1, 1, 0],        # Yellow
    [1, 0, 1],        # Magenta
    [0, 1, 1],        # Cyan
]

# Table to store different color paths
poseGtList = globals().get('poseGtList', {})
current_color_id = globals().get('current_color_id', 1)

# Other control variables
graphStartTime = 0
verbose = None


# -------------------------------
# ---------- UTILITIES ----------
# -------------------------------

# Reset the system and graph (this function will be called when linear and angular velocities are 0)
def resetGraphWithOffset():
    global sim
    # Reset the graph and start a new measurement for time
    sim.resetGraph(graph_vel)
    sim.resetGraph(graph_lat)


# Create streams and curves
def createCurves(graph, x1Label, y1Label, x1Units, y1Units, x2Label, y2Label, x2Units, y2Units, curve1Name, curve2Name):
    global graphStartTime, curve1Handle, curve2Handle
    global sim
    # Initialize graphStartTime
    graphStartTime = sim.getSimulationTime()
    
    # Create streams for first curve:
    curve1X = sim.addGraphStream(graph, x1Label, x1Units, 1)
    curve1Y = sim.addGraphStream(graph, y1Label, y1Units,1)
    
    # Create curve
    curve1Handle = sim.addGraphCurve(graph, curve1Name, 2, [curve1X, curve1Y], [0,0], y1Units, 0, [1,0,0])
    
    # Create streams for second curve:
    curve2X = sim.addGraphStream(graph, x2Label, x2Units, 1)
    curve2Y = sim.addGraphStream(graph, y2Label, y2Units, 1)
    
    # Create curve for right motor: x = time, y = velocity
    curve2Handle = sim.addGraphCurve(graph, curve2Name, 2, [curve2X, curve2Y], [0,0], y2Units, 0, [0,1,0])

    return curve1X, curve1Y, curve2X, curve2Y


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


def draw_path(linear, angular, color_id):
    global poseGtList, current_color_id
    global sim
    if verbose ==3:
        # Manage the path's drawing
        # If a reset is received, it clears the previous path (all colors) and finishes the function
        if linear == 0 and angular == 0:
            for poseGt in poseGtList.values():
                sim.addDrawingObjectItem(poseGt, None)
                sim.removeDrawingObject(poseGt)
            poseGtList = {}  # Clear the list
            current_color_id = 1  # Reset the color
            
            resetGraphWithOffset()
            return 
        
        # If there is movement, it draws it.
        # In case no color is received, it will choose the first one from the color dictionary
        color_id = color_id or 1 
        # Start a new cycle in the dictionary if it's capacity is exceeded
        color_id = (color_id - 1) % len(color_dict) + 1

        # If the drawing object for the current color doesn't exist, let's create it.
        if color_id not in poseGtList:
            color = color_dict[color_id - 1]
            poseGtList[color_id] = sim.addDrawingObject(sim.drawing_cyclic, 3, 0, -1, 5000, color, None, None, color)

        # Draw the current point in the trajectory
        p = sim.getObjectPosition(footprintHandle, -1)
        sim.addDrawingObjectItem(poseGtList[color_id], p)
        print('[draw_path] point:', p)

        current_color_id = color_id  # Update current color

        return


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
    global graph_vel, graph_lat
    global graphStartTime
    global subscriber_twist, publisher_odometry, publisher_ground_truth
    global latX, latY, distanceX, distanceY
    global leftMotorCurveX, leftMotorCurveY, rightMotorCurveX, rightMotorCurveY

    sim = require('sim')    # type: ignore

    # HANDLES
    robotHandle = sim.getObject('..')                   # the robot
    robotAlias = sim.getObjectAlias(robotHandle,3)      # robot name
    footprintHandle= sim.getObject('/base_link_visual')
    if robotAlias == 'Burger':
        laserHandle = sim.getObject('/Laser')
    elif robotAlias == 'Turtlebot2':
        laserHandle = sim.getObject('/fastHokuyo_ROS2')
    else:
        raise RuntimeError(f'[Turtle] Robot alias {robotAlias} not recognized. Available: Burger, Turtlebot2')
    
    motorLeft = sim.getObject("/wheel_left_joint")
    motorRight = sim.getObject("/wheel_right_joint")

    
    # INITIAL POSE
    # Create a dummy to set the initial robot pose (used on odometry estimation)
    robot_initial_pose_handle = sim.createDummy(0.1)
    sim.setObjectAlias(robot_initial_pose_handle, robotAlias + '_initial_pose')
    p = sim.getObjectPosition(footprintHandle,-1)
    o = sim.getObjectQuaternion(footprintHandle,-1)
    sim.setObjectPosition(robot_initial_pose_handle,-1,p)
    sim.setObjectQuaternion(robot_initial_pose_handle,-1,o)
    
    # ROS2 PUBLISHERS AND SUBSCRIBERS
    if simROS2_flag:
        try:
            subscriber_twist = simROS2.createSubscription(robotAlias + '/cmd_vel', 'geometry_msgs/msg/Twist', 'cmd_vel')    # type: ignore
            publisher_odometry = simROS2.createPublisher(robotAlias + '/odom', 'geometry_msgs/msg/Pose')                    # type: ignore      
            publisher_ground_truth = simROS2.createPublisher(robotAlias + '/ground_truth', 'geometry_msgs/msg/Pose')        # type: ignore
        except:
            raise RuntimeError('[Turtle] simROS2 not available. Unable to set Publishers and Subscribers')
    
    # GRAPHS
    if verbose == 3:
        # Instantiate graph objects
        graph_vel = sim.getObject("/Velocity_graph")
        graph_lat = sim.getObject("/LAT_vs_Distance_graph")
        
        # Initialize graphStartTime
        graphStartTime = sim.getSimulationTime()
        
        # Velocity graph
        leftMotorCurveX,leftMotorCurveY, rightMotorCurveX, rightMotorCurveY = createCurves(
            graph_vel, 
            'Time', 
            'Left Motor Velocity', 
            's', 
            'rad/s', 
            'Time', 
            'Right Motor Velocity', 
            's', 
            'rad/s', 
            'Left Motor', 
            'Right Motor'
        )
        
        # LAT graph
        latX, latY, distanceX, distanceY = createCurves(
            graph_lat, 
            'Time', 
            'LAT', 
            's', 
            's', 
            'Distance Time', 
            'Traveled distance', 
            's', 
            'm', 
            'LAT', 
            'Traveled distance'
        )


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
    # Update the graphs with current information
    global verbose, graphStartTime, motorLeft, motorRight
    global wheel_radius, graph_vel, graph_lat
    global leftMotorCurveX, leftMotorCurveY, rightMotorCurveX, rightMotorCurveY
    global latX, latY, distanceX, distanceY, linealSpeed
    global sim

    if verbose == 3:
        # Get the current simulation time
        currentTime = sim.getSimulationTime()
        relativeTime = currentTime - graphStartTime

        # Get wheel angular velocities
        omegaLeft = sim.getJointVelocity(motorLeft)
        vLeft = omegaLeft * wheel_radius

        omegaRight = sim.getJointVelocity(motorRight)
        vRight = omegaRight * wheel_radius

        # Update the graph with the linear and angular velocities
        sim.setGraphStreamValue(graph_vel, leftMotorCurveX, relativeTime)
        sim.setGraphStreamValue(graph_vel, leftMotorCurveY, omegaLeft)
        sim.setGraphStreamValue(graph_vel, rightMotorCurveX, relativeTime)
        sim.setGraphStreamValue(graph_vel, rightMotorCurveY, omegaRight)

        # Get traveled distance according to last simulation LAT and linear speed
        latValue = sim.getFloatSignal('latValueSignal')

        if latValue is not None and linealSpeed != -1:
            distValue = linealSpeed * latValue

            # Update LAT vs Distance graph
            sim.setGraphStreamValue(graph_lat, latX, relativeTime)
            sim.setGraphStreamValue(graph_lat, latY, latValue)
            sim.setGraphStreamValue(graph_lat, distanceX, relativeTime)
            sim.setGraphStreamValue(graph_lat, distanceY, distValue)


def sysCall_cleanup():
    # Do some clean-up here if needed
    pass