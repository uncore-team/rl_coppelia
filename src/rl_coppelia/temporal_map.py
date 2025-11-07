"""
Trajectory generation and 1 cm sampling with onboard laser (4-sector minima)
and time-policy inference in CoppeliaSim (ZMQ Remote API).

Features:
  - Create or reuse a Path (square / "house" / existing).
  - Uniform 1 cm resampling (arc-length) -> poses [x,y,z,yaw] (yaw from path tangent).
  - Synchronous stepping: teleport robot base, step, read laser via Lua function
    sysCall_scriptFunction.getLaser4Sectors(), then run policy π(t|obs) and store results.

Outputs:
  - CSV: idx, s, x, y, z, yaw, t_argmax, t_expectation, obs_json(4), p_json
  - NPZ: poses (N,4), s (N,), obs_list (N,4), P (N,K), timesteps (K,)

Replace `policy_time_distribution()` with your trained model inference (e.g., PyTorch).
"""

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import math, json, csv, time
from typing import List

from common.rl_coppelia_manager import RLCoppeliaManager

# ------------- SCENE CONFIG (ADAPT THIS) -------------
BASE_PATH_NAME    = '/Turtlebot2'           # robot base to teleport
LASER_SCRIPT_OBJ  = '/Turtlebot2/fastHokuyo_ROS2'     # object that holds your laser Lua script
PATH_OBJECT_NAME  = 'RecordedPath'     # existing path name (GUI) if CREATE_TRAJ_MODE='existing'
CREATE_TRAJ_MODE  = 'existing'         # 'existing' | 'square' | 'house'
# -----------------------------------------------------

# Sampling & visualization
DS = 0.01            # 1 cm spacing
CONST_Z = 0.05       # z height if your path is 2D
USE_TANGENT_YAW = True
WARMUP_STEPS = 1     # extra sim.step() after teleport to refresh sensors

# Laser call
LUA_FUNC_4SECT = 'getLaser4Sectors'    # wrapper you added in Lua

# Time discretization (seconds) for your policy
TIMESTEPS = np.array([0.02, 0.05, 0.10, 0.20], dtype=float)


# ----------------- GEOMETRY HELPERS -----------------
def yaw_from_tangent(p0, p1):
    """Compute yaw (Z) from 2D tangent."""
    dx, dy = p1[0]-p0[0], p1[1]-p0[1]
    return math.atan2(dy, dx) if abs(dx)+abs(dy) > 1e-12 else 0.0

def cumulative_s(xy: np.ndarray) -> np.ndarray:
    """Cumulative arc-length of 2D polyline."""
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])

def uniform_resample(xy: np.ndarray, ds: float) -> np.ndarray:
    """Uniform arc-length resampling every ds meters."""
    s = cumulative_s(xy)
    L = s[-1]
    if L < 1e-9:
        return xy[:1].copy()
    s_new = np.arange(0.0, L + 1e-12, ds)
    out = np.zeros((len(s_new), 2))
    j = 0
    for i, sv in enumerate(s_new):
        while j+1 < len(s) and s[j+1] < sv:
            j += 1
        if j+1 == len(s):
            out[i] = xy[-1]
        else:
            t = (sv - s[j]) / max(1e-12, (s[j+1]-s[j]))
            out[i] = (1.0 - t) * xy[j] + t * xy[j+1]
    return out
# ----------------------------------------------------


# --------------- COPPELIA HELPERS -------------------
def connect_sim():
    """Connect with synchronous stepping enabled."""
    client = RemoteAPIClient()
    sim = client.require('sim')
    client.setStepping(True)
    return client, sim

def make_quat_from_yaw(yaw: float):
    """Return [qx,qy,qz,qw] for a Z-yaw."""
    cy = math.cos(0.5 * yaw)
    sy = math.sin(0.5 * yaw)
    return [0.0, 0.0, sy, cy]

def create_path_from_polyline(sim, name: str, points_xy, closed=True):
    """Create a Path from 2D polyline points."""
    pts = list(points_xy)
    if closed and (len(pts) < 2 or pts[0] != pts[-1]):
        pts = pts + [pts[0]]
    # simple yaw per point using local tangent
    ctrl = []
    for i in range(len(pts)):
        p0 = pts[max(0, i-1)]
        p1 = pts[min(len(pts)-1, i+1)]
        yaw = yaw_from_tangent(p0, p1)
        qx,qy,qz,qw = make_quat_from_yaw(yaw)
        ctrl += [pts[i][0], pts[i][1], CONST_Z, qx,qy,qz,qw]
    h = sim.createPath(ctrl, 8, max(2, len(pts)), 0.9, 0, [0,0,1])
    sim.setObjectName(h, name)
    return h

def get_or_create_path(sim, name: str):
    """Return an existing path or create one based on CREATE_TRAJ_MODE."""
    if CREATE_TRAJ_MODE == 'existing':
        return sim.getObject(f'/{name}')
    elif CREATE_TRAJ_MODE == 'square':
        square = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
        return create_path_from_polyline(sim, name, square, closed=True)
    elif CREATE_TRAJ_MODE == 'house':
        # Example "house" outline (feel free to edit to match your scene):
        house = [
            (0.0, 0.0), (2.0, 0.0), (2.0, 1.2), (1.2, 1.2),
            (1.2, 2.0), (0.0, 2.0), (0.0, 0.0)
        ]
        return create_path_from_polyline(sim, name, house, closed=False)
    else:
        raise ValueError('Unknown CREATE_TRAJ_MODE')

def sample_path_dense(sim, path_h, N=5000):
    """Sample many parametric points along t∈[0,1] (not arc-length uniform)."""
    ts = np.linspace(0.0, 1.0, N)
    xy = []
    for t in ts:
        x,y,_ = sim.getPositionOnPath(path_h, t)
        xy.append((x,y))
    return np.array(xy)

def build_resampled_poses(sim, path_h, ds):
    """Return poses (x,y,z,yaw) spaced every ds meters."""
    xy_dense = sample_path_dense(sim, path_h, N=5000)
    xy = uniform_resample(xy_dense, ds)
    yaws = []
    for i in range(len(xy)):
        p0 = xy[max(0, i-1)]
        p1 = xy[min(len(xy)-1, i+1)]
        yaws.append(yaw_from_tangent(p0, p1) if USE_TANGENT_YAW else 0.0)
    z = np.full((len(xy),), CONST_Z)
    return np.column_stack([xy, z, np.array(yaws)])

def teleport(sim, handle, pose):
    """Teleport base to [x,y,z,yaw] in world."""
    x,y,z,yaw = pose
    q = make_quat_from_yaw(yaw)
    sim.setObjectPose(handle, sim.handle_world, [x,y,z] + q)
# ----------------------------------------------------


# ---------------- LASER + POLICY --------------------
def read_laser_4sectors(sim, laser_script_handle) -> List[float]:
    """Call your Lua wrapper to get 4 minima (one per sector)."""
    # out = sim.callScriptFunction(LUA_FUNC_4SECT, laser_script_handle, [])
    # return list(out) if out is not None else []
    laser=sim.getObject('/Turtlebot2/fastHokuyo_ROS2')
    handle_laser_get_observation_script=sim.getScript(1,laser,'laser_get_observations')
    lasers_obs = sim.callScriptFunction(
        'laser_get_observations', handle_laser_get_observation_script
    )
    return lasers_obs

def softmax(x):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

def policy_time_distribution(obs4: List[float], timesteps: np.ndarray) -> np.ndarray:
    """Replace this with your trained policy (e.g., PyTorch model inference).

    Example heuristic: prefer larger t when ranges are large (freer space).
    """
    if not obs4:
        return np.full_like(timesteps, 1.0/len(timesteps), dtype=float)
    arr = np.asarray(obs4, dtype=float)
    m, v = float(np.mean(arr)), float(np.std(arr))
    logits = (m - 0.3*v) * (timesteps / (timesteps.max() + 1e-9))
    return softmax(logits)
# ----------------------------------------------------


def main(args):
    rl_copp = RLCoppeliaManager(args)
    rl_copp.start_coppelia_sim("Temporal-map")
    
    client, sim = connect_sim()

    base_h  = sim.getObject(BASE_PATH_NAME)
    laser_h = sim.getObject(LASER_SCRIPT_OBJ)
    path_h  = get_or_create_path(sim, PATH_OBJECT_NAME)

    poses = build_resampled_poses(sim, path_h, DS)
    s_vals = cumulative_s(poses[:, :2])
    N = len(poses)
    print(f'[Info] Trajectory poses: {N} (spacing {DS*100:.0f} mm)')

    # Start sim in synchronous mode
    sim.startSimulation()
    time.sleep(0.05)

    obs_list = []
    P = np.zeros((N, len(TIMESTEPS)), dtype=float)
    t_arg = np.zeros((N,), dtype=float)
    t_exp = np.zeros((N,), dtype=float)

    for i in range(N):
        teleport(sim, base_h, poses[i])
        for _ in range(max(1, WARMUP_STEPS)):
            sim.step()

        obs4 = read_laser_4sectors(sim, laser_h)   # [min_s1, min_s2, min_s3, min_s4]
        obs_list.append(obs4)

        p = policy_time_distribution(obs4, TIMESTEPS)  # π(t|obs)
        P[i, :] = p
        t_arg[i] = float(TIMESTEPS[int(np.argmax(p))])
        t_exp[i] = float(np.sum(TIMESTEPS * p))

        if (i+1) % 200 == 0 or i == N-1:
            print(f'  [{i+1}/{N}]')

    sim.stopSimulation()

    # Save CSV (compact JSON for vectors)
    with open('traj_time_dataset_4sectors.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx','s','x','y','z','yaw','t_argmax','t_expectation','obs4_json','p_json'])
        for i in range(N):
            w.writerow([
                i, f'{s_vals[i]:.4f}',
                f'{poses[i,0]:.6f}', f'{poses[i,1]:.6f}', f'{poses[i,2]:.6f}', f'{poses[i,3]:.6f}',
                f'{t_arg[i]:.4f}', f'{t_exp[i]:.4f}',
                json.dumps(obs_list[i]), json.dumps(P[i,:].tolist())
            ])

    np.savez('traj_time_dataset_4sectors.npz',
             poses=poses, s=s_vals,
             obs_list=np.array(obs_list, dtype=object),
             P=P, timesteps=TIMESTEPS)

    # Visual helper: a path showing the resampled poses
    ctrl = []
    for i in range(N):
        x,y,z,yaw = poses[i]
        q = make_quat_from_yaw(yaw)
        ctrl += [x,y,z] + q
    vis_h = sim.createPath(ctrl, 8, max(2, N//10), 0.9, 0, [0,0,1])
    sim.setObjectName(vis_h, f'ResampledPath_{int(DS*100)}mm')
    print('[Save] CSV/NPZ done. Visual path created.')

if __name__ == '__main__':
    main()