import random
import math
from typing import List, Tuple, Optional


outer_disk_rad = -1
distance_between_wheels = -1


# -------------------------------
# ---------- UTILITIES ----------
# -------------------------------

def _linspace_interior(length: float, n: int) -> List[float]:
    """Return n strictly interior equispaced positions in [0, length].

    The i-th position is: (i / (n + 1)) * length, for i=1..n.

    Args:
        length: Interval length (meters).
        n: Number of interior points (>= 0).

    Returns:
        List of positions in (0, length), excluding the endpoints.
    """
    if n <= 0:
        return []
    step = length / float(n + 1)
    return [i * step for i in range(1, n + 1)]


def _build_grid_equispaced(
    x_size: float,
    y_size: float,
    max_x: float,
    max_y: float,
    cols_per_part: int,
    rows_per_part: int,
    quads_x: int,
    quads_y: int,
    verbose: bool = False,
) -> List[Tuple[float, float]]:
    """Compute equispaced grid centers per floor partition (tablero).

    The floor spans [-x_size/2, +x_size/2] × [-y_size/2, +y_size/2] in world coords.
    We split it into quads_x × quads_y partitions of equal size. Inside each partition,
    we place cols_per_part × rows_per_part points at interior equispaced positions.

    Points are then filtered against ±max_x/±max_y (usable area).

    Args:
        x_size: Total floor size along X (meters).
        y_size: Total floor size along Y (meters).
        max_x: Usable half-extent for centers along X (meters).
        max_y: Usable half-extent for centers along Y (meters).
        cols_per_part: Number of interior columns per partition along X.
        rows_per_part: Number of interior rows per partition along Y.
        quads_x: Number of partitions along X.
        quads_y: Number of partitions along Y.
        verbose: If True, prints the generated positions.

    Returns:
        List of (x, y) world positions for obstacle centers.
    """
    part_w = x_size / float(quads_x)
    part_h = y_size / float(quads_y)

    x_min = -x_size / 2.0
    y_min = -y_size / 2.0

    # Precompute interior offsets (relative to each partition's lower-left corner)
    x_offsets = _linspace_interior(part_w, cols_per_part)
    y_offsets = _linspace_interior(part_h, rows_per_part)

    cells: List[Tuple[float, float]] = []
    for ix in range(quads_x):
        x_left = x_min + ix * part_w
        for iy in range(quads_y):
            y_bottom = y_min + iy * part_h
            # Cartesian product of interior offsets
            for dx in x_offsets:
                xk = x_left + dx
                if abs(xk) > max_x:
                    continue
                for dy in y_offsets:
                    ym = y_bottom + dy
                    if abs(ym) <= max_y:
                        cells.append((xk, ym))

    if verbose:
        print(f"[Grid] part_w={part_w:.3f}, part_h={part_h:.3f}, "
              f"cols={cols_per_part}, rows={rows_per_part}, "
              f"quads_x={quads_x}, quads_y={quads_y}")
        print("[Grid] Generated", len(cells), "positions:")
        for (x, y) in cells:
            print(f"   ({x:.2f}, {y:.2f})")

    return cells


def _clear_previous_and_make_collection(obstacles_handle: int) -> int:
    """
    Remove previously generated obstacles and create a new scene collection.

    This function deletes all existing obstacle objects that are children of the
    specified obstacle generator node in the CoppeliaSim scene, ensuring a clean
    environment before generating new obstacles. It then creates a fresh collection
    that includes all scene objects except the generator itself, which can later be
    used for distance calculations, collision checks, or environment grouping.

    Args:
        obstacles_handle (int): Handle of the obstacle generator node whose child 
            obstacles should be removed.

    Returns:
        int: Handle of the newly created collection containing all scene objects 
        except the generator and its subtree.
    """
    global sim
    collection_handle = sim.createCollection(0)

    # Remove previous obstacles (children of the generator)
    children = sim.getObjectsInTree(obstacles_handle, sim.handle_all, 1) or []
    if children:
        valid = [h for h in children if h is not None and h > 0]
        if valid:
            sim.removeObjects(valid)

    # Include all items (except the generator subtree itself)
    sim.addItemToCollection(collection_handle, sim.handle_all, -1, 1)
    return collection_handle


def _is_position_valid(
    collection_handle: int,
    r_x: float,
    r_y: float,
    diam_obstacles: float,
) -> bool:
    """Check clearance to all objects in collection using XY distance.

    Uses a special threshold 'Target_center' object,based on its bounding 
    box radius plus obstacle radius + a small margin.
    For any other object, uses a diameter-scaled margin.

    Args:
        collection_handle: Scene collection to check against.
        r_x: Candidate X (world).
        r_y: Candidate Y (world).
        diam_obstacles: Obstacle diameter (meters).

    Returns:
        True if the position is valid, False otherwise.
    """
    global sim
    global outer_disk_rad
    objs = sim.getCollectionObjects(collection_handle) or []
    for obj in objs:
        pos = sim.getObjectPosition(obj, sim.handle_world)  # [x, y, z]
        dx, dy = (r_x - pos[0]), (r_y - pos[1])
        d = math.hypot(dx, dy)

        name = sim.getObjectAlias(obj) or ""
        if "Outer_disk" in name:
            # Bounding box provides a characteristic radius
            min_x = sim.getObjectFloatParam(obj, 15)  # bbox min x
            max_x = sim.getObjectFloatParam(obj, 18)  # bbox max x
            radius = (max_x - min_x) / 2.0
            threshold = radius + (diam_obstacles / 2.0) + 0.02
            print("target th case")
        elif "Burger" in name:
            threshold = 0.033/2.0 + 0.18/2.0 + 0.01 + diam_obstacles/2.0 + 0.02 # TODO Fix, right now it's hardcoded
        elif "Turtlebot2" in name or "ctrlPt" in name:
            threshold = 0.035/2.0 + 0.13/2.0 + 0.01 + diam_obstacles/2.0 + 0.02
            if "ctrlPt" in name:
                print("path th case")
            else:
                print("Turtlebot2 th case")
        else:
            threshold = diam_obstacles + 0.18 + 0.02
        if d < threshold:
            print("Not valid position, obstacle is too close")
            return False
    print("Valid position")
    return True


def _place_obstacle(
    obstacles_handle: int,
    r_x: float,
    r_y: float,
    height_obstacles: float,
    diam_obstacles: float,
    idx: int,
    collection_handle: int,
) -> None:
    """
    Create and configure a single cylindrical obstacle in the CoppeliaSim environment.

    This function generates a primitive cylindrical shape representing an obstacle at the
    given (r_x, r_y) position, sets its physical and simulation properties, assigns it a
    unique alias (e.g., "Obstacle0"), and attaches it to both the obstacle parent handle
    and the provided collection for distance checks or collision detection.

    Args:
        obstacles_handle (int): Handle of the parent object or group to which the obstacle 
            will be attached in the scene hierarchy.
        r_x (float): X-coordinate of the obstacle's center position.
        r_y (float): Y-coordinate of the obstacle's center position.
        height_obstacles (float): Height of the cylindrical obstacle.
        diam_obstacles (float): Diameter of the cylindrical obstacle.
        idx (int): Index of the obstacle, used for naming (e.g., "Obstacle{idx}").
        collection_handle (int): Handle of the collection where the obstacle will be added 
            for subsequent distance or collision checks.

    Returns:
        None
    """
    global sim
    obs = sim.createPrimitiveShape(5, [diam_obstacles, diam_obstacles, height_obstacles])
    sim.setObjectPosition(obs, sim.handle_world, [r_x, r_y, height_obstacles / 2.0])
    sim.setObjectAlias(obs, f"Obstacle{idx}")
    sim.setObjectParent(obs, obstacles_handle, True)

    sim.setObjectSpecialProperty(
        obs,
        sim.objectspecialproperty_collidable
        | sim.objectspecialproperty_measurable
        | sim.objectspecialproperty_detectable
    )
    sim.setObjectInt32Param(obs, sim.shapeintparam_respondable, 1)
    sim.setObjectInt32Param(obs, sim.shapeintparam_static, 1)

    # Add obstacle to the collection for subsequent distance checks
    sim.addItemToCollection(collection_handle, sim.handle_single, obs, 0)
    
    # Uncomment when debugging
    # print(f"Obstacle{idx} placed correctly")


# -------------------------------
# ------- MAIN FUNCTIONS --------
# -------------------------------

def sysCall_init():
    """CoppeliaSim init: read config, compute bounds, prebuild grid if needed."""
    global sim
    global n_obstacles, diam_obstacles, height_obstacles
    global n_quads_x, n_quads_y, grid_rows_per_quad, grid_cols_per_quad, flag_grid, grid_visible
    global floor_xSize, floor_ySize
    global max_X, max_Y, obstacles, cells
    sim = require('sim')    # type: ignore
    
    # Read config from customization script 
    gen = sim.getObject("..") #The same as: gen = sim.getObjectHandle('/Burger/Obs_Generator')
    raw_gen = sim.readCustomBufferData(gen, '__config__')   
    cfg_gen   = sim.unpackTable(raw_gen) if raw_gen else {}
    
    # Get values and apply default ones if there is no config
    n_obstacles      = int(cfg_gen.get('n_obstacles', 8))
    diam_obstacles   = float(cfg_gen.get('diam_obstacles', 0.12))
    height_obstacles = float(cfg_gen.get('height_obstacles', 0.25))
    flag_grid = bool(cfg_gen.get('flag_grid', True))
    grid_visible = bool(cfg_gen.get('grid_visible', False))
    n_quads_x = int(cfg_gen.get('quads_x', 2))
    n_quads_y = int(cfg_gen.get('quads_y', 2))
    grid_rows_per_quad = int(cfg_gen.get('grid_rows_per_quad', 5))
    grid_cols_per_quad = int(cfg_gen.get('grid_cols_per_quad', 5))
    
    # Get wall config: it's needed for placing the obstacles within the container
    wall = sim.getObjectHandle('ExternalWall')
    raw_wall = sim.readCustomBufferData(wall, '__config__')
    cfg_wall   = sim.unpackTable(raw_wall) if raw_wall else {}
    floor_xSize    = cfg_wall.get('scene_x_dim', None)
    floor_ySize    = cfg_wall.get('scene_y_dim', None)
    
    # Usable area for positions: we substract obstacle radius as we cannot place obstacles above the walls
    max_X=floor_xSize/2-diam_obstacles/2.0
    max_Y=floor_ySize/2-diam_obstacles/2.0
    
    obstacles=sim.getObject(".")
    
    # Calculate avaliable cells when grid mode is active
    if flag_grid:
        cells = _build_grid_equispaced(
            floor_xSize, floor_ySize,
            max_X, max_Y,
            grid_cols_per_quad, grid_rows_per_quad,
            n_quads_x, n_quads_y, verbose=True
        )
        # Optional: draw grid points for debugging
        if grid_visible:
            h_dbg = sim.addDrawingObject(sim.drawing_points, 6.0, 0, -1, len(cells), [0.1, 0.7, 1.0])
            for (x, y) in cells:
                sim.addDrawingObjectItem(h_dbg, [x, y, 0.02])
    

def generate_obs(positions: Optional[List[Tuple[float, float]]]):
    """
    Generate obstacles either:
      - Grid mode (flag_grid=True): just grid positions are allowed.
      - Original mode (flag_grid=False): uniform random position within bounds.
    """
    global cells
    global sim

    collection_handle = _clear_previous_and_make_collection(obstacles)

    placed = 0
    to_place = n_obstacles
    
    if positions:
        for (r_x, r_y) in positions:
            _place_obstacle(obstacles, r_x, r_y, height_obstacles, diam_obstacles, placed+1, collection_handle)
            placed += 1 
    else:

        # Grid mode
        if flag_grid:
            random.shuffle(cells)
            to_place = min(n_obstacles, len(cells))

            for (r_x, r_y) in cells:
                if placed >= to_place:
                    break
                if _is_position_valid(collection_handle, r_x, r_y, diam_obstacles):
                    _place_obstacle(obstacles, r_x, r_y, height_obstacles, diam_obstacles, placed+1, collection_handle)
                    placed += 1

            if placed < to_place:
                print(f"[ObsGen] Grid mode: placed {placed}/{to_place} (cells near objects).")
                
        else:
            # Completely random mode (without grid)
            for i in range(to_place):
                done = False
                timeout = 50
                while not done and timeout > 0:
                    timeout -= 1
                    r_x = random.uniform(-max_X, max_X)
                    r_y = random.uniform(-max_Y, max_Y)

                    if _is_position_valid(collection_handle, r_x, r_y, diam_obstacles):
                        _place_obstacle(obstacles, r_x, r_y, height_obstacles, diam_obstacles, i+1, collection_handle)
                        done = True

    sim.destroyCollection(collection_handle)
    return obstacles