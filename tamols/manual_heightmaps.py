import numpy as np

GRID_SIZE = 12

def get_flat_heightmap():
    return np.zeros((GRID_SIZE, GRID_SIZE))


def get_platform_heightmap():
    grid_size = GRID_SIZE  # 50 x 0.04 = 2 meters
    elevation_map = np.zeros((grid_size, grid_size))

    platform_size_x = 10
    platform_size_y = 5
    platform_height = 0.05

    start_x = 5 + (grid_size - platform_size_x) // 2
    end_x = start_x + platform_size_x
    start_y = -5 + (grid_size - platform_size_y) // 2
    end_y = start_y + platform_size_y

    # Add the raised square platform
    elevation_map[start_x:end_x, start_y:end_y] = platform_height

    return elevation_map  


def get_random_rough_heightmap():
    grid_size = GRID_SIZE
    elevation_map = np.random.rand(grid_size, grid_size) * 0.05

    # Smooth the heightmap by averaging with neighbors
    for _ in range(1):  # Number of smoothing iterations
        elevation_map = (np.roll(elevation_map, 1, axis=0) + np.roll(elevation_map, -1, axis=0) +
                         np.roll(elevation_map, 1, axis=1) + np.roll(elevation_map, -1, axis=1) +
                         elevation_map) / 5.0
    return elevation_map


def get_heightmap_with_holes():
    grid_size = GRID_SIZE
    elevation_map = np.random.rand(grid_size, grid_size) * 0.05
    drop_height = 0.75

    # Smooth the heightmap by averaging with neighbors
    for _ in range(1):  # Number of smoothing iterations
        elevation_map = (np.roll(elevation_map, 1, axis=0) + np.roll(elevation_map, -1, axis=0) +
                         np.roll(elevation_map, 1, axis=1) + np.roll(elevation_map, -1, axis=1) +
                         elevation_map) / 5.0

    # Add random holes
    num_holes = np.random.randint(5, 15)  # Random number of holes
    for _ in range(num_holes):
        hole_x = np.random.randint(0, grid_size)
        hole_y = np.random.randint(0, grid_size)
        elevation_map[hole_x, hole_y] = -drop_height

    return elevation_map
