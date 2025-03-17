import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue


# =============================================
# Mock Robot Simulator (Corrected Scaling)
# =============================================
class RobotSimulator:
    def __init__(self, start_pos=(0, 0)):
        self.position = np.array(start_pos, dtype=float)  # World coordinates (meters)
        self.speed = 1.0
        self.obstacles = [  # Obstacles in world coordinates (meters)
            (2.5, 2.5), (2.5, 3.0), (3.0, 2.5), (3.0, 3.0),  # Central obstacle
            (1.0, 4.0), (1.5, 4.0), (2.0, 4.0)  # Wall
        ]

    def move(self, dx, dy):
        """Move robot toward target (dx, dy)."""
        step = np.array([dx, dy]) * self.speed
        self.position += step

    def get_sensor_data(self):
        """Detect nearby obstacles in world coordinates."""
        detected = []
        for (ox, oy) in self.obstacles:
            distance = np.linalg.norm(self.position - [ox, oy])
            if distance < 2.0:  # Sensor range = 2 meters
                detected.append((ox, oy))
        return detected


# =============================================
# Occupancy Grid (Fixed Coordinate Conversion)
# =============================================
class OccupancyGrid:
    def __init__(self, width=20, height=20, resolution=0.5):
        self.grid = np.full((width, height), 0.5)  # 0.5 = unknown
        self.resolution = resolution  # meters per cell

    def update_grid(self, robot_pos_world, sensor_data_world):
        """Update grid using world coordinates."""
        # Convert robot position to grid coordinates
        x_grid = int(robot_pos_world[0] / self.resolution)
        y_grid = int(robot_pos_world[1] / self.resolution)
        if 0 <= x_grid < self.grid.shape[0] and 0 <= y_grid < self.grid.shape[1]:
            self.grid[x_grid, y_grid] = 0.1  # Free

        # Mark obstacles
        for (ox, oy) in sensor_data_world:
            x_grid = int(ox / self.resolution)
            y_grid = int(oy / self.resolution)
            if 0 <= x_grid < self.grid.shape[0] and 0 <= y_grid < self.grid.shape[1]:
                self.grid[x_grid, y_grid] = 0.9  # Occupied


# =============================================
# A* Algorithm (Corrected Path Indexing)
# =============================================
def astar(grid, start, goal):
    """A* for grid coordinates."""
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}

    while not open_set.empty():
        current = open_set.get()[1]
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]  # Reverse to get start->goal

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0], neighbor[1]] > 0.7:
                    continue  # Skip obstacles
                tentative_g = g_score[current] + np.sqrt(dx ** 2 + dy ** 2)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + np.linalg.norm(np.array(goal) - np.array(neighbor))
                    open_set.put((f_score, neighbor))
    return None


# =============================================
# Simulation Loop (Fixed Goal and Path Indexing)
# =============================================
if __name__ == "__main__":
    robot = RobotSimulator(start_pos=(0.0, 0.0))
    grid = OccupancyGrid(width=20, height=20, resolution=0.5)
    goal_world = (7.5, 7.5)  # Goal in world coordinates (meters)

    plt.ion()
    fig, ax = plt.subplots()

    for step in range(50):
        # Update grid with sensor data
        sensor_data = robot.get_sensor_data()
        grid.update_grid(robot.position, sensor_data)

        # Convert positions to grid coordinates
        start_grid = (
            int(robot.position[0] / grid.resolution),
            int(robot.position[1] / grid.resolution)
        )
        goal_grid = (
            int(goal_world[0] / grid.resolution),
            int(goal_world[1] / grid.resolution)
        )

        # Plan path
        path = astar(grid.grid, start_grid, goal_grid)

        # Move robot
        if path and len(path) > 1:
            next_step_grid = path[1]  # First step after current position
            next_step_world = (
                next_step_grid[0] * grid.resolution,
                next_step_grid[1] * grid.resolution
            )
            dx = next_step_world[0] - robot.position[0]
            dy = next_step_world[1] - robot.position[1]
            robot.move(dx, dy)

        # Visualization
        ax.clear()
        ax.imshow(grid.grid.T, origin='lower', cmap='gray_r', vmin=0, vmax=1)
        ax.scatter(
            robot.position[0] / grid.resolution,
            robot.position[1] / grid.resolution,
            c='red', s=100, label='Robot'
        )
        ax.scatter(
            goal_grid[0], goal_grid[1],
            c='green', marker='*', s=200, label='Goal'
        )
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax.plot(path_x, path_y, c='blue', linestyle='--', label='Path')
        ax.legend()
        plt.pause(0.5)

    plt.ioff()
    plt.show()