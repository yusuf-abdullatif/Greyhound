import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue


# =============================================
# Mock Robot Simulator (Fixed Movement)
# =============================================
class RobotSimulator:
    def __init__(self, start_pos=(0, 0)):
        self.position = np.array(start_pos, dtype=float)  # (x, y)
        self.speed = 1.0  # Increased speed for visibility
        self.obstacles = [
            (5, 5), (5, 6), (6, 5), (6, 6),  # Central obstacle
            (2, 8), (3, 8), (4, 8)  # Wall
        ]

    def move(self, dx, dy):
        """Simulate movement with noise."""
        noise = np.random.normal(0, 0.05, 2)  # Reduced noise
        step = np.array([dx, dy]) * self.speed + noise
        self.position += step  # Fixed parentheses

    def get_sensor_data(self):
        """Simulate detecting obstacles near the robot."""
        detected = []
        for (ox, oy) in self.obstacles:
            distance = np.linalg.norm(self.position - [ox, oy])
            if distance < 3.0:  # Sensor range = 3 meters
                detected.append((ox, oy))
        return detected


# =============================================
# Occupancy Grid Mapping (Unchanged)
# =============================================
class OccupancyGrid:
    def __init__(self, width=20, height=20, resolution=0.5):
        self.grid = np.full((width, height), 0.5)
        self.resolution = resolution

    def update_grid(self, robot_pos, sensor_data):
        x = int(robot_pos[0] / self.resolution)
        y = int(robot_pos[1] / self.resolution)
        if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]:
            self.grid[x, y] = 0.1  # Free
        for (ox, oy) in sensor_data:
            x = int(ox / self.resolution)
            y = int(oy / self.resolution)
            if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]:
                self.grid[x, y] = 0.9  # Occupied


# =============================================
# A* Algorithm (Optimized)
# =============================================
def astar(grid, start, goal):
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}

    while not open_set.empty():
        current = open_set.get()[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0], neighbor[1]] > 0.7:  # Avoid obstacles
                    continue
                tentative_g = g_score[current] + np.sqrt(dx ** 2 + dy ** 2)  # Diagonal cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + np.linalg.norm(np.array(goal) - np.array(neighbor))
                    open_set.put((f_score, neighbor))
    return None


# =============================================
# Simulation Loop (Fixed Scaling)
# =============================================
if __name__ == "__main__":
    robot = RobotSimulator(start_pos=(0, 0))
    grid = OccupancyGrid(width=20, height=20, resolution=0.5)
    goal = (15, 15)  # Goal in grid coordinates

    plt.ion()
    fig, ax = plt.subplots()

    for step in range(100):
        sensor_data = robot.get_sensor_data()
        grid.update_grid(robot.position, sensor_data)

        # Convert robot position to grid coordinates
        start_grid = (
            int(robot.position[0] / grid.resolution),
            int(robot.position[1] / grid.resolution)
        )
        goal_grid = (
            int(goal[0] / grid.resolution),
            int(goal[1] / grid.resolution)
        )

        path = astar(grid.grid, start_grid, goal_grid)

        if path and len(path) > 1:
            next_step = path[0]  # Immediate next step
            target = (
                next_step[0] * grid.resolution,
                next_step[1] * grid.resolution
            )
            dx = target[0] - robot.position[0]
            dy = target[1] - robot.position[1]
            robot.move(dx, dy)  # Removed damping

        # Visualization
        ax.clear()
        ax.imshow(grid.grid.T, origin='lower', cmap='gray_r', vmin=0, vmax=1)
        ax.scatter(robot.position[0] / grid.resolution, robot.position[1] / grid.resolution,
                   c='red', s=100, label='Robot')
        ax.scatter(goal_grid[0], goal_grid[1], c='green', marker='*', s=200, label='Goal')
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax.plot(path_x, path_y, c='blue', linestyle='--', label='Path')
        ax.legend()
        plt.pause(0.5)  # Slowed down for visibility

    plt.ioff()
    plt.show()