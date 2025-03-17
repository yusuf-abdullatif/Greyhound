import numpy as np
import pygame
from queue import PriorityQueue

# =============================================
# Simulation Constants
# =============================================
GRID_RESOLUTION = 0.5  # meters per cell
GRID_WIDTH = 20  # cells
GRID_HEIGHT = 20  # cells
CELL_SIZE = 40  # pixels per cell
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE

# =============================================
# Color Scheme
# =============================================
COLORS = {
    'background': (255, 255, 255),
    'unknown': (200, 200, 200),
    'free': (255, 255, 255),
    'occupied': (0, 0, 0),
    'robot': (255, 0, 0),
    'goal': (0, 255, 0),
    'path': (0, 0, 255)
}


# =============================================
# Robot Simulation Components (Same as Before)
# =============================================
class RobotSimulator:
    def __init__(self, start_pos=(0, 0)):
        self.position = np.array(start_pos, dtype=float)
        self.speed = 0.8
        self.obstacles = [
            (2.5, 2.5), (2.5, 3.0), (3.0, 2.5), (3.0, 3.0),
            (1.0, 4.0), (1.5, 4.0), (2.0, 4.0)
        ]

    def move(self, dx, dy):
        step = np.array([dx, dy]) * self.speed
        self.position += step

    def get_sensor_data(self):
        detected = []
        for (ox, oy) in self.obstacles:
            distance = np.linalg.norm(self.position - [ox, oy])
            if distance < 2.0:
                detected.append((ox, oy))
        return detected


class OccupancyGrid:
    def __init__(self):
        self.grid = np.full((GRID_WIDTH, GRID_HEIGHT), 0.5)

    def update_grid(self, robot_pos_world, sensor_data_world):
        x_grid = int(robot_pos_world[0] / GRID_RESOLUTION)
        y_grid = int(robot_pos_world[1] / GRID_RESOLUTION)
        if 0 <= x_grid < GRID_WIDTH and 0 <= y_grid < GRID_HEIGHT:
            self.grid[x_grid, y_grid] = 0.1
        for (ox, oy) in sensor_data_world:
            x_grid = int(ox / GRID_RESOLUTION)
            y_grid = int(oy / GRID_RESOLUTION)
            if 0 <= x_grid < GRID_WIDTH and 0 <= y_grid < GRID_HEIGHT:
                self.grid[x_grid, y_grid] = 0.9


def astar(grid, start, goal):
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
            return path[::-1]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < GRID_WIDTH and 0 <= neighbor[1] < GRID_HEIGHT:
                if grid[neighbor[0], neighbor[1]] > 0.7:
                    continue
                tentative_g = g_score[current] + np.sqrt(dx ** 2 + dy ** 2)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + np.linalg.norm(np.array(goal) - np.array(neighbor))
                    open_set.put((f_score, neighbor))
    return None


# =============================================
# Pygame Visualization Functions
# =============================================
def world_to_grid(pos):
    return (int(pos[0] / GRID_RESOLUTION), int(pos[1] / GRID_RESOLUTION))


def grid_to_pixel(grid_pos):
    return (grid_pos[0] * CELL_SIZE + CELL_SIZE // 2,
            grid_pos[1] * CELL_SIZE + CELL_SIZE // 2)


def draw_grid(screen, grid):
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if grid[x, y] > 0.7:
                color = COLORS['occupied']
            elif grid[x, y] < 0.3:
                color = COLORS['free']
            else:
                color = COLORS['unknown']
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (150, 150, 150), rect, 1)  # Grid lines


def draw_robot(screen, position):
    grid_pos = world_to_grid(position)
    pixel_pos = grid_to_pixel(grid_pos)
    pygame.draw.circle(screen, COLORS['robot'], pixel_pos, CELL_SIZE // 3)


def draw_goal(screen, goal_world):
    grid_pos = world_to_grid(goal_world)
    pixel_pos = grid_to_pixel(grid_pos)
    pygame.draw.circle(screen, COLORS['goal'], pixel_pos, CELL_SIZE // 2, 3)


def draw_path(screen, path):
    if path and len(path) >= 2:  # Only draw if path has 2+ points
        pixel_points = [grid_to_pixel(p) for p in path]
        pygame.draw.lines(screen, COLORS['path'], False, pixel_points, 3)

# =============================================
# Main Simulation Loop
# =============================================
def main():
    # Initialize components
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Robot Navigation Simulation")
    clock = pygame.time.Clock()

    robot = RobotSimulator(start_pos=(0.0, 0.0))
    grid = OccupancyGrid()
    goal_world = (7.5, 7.5)

    running = True
    for step in range(1000):
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not running:
            break

        # Check if robot reached goal
        if np.linalg.norm(robot.position - np.array(goal_world)) < 0.5:
            print("Goal reached!")
            break

        # Simulation step
        sensor_data = robot.get_sensor_data()
        grid.update_grid(robot.position, sensor_data)

        start_grid = world_to_grid(robot.position)
        goal_grid = world_to_grid(goal_world)
        path = astar(grid.grid, start_grid, goal_grid)

        if path and len(path) > 1:
            next_step_grid = path[1]
            next_step_world = (
                next_step_grid[0] * GRID_RESOLUTION,
                next_step_grid[1] * GRID_RESOLUTION
            )
            dx = next_step_world[0] - robot.position[0]
            dy = next_step_world[1] - robot.position[1]
            robot.move(dx, dy)

        # Draw everything
        screen.fill(COLORS['background'])
        draw_grid(screen, grid.grid)
        draw_path(screen, path)
        draw_goal(screen, goal_world)
        draw_robot(screen, robot.position)

        pygame.display.flip()
        clock.tick(30)  # 10 FPS

    pygame.quit()


if __name__ == "__main__":
    main()