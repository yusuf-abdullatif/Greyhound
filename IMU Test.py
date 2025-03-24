import serial
import json
import numpy as np
import pygame
import math
from queue import PriorityQueue
import time

# =============================================
# Configuration
# =============================================
SERIAL_PORT = 'COM3'  # Update with your Arduino port
BAUD_RATE = 115200
GRID_SIZE = 10.0  # 10x10 meter environment
GRID_RESOLUTION = 0.1  # 10cm per cell
CELLS = int(GRID_SIZE / GRID_RESOLUTION)
TARGET = (7.0, 7.0)  # Mock target in meters

# Pygame display
WINDOW_SIZE = 800
CELL_PIXELS = WINDOW_SIZE // CELLS


# =============================================
# IMU Data Processing
# =============================================
class IMUProcessor:
    def __init__(self):
        self.x = GRID_SIZE / 2  # Start at center
        self.y = GRID_SIZE / 2
        self.yaw = 0.0
        self.alpha = 0.98  # Complementary filter constant
        self.last_time = time.time()

        # Calibration offsets (adjust based on your IMU)
        self.accel_offset = np.array([0, 0, 0])
        self.gyro_offset = np.array([0, 0, 0])

    def update(self, accel_raw, gyro_raw):
        # Convert raw data to physical units
        accel = (np.array(accel_raw) / 16384.0 - self.accel_offset) * 9.81  # m/s²
        gyro = (np.array(gyro_raw) / 131.0 - self.gyro_offset) * (math.pi / 180)  # rad/s

        dt = time.time() - self.last_time
        self.last_time = time.time()

        # Complementary filter for orientation
        pitch_acc = math.atan2(accel[0], math.sqrt(accel[1] ** 2 + accel[2] ** 2))
        self.yaw = self.alpha * (self.yaw + gyro[2] * dt) + (1 - self.alpha) * pitch_acc

        # Dead reckoning (simplified 2D)
        ax = accel[0] * math.cos(self.yaw) - accel[2] * math.sin(self.yaw)
        ay = accel[1]

        self.x += 0.5 * ax * dt ** 2
        self.y += 0.5 * ay * dt ** 2

        return self.x, self.y, math.degrees(self.yaw)


# =============================================
# Navigation and Mapping
# =============================================
class NavigationSystem:
    def __init__(self):
        self.grid = np.full((CELLS, CELLS), 0.5)  # 0.5 = unknown
        self.path = []

    def update_grid(self, x, y):
        gx, gy = self.world_to_grid(x, y)
        if 0 <= gx < CELLS and 0 <= gy < CELLS:
            self.grid[gx][gy] = 0.1  # Mark as free

    def world_to_grid(self, x, y):
        return (int(x / GRID_RESOLUTION), int(y / GRID_RESOLUTION))

    def astar(self, start, goal):
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

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < CELLS and 0 <= neighbor[1] < CELLS:
                    if self.grid[neighbor[0]][neighbor[1]] > 0.7:  # Avoid obstacles
                        continue
                    tentative_g = g_score[current] + 1
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + math.hypot(goal[0] - neighbor[0], goal[1] - neighbor[1])
                        open_set.put((f_score, neighbor))
        return None


# =============================================
# Visualization
# =============================================
class Visualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        self.font = pygame.font.SysFont(None, 24)

    def draw(self, nav, x, y, yaw):
        self.screen.fill((255, 255, 255))

        # Draw grid
        for i in range(CELLS):
            for j in range(CELLS):
                if nav.grid[i][j] > 0.7:
                    color = (0, 0, 0)  # Obstacle
                elif nav.grid[i][j] < 0.3:
                    color = (200, 200, 200)  # Free
                else:
                    color = (150, 150, 150)  # Unknown
                pygame.draw.rect(self.screen, color,
                                 (i * CELL_PIXELS, j * CELL_PIXELS, CELL_PIXELS, CELL_PIXELS))

        # Draw robot
        rx, ry = nav.world_to_grid(x, y)
        pygame.draw.circle(self.screen, (255, 0, 0),
                           (rx * CELL_PIXELS, ry * CELL_PIXELS), 10)

        # Draw heading arrow
        arrow_len = 30
        end_x = rx * CELL_PIXELS + arrow_len * math.cos(math.radians(yaw))
        end_y = ry * CELL_PIXELS + arrow_len * math.sin(math.radians(yaw))
        pygame.draw.line(self.screen, (0, 0, 255),
                         (rx * CELL_PIXELS, ry * CELL_PIXELS), (end_x, end_y), 3)

        # Draw path
        if nav.path:
            for p in nav.path:
                pygame.draw.circle(self.screen, (0, 255, 0),
                                   (p[0] * CELL_PIXELS, p[1] * CELL_PIXELS), 3)

        pygame.display.flip()


# =============================================
# Main Execution
# =============================================
if __name__ == "__main__":
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    imu = IMUProcessor()
    nav = NavigationSystem()
    viz = Visualizer()

    target_grid = nav.world_to_grid(TARGET[0], TARGET[1])

    try:
        while True:
            # Read IMU data
            line = ser.readline().decode().strip()
            if line:
                try:
                    data = json.loads(line)
                    x, y, yaw = imu.update(
                        [data['ax'], data['ay'], data['az']],
                        [data['gx'], data['gy'], data['gz']]
                    )

                    # Update mapping
                    nav.update_grid(x, y)

                    # Plan path every 2 seconds
                    if time.time() % 2 < 0.1:
                        current_grid = nav.world_to_grid(x, y)
                        nav.path = nav.astar(current_grid, target_grid)

                    # Visualize
                    viz.draw(nav, x, y, yaw)

                    # Print status
                    print(f"Position: ({x:.2f}, {y:.2f}) | Yaw: {yaw:.1f}°")

                except json.JSONDecodeError:
                    pass

            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        ser.close()
        pygame.quit()
        print("System shutdown")