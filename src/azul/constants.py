import numpy as np

# --- 1. TILE ENCODING ---
# Using integers for Neural Network compatibility
EMPTY = 0
BLUE = 1
YELLOW = 2
RED = 3
BLACK = 4
WHITE = 5
FIRST_PLAYER_TOKEN = 6 

ID_TO_COLOR = {
    EMPTY: ".",
    BLUE: "B",
    YELLOW: "Y",
    RED: "R",
    BLACK: "K", 
    WHITE: "W",
    FIRST_PLAYER_TOKEN: "S"
}

PLAYABLE_COLORS = [BLUE, YELLOW, RED, BLACK, WHITE]

# --- 2. GAME CONFIGURATION ---
TILES_PER_COLOR = 20
TILES_PER_FACTORY = 4
FACTORY_COUNTS = {2: 5, 3: 7, 4: 9}

# --- 3. BOARD STRUCTURE ---
GRID_SIZE = 5
FLOOR_LINE_CAPACITY = 7
# The penalty points for the floor line (1st slot = -1, 2nd = -1, etc.)
FLOOR_LINE_SCORES = np.array([-1, -1, -2, -2, -2, -3, -3], dtype=np.int8)

# --- 4. THE WALL PATTERN ---
# Pre-calculated matrix of what color goes where on the wall
WALL_PATTERN = np.array([
    [BLUE,   YELLOW, RED,    BLACK,  WHITE],
    [WHITE,  BLUE,   YELLOW, RED,    BLACK],
    [BLACK,  WHITE,  BLUE,   YELLOW, RED],
    [RED,    BLACK,  WHITE,  BLUE,   YELLOW],
    [YELLOW, RED,    BLACK,  WHITE,  BLUE]
], dtype=np.int8)