"""
Game constants and configuration parameters for Brick Blast.
"""

# Grid dimensions
GRID_COLS = 8
GRID_ROWS = 10

# Starting player state
INITIAL_BALLS = 60

# Screen and layout dimensions
WIDTH = 480
HEIGHT = 800
TOP_MARGIN = 140
BOTTOM_MARGIN = 100
CELL_WIDTH = WIDTH // GRID_COLS           # 60px per column
CELL_HEIGHT = (HEIGHT - TOP_MARGIN - BOTTOM_MARGIN) // GRID_ROWS  # 56px per row
BRICK_WIDTH = CELL_WIDTH - 2              # 58px wide brick (1px margin on each side)
BRICK_HEIGHT = CELL_HEIGHT - 2            # 54px high brick

# Physics constants
BALL_RADIUS = 7
BALL_SPEED = 12.0
LAUNCH_INTERVAL = 3  # frames between ball launches in human play

# Colors (RGB)
COLOR_BG = (26, 38, 78)            # Deep navy blue matching screenshot
COLOR_TEXT_WHITE = (255, 255, 255)
COLOR_BALL = (255, 215, 0)         # Golden yellow
COLOR_AIM_LINE = (220, 230, 255)
COLOR_BORDER = (40, 56, 110)
COLOR_GROUND = (20, 30, 60)

# Brick health color tiers
COLOR_GREEN = (77, 192, 77)        # Low HP (e.g. <= 80)
COLOR_BLUE = (40, 160, 225)        # Mid HP (e.g. 81 - 120)
COLOR_YELLOW = (240, 192, 32)      # High HP (e.g. 121 - 180)
COLOR_RED = (225, 64, 64)          # Very high HP (e.g. 181 - 250)
COLOR_PURPLE = (175, 48, 215)      # Boss/Extreme HP (> 250)

# Powerup colors
COLOR_REDIRECT = (46, 204, 113)    # Emerald green
COLOR_MULTIPLIER = (241, 196, 15)  # Gold/amber
COLOR_LASER = (231, 76, 60)        # Laser crimson

# Framerate and simulation
FPS = 60
