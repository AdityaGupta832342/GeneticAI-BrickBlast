"""
Brick class representing numbered blocks on the 8x10 grid.
"""
from brickblast.constants import (
    CELL_WIDTH,
    CELL_HEIGHT,
    BRICK_WIDTH,
    BRICK_HEIGHT,
    TOP_MARGIN,
    COLOR_GREEN,
    COLOR_BLUE,
    COLOR_YELLOW,
    COLOR_RED,
    COLOR_PURPLE,
    COLOR_TEXT_WHITE,
)
from brickblast.pygame_compat import pygame, Rect


class Brick:
    def __init__(self, col, row, hp):
        self.col = int(col)
        self.row = int(row)
        self.hp = int(hp)
        self.max_hp = int(hp)
        self.update_rect()

    def update_rect(self):
        x = self.col * CELL_WIDTH + 1
        y = TOP_MARGIN + self.row * CELL_HEIGHT + 1
        self.rect = Rect(x, y, BRICK_WIDTH, BRICK_HEIGHT)

    def get_color(self):
        if self.hp <= 50:
            return COLOR_GREEN
        elif self.hp <= 100:
            return COLOR_BLUE
        elif self.hp <= 150:
            return COLOR_YELLOW
        elif self.hp <= 250:
            return COLOR_RED
        else:
            return COLOR_PURPLE

    def hit(self, damage=1):
        """
        Apply damage to brick. Returns True if destroyed.
        """
        self.hp -= damage
        return self.hp <= 0

    def move_down(self):
        """
        Shift down by one row at end of turn.
        """
        self.row += 1
        self.update_rect()

    def draw(self, surface, font):
        color = self.get_color()
        r = self.rect

        # 1. Outer drop shadow
        shadow_rect = Rect(r.x + 1, r.y + 3, r.w, r.h)
        pygame.draw.rect(surface, (15, 22, 45), shadow_rect, border_radius=8)

        # 2. Main brick block
        pygame.draw.rect(surface, color, r, border_radius=8)

        # 3. Top-half glossy sheen (acrylic 3D effect)
        sheen_color = (
            min(255, color[0] + 30),
            min(255, color[1] + 30),
            min(255, color[2] + 30),
        )
        sheen_rect = Rect(r.x + 3, r.y + 3, r.w - 6, (r.h // 2) - 2)
        pygame.draw.rect(surface, sheen_color, sheen_rect, border_radius=5)

        # 4. Inner 3D bezel border (light top-left, darker border overall)
        border_light = (
            min(255, color[0] + 60),
            min(255, color[1] + 60),
            min(255, color[2] + 60),
        )
        border_dark = (
            max(0, color[0] - 50),
            max(0, color[1] - 50),
            max(0, color[2] - 50),
        )
        pygame.draw.rect(surface, border_light, r, width=2, border_radius=8)
        inner_rect = Rect(r.x + 2, r.y + 2, r.w - 4, r.h - 4)
        pygame.draw.rect(surface, border_dark, inner_rect, width=1, border_radius=6)

        # 5. Draw health text with drop shadow in the center
        label = str(max(0, self.hp))
        text_shadow = font.render(label, True, (20, 28, 55))
        text_surf = font.render(label, True, COLOR_TEXT_WHITE)
        tx = r.centerx - text_surf.get_width() // 2
        ty = r.centery - text_surf.get_height() // 2
        surface.blit(text_shadow, (tx + 1, ty + 2))
        surface.blit(text_surf, (tx, ty))
