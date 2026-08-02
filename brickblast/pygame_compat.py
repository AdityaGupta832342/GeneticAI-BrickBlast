"""
Pygame compatibility layer.
If real Pygame is installed, imports and exposes pygame.
Otherwise, provides a Pillow (PIL) backed fallback so the game and RL environment
can run headless, render PNG frames, and execute unit tests without X11 or Pygame.
"""
import os
import sys

try:
    import pygame as _real_pygame
    pygame = _real_pygame
    PYGAME_AVAILABLE = True
    # Re-export real pygame types so `from pygame_compat import Rect, Surface` works
    Rect = _real_pygame.Rect
    Surface = _real_pygame.Surface
except ImportError:
    PYGAME_AVAILABLE = False
    from PIL import Image, ImageDraw, ImageFont

    class Rect:
        def __init__(self, x, y, w, h):
            self.x = int(x)
            self.y = int(y)
            self.w = int(w)
            self.h = int(h)

        @property
        def left(self): return self.x
        @property
        def top(self): return self.y
        @property
        def right(self): return self.x + self.w
        @property
        def bottom(self): return self.y + self.h
        @property
        def width(self): return self.w
        @property
        def height(self): return self.h
        @property
        def center(self): return (self.x + self.w // 2, self.y + self.h // 2)
        @property
        def centerx(self): return self.x + self.w // 2
        @property
        def centery(self): return self.y + self.h // 2

        def collidepoint(self, px, py):
            return (self.left <= px <= self.right) and (self.top <= py <= self.bottom)

        def colliderect(self, other):
            return not (self.right < other.left or self.left > other.right or
                        self.bottom < other.top or self.top > other.bottom)

        def inflate(self, dx, dy):
            return Rect(self.x - dx//2, self.y - dy//2, self.w + dx, self.h + dy)

        def __iter__(self):
            return iter((self.x, self.y, self.w, self.h))

    class Surface:
        def __init__(self, size, flags=0, depth=32):
            self.size = size
            self.image = Image.new("RGBA", (size[0], size[1]), (0, 0, 0, 0))
            self.draw = ImageDraw.Draw(self.image)

        def get_width(self): return self.size[0]
        def get_height(self): return self.size[1]
        def get_rect(self, **kwargs):
            r = Rect(0, 0, self.size[0], self.size[1])
            if "center" in kwargs:
                cx, cy = kwargs["center"]
                r.x = cx - r.w // 2
                r.y = cy - r.h // 2
            return r

        def fill(self, color):
            if isinstance(color, (list, tuple)):
                c = tuple(color[:4])
            else:
                c = color
            self.image.paste(c, [0, 0, self.size[0], self.size[1]])
            self.draw = ImageDraw.Draw(self.image)

        def blit(self, source, dest):
            if isinstance(dest, Rect):
                pos = (dest.x, dest.y)
            elif isinstance(dest, (tuple, list)):
                pos = (int(dest[0]), int(dest[1]))
            else:
                pos = (0, 0)
            try:
                self.image.paste(source.image, pos, source.image)
            except Exception:
                self.image.paste(source.image, pos)
            self.draw = ImageDraw.Draw(self.image)

        def save(self, filepath):
            self.image.save(filepath)

    class FontObj:
        def __init__(self, size):
            self.size = size
            try:
                self.font = ImageFont.truetype("DejaVuSans.ttf", size)
            except Exception:
                self.font = ImageFont.load_default()

        def render(self, text, antialias, color, background=None):
            text = str(text)
            # Estimate text size
            w = max(len(text) * int(self.size * 0.6), 10)
            h = int(self.size * 1.3)
            surf = Surface((w, h))
            if background:
                surf.fill(background)
            c = tuple(color[:4]) if isinstance(color, (tuple, list)) else color
            surf.draw.text((2, 2), text, fill=c, font=self.font)
            return surf

    class _FontModule:
        def __init__(self):
            pass
        def init(self): pass
        def SysFont(self, name, size, bold=False):
            return FontObj(size)
        def Font(self, path, size):
            return FontObj(size)

    class _DrawModule:
        def rect(self, surface, color, rect, width=0, border_radius=0):
            r = (rect.x, rect.y, rect.right - 1, rect.bottom - 1)
            c = tuple(color[:4]) if isinstance(color, (tuple, list)) else color
            if width > 0:
                surface.draw.rectangle(r, outline=c, width=width)
            else:
                surface.draw.rectangle(r, fill=c)

        def circle(self, surface, color, center, radius, width=0):
            cx, cy = int(center[0]), int(center[1])
            r = (cx - radius, cy - radius, cx + radius, cy + radius)
            c = tuple(color[:4]) if isinstance(color, (tuple, list)) else color
            if width > 0:
                surface.draw.ellipse(r, outline=c, width=width)
            else:
                surface.draw.ellipse(r, fill=c)

        def line(self, surface, color, start_pos, end_pos, width=1):
            c = tuple(color[:4]) if isinstance(color, (tuple, list)) else color
            surface.draw.line([tuple(start_pos), tuple(end_pos)], fill=c, width=width)

        def polygon(self, surface, color, points, width=0):
            pts = [tuple(p) for p in points]
            c = tuple(color[:4]) if isinstance(color, (tuple, list)) else color
            if width > 0:
                surface.draw.polygon(pts, outline=c)
            else:
                surface.draw.polygon(pts, fill=c)

    class _DisplayModule:
        def __init__(self):
            self._surface = None
        def set_mode(self, size, flags=0, depth=0):
            self._surface = Surface(size)
            return self._surface
        def set_caption(self, title):
            pass
        def update(self):
            if os.environ.get("PYGAME_COMPAT_SAVE_FRAME"):
                if self._surface:
                    self._surface.save(os.environ["PYGAME_COMPAT_SAVE_FRAME"])
        def flip(self):
            self.update()
        def get_surface(self):
            return self._surface

    class _TimeModule:
        class Clock:
            def tick(self, fps=60):
                return int(1000 / max(1, fps))

    class _EventModule:
        def get(self, *args):
            return []
        def pump(self): pass

    class _KeyModule:
        def get_pressed(self):
            from collections import defaultdict
            return defaultdict(lambda: False)

    class _MouseModule:
        def get_pos(self):
            return (240, 400)
        def get_pressed(self):
            return (0, 0, 0)

    class _PygameStub:
        def __init__(self):
            self.SRCALPHA = 65536
            self.QUIT = 256
            self.MOUSEBUTTONDOWN = 1025
            self.MOUSEBUTTONUP = 1026
            self.KEYDOWN = 768
            self.K_SPACE = 32
            self.K_f = 102
            self.K_r = 114
            self.K_ESCAPE = 27
            self.font = _FontModule()
            self.draw = _DrawModule()
            self.display = _DisplayModule()
            self.time = _TimeModule()
            self.event = _EventModule()
            self.key = _KeyModule()
            self.mouse = _MouseModule()
            self.Surface = Surface
            self.Rect = Rect

        def init(self):
            pass
        def quit(self):
            pass

    pygame = _PygameStub()
