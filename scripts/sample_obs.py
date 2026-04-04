"""
Renders sample observation images + raycast values from mid-lap positions.
Run from project root: uv run python scripts/sample_obs.py
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import math
import sys
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

pygame.init()
pygame.display.set_mode((1, 1))

from game.tracks import TRACKS
from game.oval_racer import draw_headlights, SCREEN_W, SCREEN_H

VIEW_PX   = 120
OUT_PX    = 64
GRASS     = (45, 110, 45)
RAY_ANGLES = [-90, -45, 0, 45, 90]
RAY_MAX    = 120
RAY_STEP   = 2


def raycast(track, x, y, angle):
    results = []
    for rel in RAY_ANGLES:
        abs_rad = math.radians(angle + rel)
        dx = math.cos(abs_rad) * RAY_STEP
        dy = math.sin(abs_rad) * RAY_STEP
        px, py = x, y
        dist = 0.0
        while dist < RAY_MAX:
            px += dx; py += dy; dist += RAY_STEP
            if not track.on_track(px, py):
                break
        results.append(dist / RAY_MAX)
    return results   # [left, front-left, front, front-right, right]


def render_obs(track, x, y, angle):
    surf = pygame.Surface((SCREEN_W, SCREEN_H))
    surf.blit(track.surface, (0, 0))
    draw_headlights(surf, x, y, angle)

    half = VIEW_PX // 2
    canvas = pygame.Surface((VIEW_PX, VIEW_PX))
    canvas.fill(GRASS)
    src = pygame.Rect(int(x) - half, int(y) - half, VIEW_PX, VIEW_PX)
    clipped = src.clip(pygame.Rect(0, 0, SCREEN_W, SCREEN_H))
    if clipped.width > 0 and clipped.height > 0:
        canvas.blit(surf, (clipped.x - src.x, clipped.y - src.y), clipped)

    rotated = pygame.transform.rotate(canvas, -(angle - 270))
    rw, rh  = rotated.get_size()
    cx, cy  = rw // 2, rh // 2
    inner   = pygame.Rect(cx - half, cy - half, VIEW_PX, VIEW_PX)
    inner   = inner.clip(rotated.get_rect())
    cropped = pygame.Surface((VIEW_PX, VIEW_PX))
    cropped.fill(GRASS)
    cropped.blit(rotated, (inner.x - (cx - half), inner.y - (cy - half)), inner)
    return pygame.transform.scale(cropped, (OUT_PX, OUT_PX))


def sim_steps(track, steps, steer=0):
    x = float(track.start_pos[0])
    y = float(track.start_pos[1])
    angle = float(track.start_angle)
    speed = 0.0
    ms = track.max_speed
    for i in range(steps):
        s = steer + math.sin(i * 0.04) * 0.4
        ratio = min(abs(speed) / ms, 1.0) if ms > 0 else 0
        angle += s * 2.7 * max(0.3, ratio)
        speed  = min(speed + 0.13, ms)
        speed  = max(0.0, speed - 0.038)
        rad    = math.radians(angle)
        x     += speed * math.cos(rad)
        y     += speed * math.sin(rad)
    return x, y, angle


SAMPLES = [
    (0,   80,  0.0, "straight"),
    (0,  220,  0.3, "turn"),
    (4,  150,  0.2, "rect_corner"),
    (8,  200,  0.5, "hairpin"),
    (12, 120,  0.0, "polygon"),
]

out_dir = os.path.join(os.path.dirname(__file__), "obs_samples")
os.makedirs(out_dir, exist_ok=True)

# Grid: image row + ray bar row
CELL     = OUT_PX + 4
BAR_H    = 40
LABEL_H  = 12
NCOLS    = len(SAMPLES)
grid_w   = NCOLS * CELL + 4
grid_h   = 4 + OUT_PX + 4 + BAR_H + LABEL_H + 4
grid     = pygame.Surface((grid_w, grid_h))
grid.fill((20, 20, 20))

font_s = pygame.font.SysFont("consolas", 8)
font_t = pygame.font.SysFont("consolas", 9, bold=True)

RAY_LABELS = ["L", "FL", "F", "FR", "R"]
BAR_COLORS = [
    (100, 180, 255),  # left   — blue
    (150, 220, 150),  # front-left — light green
    (255, 220,  80),  # front  — yellow
    (150, 220, 150),  # front-right
    (100, 180, 255),  # right  — blue
]

for col, (tidx, steps, steer, label) in enumerate(SAMPLES):
    track = TRACKS[tidx]
    track.build()
    x, y, angle = sim_steps(track, steps, steer=steer)
    rays = raycast(track, x, y, angle)

    # observation image
    img_surf = render_obs(track, x, y, angle)
    gx = 2 + col * CELL
    grid.blit(img_surf, (gx, 4))
    pygame.image.save(img_surf, os.path.join(out_dir, f"{col+1}_{label}.png"))

    # raycast bar chart
    bar_y = 4 + OUT_PX + 4
    bar_w = (CELL - 4) // len(rays)
    for ri, (r, c) in enumerate(zip(rays, BAR_COLORS)):
        bx = gx + ri * bar_w
        full_h = BAR_H - 2
        fill_h = int(r * full_h)
        # background
        pygame.draw.rect(grid, (50, 50, 50), (bx, bar_y, bar_w - 1, full_h))
        # fill
        pygame.draw.rect(grid, c, (bx, bar_y + full_h - fill_h, bar_w - 1, fill_h))
        # value text
        txt = font_s.render(f"{r:.2f}", True, (200, 200, 200))
        grid.blit(txt, (bx + 1, bar_y + full_h - fill_h - 9))

    # column label
    txt = font_t.render(label, True, (180, 180, 180))
    grid.blit(txt, (gx, bar_y + BAR_H))

    on = track.on_track(x, y)
    print(f"{label:20s}  on_track={on}  rays={[f'{r:.2f}' for r in rays]}")

grid_path = os.path.join(out_dir, "grid.png")
pygame.image.save(grid, grid_path)
print(f"saved {grid_path}")
pygame.quit()
