"""
UM Fractal Growth — Eden Model + Fibonacci Tiling
Two complementary visuals:

Panel A: Eden growth model (DLA family, ~100x faster than random-walk DLA)
  Sticking probability weighted by channel zone — same physics as DLA,
  computable in seconds via frontier-set approach.

Panel B: Fibonacci / φ-tiling
  The golden ratio IS r = 1/(2φ). The tiling that emerges from φ-ratios
  naturally partitions into three zone types matching the channel weights.
"""

import math, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque

rng = np.random.default_rng(13)
random.seed(13)

phi = (1 + 5**0.5) / 2
r_  = 1 / (2 * phi)
W_L = (1 - r_)**2
W_B = 2 * r_ * (1 - r_)
W_M = r_**2

# ---------------------------------------------------------------
# Panel A: Eden Growth
# ---------------------------------------------------------------
GRID  = 401
CX    = GRID // 2
R_MAX = GRID // 2 - 5

R_M = int(R_MAX * W_M)
R_B = int(R_MAX * (W_M + W_B))

STICK = {'L': 0.95, 'B': 0.55, 'M': 0.20}

def channel_at(x, y):
    d = math.hypot(x - CX, y - CX)
    if d < R_M: return 'M'
    if d < R_B: return 'B'
    return 'L'

channel_code = {'L': 1, 'B': 2, 'M': 3}
COLORS_ARR = {0: [8,8,8], 1: [79,195,247], 2: [255,213,79], 3: [239,83,80]}

grid         = np.zeros((GRID, GRID), dtype=np.int8)
channel_grid = np.zeros((GRID, GRID), dtype=np.int8)

grid[CX, CX] = 1
channel_grid[CX, CX] = 3  # matter seed

# Frontier: set of empty cells adjacent to cluster
frontier = set()
DIRS = [(0,1),(0,-1),(1,0),(-1,0)]
for dx, dy in DIRS:
    nx_, ny_ = CX+dx, CX+dy
    if 0 < nx_ < GRID-1 and 0 < ny_ < GRID-1:
        frontier.add((nx_, ny_))

N_GROW = 25000
stuck  = 1

print(f"Running Eden growth  ({N_GROW} attempts) ...")
attempts = 0
while stuck < N_GROW and frontier:
    attempts += 1
    # Pick random frontier cell
    cell = random.choice(list(frontier))
    x, y = cell
    ch = channel_at(x, y)
    if rng.random() < STICK[ch]:
        grid[x, y] = 1
        channel_grid[x, y] = channel_code[ch]
        stuck += 1
        frontier.discard(cell)
        for dx, dy in DIRS:
            nx_, ny_ = x+dx, y+dy
            if (0 < nx_ < GRID-1 and 0 < ny_ < GRID-1
                    and grid[nx_, ny_] == 0
                    and math.hypot(nx_-CX, ny_-CX) < R_MAX):
                frontier.add((nx_, ny_))
    else:
        frontier.discard(cell)  # rejected, remove to avoid infinite loop
        # Re-add with some probability to keep frontier alive
        for dx, dy in DIRS:
            nx_, ny_ = x+dx, y+dy
            if (0 < nx_ < GRID-1 and 0 < ny_ < GRID-1
                    and grid[nx_, ny_] == 0):
                frontier.add((nx_, ny_))
    if stuck % 5000 == 0 and stuck > 0:
        print(f"  {stuck} cells grown")

print(f"  Done — {stuck} cells, {attempts} attempts")

# ---------------------------------------------------------------
# Panel B: Fibonacci / φ tiling coloured by channel
# ---------------------------------------------------------------
# Sunflower seed pattern: n seeds at angle n*2π/φ², radius √n
# Channel assignment by radial zone (same proportions as Eden)

N_SEEDS  = 4000
PLOT_R   = 35.0

angles = np.arange(1, N_SEEDS+1) * 2 * math.pi / phi**2
radii  = np.sqrt(np.arange(1, N_SEEDS+1))
xs     = radii * np.cos(angles)
ys     = radii * np.sin(angles)

# Normalise radii to [0,1]
r_norm = radii / radii.max()
ch_arr = np.where(r_norm < W_M, 'M',
         np.where(r_norm < W_M + W_B, 'B', 'L'))
colors_seed = np.where(ch_arr == 'L', '#4fc3f7',
              np.where(ch_arr == 'B', '#ffd54f', '#ef5350'))

# ---------------------------------------------------------------
# Figure
# ---------------------------------------------------------------
fig = plt.figure(figsize=(18, 8))
fig.patch.set_facecolor('#080808')

# Panel A
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_facecolor('#080808')
ax1.set_title("Eden Growth — channel by sticking zone", color='white', fontsize=12)

img = np.zeros((GRID, GRID, 3), dtype=np.uint8)
for code, col in COLORS_ARR.items():
    mask = channel_grid == code
    img[mask] = col
img[grid == 0] = [8, 8, 8]

ax1.imshow(img, origin='lower', interpolation='nearest')

theta = np.linspace(0, 2*math.pi, 300)
for R, col, lbl in [(R_M,'#ef5350',''), (R_B,'#ffd54f',''), (R_MAX,'#4fc3f7','')]:
    ax1.plot(CX + R*np.cos(theta), CX + R*np.sin(theta),
             color=col, lw=0.8, alpha=0.4, linestyle='--')

patches = [
    mpatches.Patch(color='#4fc3f7', label=f'Light  (1-r)²={W_L:.3f}  stick={STICK["L"]}'),
    mpatches.Patch(color='#ffd54f', label=f'Boundary 2r(1-r)={W_B:.3f}  stick={STICK["B"]}'),
    mpatches.Patch(color='#ef5350', label=f'Matter r²={W_M:.3f}  stick={STICK["M"]}'),
]
ax1.legend(handles=patches, loc='lower right', facecolor='#111',
           labelcolor='white', fontsize=8)
ax1.axis('off')

# Panel B
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_facecolor('#080808')
ax2.set_title(f"Fibonacci φ-Tiling — {N_SEEDS} seeds, channel by radial zone",
              color='white', fontsize=12)

dot_sizes = np.where(ch_arr == 'M', 18, np.where(ch_arr == 'B', 10, 5))
ax2.scatter(xs, ys, c=colors_seed, s=dot_sizes, alpha=0.85, linewidths=0)

# Annotate zone boundaries
for frac, col, lbl in [(W_M, '#ef5350', f'Matter r²={W_M:.3f}'),
                        (W_M+W_B, '#ffd54f', f'Boundary 2r(1-r)={W_B:.3f}')]:
    R_ring = math.sqrt(frac * N_SEEDS)
    circle = plt.Circle((0,0), R_ring, color=col, fill=False, lw=0.8,
                         linestyle='--', alpha=0.5)
    ax2.add_patch(circle)
    ax2.text(R_ring*0.707, R_ring*0.707, lbl, color=col, fontsize=7)

ax2.set_aspect('equal')
ax2.axis('off')
ax2.text(0, -math.sqrt(N_SEEDS)*1.05,
         f"r = 1/(2φ) = {r_:.4f}   φ = (1+√5)/2 = {phi:.4f}",
         color='#aaa', fontsize=9, ha='center')

fig.suptitle(
    "UM Fractal Growth  |  c²=c+1  |  Eden model + Fibonacci φ-tiling",
    color='white', fontsize=13, y=1.01
)
plt.tight_layout(pad=1.5)

out = 'figures/04_dla.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#080808')
print(f"Saved: {out}")
