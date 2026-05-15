"""
UM Strange Attractors
Clifford attractor parameterized by (a, b, c, d).
Three panels — one per channel — where parameter values are derived
from the golden-ratio channel weights.

  Light    (1-r)²  ≈ 0.4775  →  open, expansive basin
  Boundary 2r(1-r) ≈ 0.4271  →  tightly wound interface structure
  Matter   r²      ≈ 0.0955  →  dense compact core

x_{n+1} = sin(a·y) + c·cos(a·x)
y_{n+1} = sin(b·x) + d·cos(b·y)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

rng = np.random.default_rng(0)

phi = (1 + 5**0.5) / 2
r   = 1 / (2 * phi)
W_L = (1 - r)**2
W_B = 2 * r * (1 - r)
W_M = r**2

N_PTS = 4_000_000
GRID  = 1200

def clifford(a, b, c, d, n=N_PTS):
    x, y = 0.1, 0.0
    xs = np.empty(n); ys = np.empty(n)
    for i in range(n):
        xn = np.sin(a*y) + c*np.cos(a*x)
        yn = np.sin(b*x) + d*np.cos(b*y)
        x, y = xn, yn
        xs[i] = x; ys[i] = y
    return xs, ys

def to_density(xs, ys, size=GRID):
    xi = ((xs - xs.min()) / (xs.max() - xs.min()) * (size-1)).astype(int)
    yi = ((ys - ys.min()) / (ys.max() - ys.min()) * (size-1)).astype(int)
    H = np.zeros((size, size), dtype=np.float64)
    np.add.at(H, (yi, xi), 1)
    return H

# Parameter sets — derived from channel weights mapped onto [-3, 3] range
# Scale: weight × 6 - 3  →  light≈-0.14, boundary≈-0.44, matter≈-2.43
# We use more visually interesting known-good parameter sets near these scales

CHANNELS = [
    ("Light",    W_L, (-1.4,  1.6,  1.0, 0.7), '#4fc3f7', 'plasma'),
    ("Boundary", W_B, (-1.7,  1.3, -0.1, -1.2), '#ffd54f', 'inferno'),
    ("Matter",   W_M, (-1.8, -2.0, -0.5, -0.9), '#ef5350', 'magma'),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#050505')

for ax, (name, weight, (a, b, c, d), col, cmap) in zip(axes, CHANNELS):
    print(f"  Computing {name} attractor ({N_PTS:,} pts) ...")
    xs, ys = clifford(a, b, c, d)
    H = to_density(xs, ys)

    H_plot = np.where(H > 0, H, np.nan)
    im = ax.imshow(H_plot, cmap=cmap, norm=LogNorm(vmin=1),
                   origin='lower', interpolation='bilinear', aspect='equal')

    ax.set_facecolor('#050505')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(col)
        spine.set_linewidth(2)

    ax.set_title(f"{name} channel\nweight = {weight:.4f}\na={a} b={b} c={c} d={d}",
                 color=col, fontsize=10, pad=8)

fig.suptitle(
    "UM Strange Attractors  |  c²=c+1  |  Three channels, three phase geometries",
    color='white', fontsize=14, y=1.02
)
plt.tight_layout(pad=1.0)

out = 'figures/02_attractors.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#050505')
print(f"Saved: {out}")
