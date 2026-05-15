"""
UM Gray-Scott Reaction-Diffusion
Three channel weights from c²=c+1 encoded as spatially-varying feed/kill fields.
Each channel region produces a morphologically distinct Turing pattern.

  Light    (1-r)²  ≈ 0.4775  →  coral / propagating wave   (high f, medium k)
  Boundary 2r(1-r) ≈ 0.4271  →  labyrinthine stripes        (balanced f, k)
  Matter   r²      ≈ 0.0955  →  dense isolated spots        (low f, high k)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import laplace

rng = np.random.default_rng(42)

phi = (1 + 5**0.5) / 2
r   = 1 / (2 * phi)
W_L = (1 - r)**2       # 0.4775
W_B = 2 * r * (1 - r)  # 0.4271
W_M = r**2             # 0.0955

N     = 512
steps = 8000
dt    = 1.0
Du    = 0.16
Dv    = 0.08

# Known Gray-Scott parameter sets for each channel's character
CHANNEL_PARAMS = {
    'L': (0.055, 0.062),   # coral — propagating, open
    'B': (0.037, 0.060),   # labyrinth — boundary, stripe
    'M': (0.030, 0.057),   # spots — dense, accumulating
}

# Build spatially-varying f, k fields tiled in three vertical bands
# proportional to channel weights
x_L = int(N * W_L)
x_B = int(N * (W_L + W_B))

f_field = np.zeros((N, N))
k_field = np.zeros((N, N))

f_field[:, :x_L]      = CHANNEL_PARAMS['L'][0]
k_field[:, :x_L]      = CHANNEL_PARAMS['L'][1]
f_field[:, x_L:x_B]   = CHANNEL_PARAMS['B'][0]
k_field[:, x_L:x_B]   = CHANNEL_PARAMS['B'][1]
f_field[:, x_B:]       = CHANNEL_PARAMS['M'][0]
k_field[:, x_B:]       = CHANNEL_PARAMS['M'][1]

# Initialise: u=1, v=0 everywhere, small v perturbation in centre
U = np.ones((N, N))
V = np.zeros((N, N))
c = N // 2
r0 = 20
U[c-r0:c+r0, c-r0:c+r0] = 0.50
V[c-r0:c+r0, c-r0:c+r0] = 0.25
U += 0.02 * rng.random((N, N))
V += 0.02 * rng.random((N, N))

print("Running Gray-Scott  (8000 steps) ...")
for i in range(steps):
    uvv  = U * V * V
    dU   = Du * laplace(U, mode='wrap') - uvv + f_field * (1 - U)
    dV   = Dv * laplace(V, mode='wrap') + uvv - (f_field + k_field) * V
    U   += dt * dU
    V   += dt * dV
    if (i+1) % 2000 == 0:
        print(f"  step {i+1}/{steps}")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('#0a0a0a')

for ax, (data, title, cmap) in zip(axes, [
    (V,   "V  (activator)", 'inferno'),
    (U,   "U  (inhibitor)", 'viridis'),
]):
    im = ax.imshow(data, cmap=cmap, interpolation='bilinear', origin='lower')
    ax.set_title(title, color='white', fontsize=13)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    # Annotate channel boundaries
    for xpos, label, col in [
        (x_L/N,        f"← Light (1-r)²={W_L:.3f}",    '#4fc3f7'),
        ((x_L+x_B)/2/N, f"Boundary 2r(1-r)={W_B:.3f}", '#ffd54f'),
        ((x_B+N)/2/N,   f"Matter r²={W_M:.3f} →",       '#ef5350'),
    ]:
        ax.text(xpos, 1.03, label, transform=ax.transAxes,
                color=col, fontsize=8, ha='center')

    ax.axvline(x_L,  color='#4fc3f7', lw=1.2, alpha=0.7)
    ax.axvline(x_B,  color='#ffd54f', lw=1.2, alpha=0.7)

fig.suptitle(
    f"UM Gray-Scott  |  c²=c+1  →  r={r:.4f}  |  Three channels, one field",
    color='white', fontsize=14, y=1.01
)
plt.tight_layout(pad=1.5)

out = 'figures/01_gray_scott.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
print(f"Saved: {out}")
