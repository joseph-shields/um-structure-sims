"""
UM fractal — chaos game on a golden-angle triangle.

Rendered as a 1600×1600 density image (histogram2d per channel,
log-scaled, additive RGB blending) rather than 5M scatter points.
Loads instantly, looks better.
"""

import numpy as np
import plotly.graph_objects as go

PHI = (1 + np.sqrt(5)) / 2
R   = 1 / (2 * PHI)
W_L = (1 - R) ** 2
W_B = 2 * R * (1 - R)
W_M = R ** 2

# Channel RGB (0–1 float)
C_L = np.array([79,  195, 247]) / 255   # cyan
C_B = np.array([255, 213,  79]) / 255   # gold
C_M = np.array([239,  83,  80]) / 255   # red

# Vertices at golden-angle intervals
GA     = 2 * np.pi / PHI**2
angles = np.array([0, GA, 2*GA])
vx = np.cos(angles)
vy = np.sin(angles)

# Step sizes per channel (larger weight → smaller step → denser at vertex)
step  = np.array([1 - W_L, 1 - W_B, 1 - W_M])
probs = np.array([W_L, W_B, W_M])

N = 6_000_000
print(f"Running chaos game — {N:,} iterations …")

rng     = np.random.default_rng(42)
choices = rng.choice(3, size=N, p=probs)

x, y = 0.0, 0.0
xs = np.empty(N, dtype=np.float32)
ys = np.empty(N, dtype=np.float32)
ch = np.empty(N, dtype=np.int8)

for i in range(N):
    c    = choices[i]
    s    = step[c]
    x   += s * (vx[c] - x)
    y   += s * (vy[c] - y)
    xs[i] = x; ys[i] = y; ch[i] = c

print("Binning into density image …")
BINS  = 1600
xmin, xmax = xs.min(), xs.max()
ymin, ymax = ys.min(), ys.max()
extent = [[xmin, xmax], [ymin, ymax]]

img = np.zeros((BINS, BINS, 3), dtype=np.float32)

for idx, col in enumerate([C_L, C_B, C_M]):
    H, _, _ = np.histogram2d(xs[ch == idx], ys[ch == idx],
                              bins=BINS, range=extent)
    H = np.log1p(H)
    if H.max() > 0:
        H /= H.max()
    img[:, :, 0] += H.T * col[0]
    img[:, :, 1] += H.T * col[1]
    img[:, :, 2] += H.T * col[2]

img = np.clip(img * 255, 0, 255).astype(np.uint8)

print("Rendering …")
fig = go.Figure()
fig.add_trace(go.Image(z=img, hoverinfo='skip'))

fig.update_layout(
    title=dict(
        text='UM Fractal  ·  φ² = φ + 1  ·  Golden-Angle Chaos Game',
        font=dict(color='#ff8c00', family='monospace', size=15),
        x=0.02, y=0.97,
    ),
    paper_bgcolor='#000000',
    plot_bgcolor ='#000000',
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
               scaleanchor='y', scaleratio=1),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
               autorange='reversed'),
    margin=dict(l=10, r=10, t=40, b=10),
)

fig.add_annotation(
    text=(
        '<span style="color:#ff8c00">φ = 1.6180  r = 0.3090</span><br>'
        '<span style="color:#ff8c00">Golden angle = 360° / φ²</span><br><br>'
        f'<span style="color:#4fc3f7">Light    prob = {W_L:.4f}</span><br>'
        f'<span style="color:#ffd54f">Boundary prob = {W_B:.4f} ←</span><br>'
        f'<span style="color:#ef5350">Matter   prob = {W_M:.4f}</span><br>'
        f'<br><span style="color:#444">{N:,} pts  ·  {BINS}² density image</span>'
    ),
    xref='paper', yref='paper', x=0.01, y=0.95,
    showarrow=False, align='left',
    font=dict(family='monospace', size=11),
    bgcolor='rgba(0,0,0,0)', borderwidth=0,
)

out = 'figures/08_um_fractal.html'
fig.write_html(out, include_plotlyjs='cdn')
print(f"Saved → {out}")
