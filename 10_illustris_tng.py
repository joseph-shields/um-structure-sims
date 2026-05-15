"""
IllustrisTNG cosmic web — subhalo positions coloured by UM channel.

Requires a free API key from https://www.tng-project.org/users/register/
Set your key below or pass as environment variable TNG_API_KEY.

Uses TNG300-1 snapshot 99 (z=0) subhalo catalog — the largest
publicly available cosmological simulation volume (300 Mpc/h box).
"""

import os
import sys
import numpy as np
import requests
import plotly.graph_objects as go
from scipy.spatial import cKDTree

PHI = (1 + np.sqrt(5)) / 2
R   = 1 / (2 * PHI)
W_L = (1 - R) ** 2
W_B = 2 * R * (1 - R)
W_M = R ** 2

LIGHT_COL    = '#4fc3f7'
BOUNDARY_COL = '#ffd54f'
MATTER_COL   = '#ef5350'

# ── API key ───────────────────────────────────────────────────
API_KEY = os.environ.get('TNG_API_KEY', '')
if not API_KEY:
    print("No TNG_API_KEY found.")
    print("Register free at: https://www.tng-project.org/users/register/")
    print("Then run:  TNG_API_KEY=your_key python3 10_illustris_tng.py")
    sys.exit(1)

BASE = 'https://www.tng-project.org/api'
HEADERS = {'api-key': API_KEY}

def get(path, params=None):
    r = requests.get(f'{BASE}{path}', headers=HEADERS, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

# ── Download subhalo catalog ──────────────────────────────────
SIM  = 'TNG300-1'
SNAP = 99   # z = 0

print(f"Fetching {SIM} snapshot {SNAP} subhalo catalog …")
params = {
    'limit': 50000,
    'fields': 'SubhaloPos,SubhaloMass,SubhaloLen',
    'SubhaloMass__gt': '0.01',    # mass > 0.01 × 10^10 M☉/h
}
data = get(f'/TNG300-1/snapshots/{SNAP}/subhalos/', params=params)
results = data['results']
print(f"  {len(results)} subhalos returned")

pos  = np.array([[s['SubhaloPos'][0], s['SubhaloPos'][1], s['SubhaloPos'][2]]
                  for s in results], dtype=float)
mass = np.array([s['SubhaloMass'] for s in results], dtype=float)

# TNG positions are in comoving kpc/h — convert to Mpc/h
pos /= 1000.0
print(f"  Box: {pos.min(axis=0).round(1)} → {pos.max(axis=0).round(1)} Mpc/h")

# ── Density classification ────────────────────────────────────
print("Computing local densities …")
tree = cKDTree(pos)
dists, _ = tree.query(pos, k=21)
density = 1.0 / dists[:, 20]
d_min, d_max = np.percentile(density, 2), np.percentile(density, 98)
dn = np.clip((density - d_min) / (d_max - d_min), 0, 1)

thresh_light  = np.percentile(dn, W_L * 100)
thresh_matter = np.percentile(dn, (1 - W_M) * 100)
channel = np.full(len(dn), 'B', dtype='U1')
channel[dn <= thresh_light]  = 'L'
channel[dn >= thresh_matter] = 'M'

n_l = (channel=='L').sum(); n_b = (channel=='B').sum(); n_m = (channel=='M').sum()
tot = len(channel)
print(f"  L={n_l/tot:.3f}  B={n_b/tot:.3f}  M={n_m/tot:.3f}")

# ── Plot ──────────────────────────────────────────────────────
print("Rendering …")
x, y, z = pos[:,0], pos[:,1], pos[:,2]

def trace(mask, name, col, size, opacity):
    return go.Scatter3d(
        x=x[mask], y=y[mask], z=z[mask],
        mode='markers', name=name,
        marker=dict(size=size, color=col, opacity=opacity, line=dict(width=0)),
    )

fig = go.Figure(data=[
    trace(channel=='L', f'Light  (1-r)²={W_L:.4f}',    LIGHT_COL,    1.0, 0.20),
    trace(channel=='B', f'Boundary 2r(1-r)={W_B:.4f}',  BOUNDARY_COL, 1.8, 0.55),
    trace(channel=='M', f'Matter r²={W_M:.4f}',          MATTER_COL,   3.0, 0.90),
])

fig.update_layout(
    title=dict(
        text=f'IllustrisTNG {SIM}  ·  z=0  ·  c² = c + 1',
        font=dict(color='#ff8c00', family='monospace', size=15),
        x=0.02, y=0.97,
    ),
    paper_bgcolor='#000000',
    scene=dict(
        bgcolor='#000000',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   showbackground=False, showspikes=False, title=''),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   showbackground=False, showspikes=False, title=''),
        zaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   showbackground=False, showspikes=False, title=''),
        camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)),
    ),
    legend=dict(
        font=dict(color='#ccc', family='monospace', size=11),
        bgcolor='rgba(0,0,0,0)', borderwidth=0, x=0.01, y=0.15,
    ),
    margin=dict(l=0, r=0, t=0, b=0),
)

fig.add_annotation(
    text=(
        '<span style="color:#ff8c00">φ = 1.6180  r = 0.3090</span><br>'
        f'<span style="color:#4fc3f7">Light    (1-r)² = {W_L:.4f}</span><br>'
        f'<span style="color:#ffd54f">Boundary 2r(1-r)= {W_B:.4f} ←</span><br>'
        f'<span style="color:#ef5350">Matter   r²     = {W_M:.4f}</span><br>'
        f'<br><span style="color:#444">{SIM}  ·  {len(results):,} subhalos  ·  300 Mpc/h box</span>'
    ),
    xref='paper', yref='paper', x=0.01, y=0.88,
    showarrow=False, align='left',
    font=dict(family='monospace', size=11),
    bgcolor='rgba(0,0,0,0)', borderwidth=0,
)

out = 'figures/10_illustris_tng.html'
fig.write_html(out, include_plotlyjs='cdn')
print(f"Saved → {out}")
