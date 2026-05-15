"""
Real SDSS galaxy positions coloured by UM channel.

Pulls ~50k galaxies from SDSS DR17 via astroquery, converts
(RA, Dec, z) → comoving Cartesian (Mpc/h), classifies each
galaxy by local number density into Light / Boundary / Matter,
and renders an interactive plotly 3D scatter.

UM axiom:  c² = c + 1
  Light    (1-r)² = 0.4775  → low density   → cyan   #4fc3f7
  Boundary 2r(1-r)= 0.4271  → interface     → gold   #ffd54f
  Matter   r²     = 0.0955  → high density  → red    #ef5350
"""

import numpy as np
from astroquery.sdss import SDSS
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from scipy.spatial import cKDTree
import plotly.graph_objects as go

# ── UM constants ──────────────────────────────────────────────
PHI = (1 + np.sqrt(5)) / 2
R   = 1 / (2 * PHI)
W_L = (1 - R) ** 2   # 0.4775
W_B = 2 * R * (1 - R) # 0.4271
W_M = R ** 2          # 0.0955

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

LIGHT_COL    = '#4fc3f7'
BOUNDARY_COL = '#ffd54f'
MATTER_COL   = '#ef5350'

# ── 1. Download SDSS galaxies ─────────────────────────────────
print("Querying SDSS DR17 …")
query = """
SELECT TOP 50000
    p.ra, p.dec, s.z
FROM PhotoObj AS p
JOIN SpecObj  AS s ON s.bestObjID = p.objID
WHERE s.class = 'GALAXY'
  AND s.z BETWEEN 0.02 AND 0.15
  AND s.zWarning = 0
  AND p.mode = 1
"""
result = SDSS.query_sql(query, data_release=17)
print(f"  {len(result)} galaxies returned")

ra  = np.array(result['ra'],  dtype=float)
dec = np.array(result['dec'], dtype=float)
z   = np.array(result['z'],   dtype=float)

# ── 2. RA/Dec/z  →  comoving Cartesian (Mpc) ─────────────────
dc = COSMO.comoving_distance(z).to(u.Mpc).value

ra_r  = np.radians(ra)
dec_r = np.radians(dec)

x = dc * np.cos(dec_r) * np.cos(ra_r)
y = dc * np.cos(dec_r) * np.sin(ra_r)
z_cart = dc * np.sin(dec_r)

pts = np.column_stack([x, y, z_cart])
print(f"  Comoving box: {pts.min(axis=0).round(1)} → {pts.max(axis=0).round(1)} Mpc")

# ── 3. Local density via 20-NN distance ───────────────────────
print("Computing local densities …")
tree = cKDTree(pts)
dists, _ = tree.query(pts, k=21)   # k=1 is self
density = 1.0 / dists[:, 20]       # proxy: inverse of 20th-NN distance

# Normalise to [0, 1]
d_min, d_max = np.percentile(density, 2), np.percentile(density, 98)
dn = np.clip((density - d_min) / (d_max - d_min), 0, 1)

# ── 4. Classify by UM thresholds ──────────────────────────────
# Low density (bottom W_L fraction) → Light
# High density (top W_M fraction)   → Matter
# Interface                         → Boundary
thresh_light  = np.percentile(dn, W_L * 100)      # ~47.75th pct
thresh_matter = np.percentile(dn, (1 - W_M) * 100) # ~90.45th pct

channel = np.full(len(dn), 'B', dtype='U1')
channel[dn <= thresh_light]  = 'L'
channel[dn >= thresh_matter] = 'M'

n_l = (channel == 'L').sum()
n_b = (channel == 'B').sum()
n_m = (channel == 'M').sum()
total = len(channel)
print(f"  Light {n_l/total:.3f}  Boundary {n_b/total:.3f}  Matter {n_m/total:.3f}")
print(f"  Target: L={W_L:.4f}  B={W_B:.4f}  M={W_M:.4f}")

# ── 5. Stratified downsample (keep visual quality, cut render load) ──
rng = np.random.default_rng(0)
def subsample(mask, n):
    idx = np.where(mask)[0]
    if len(idx) <= n: return mask
    keep = rng.choice(idx, n, replace=False)
    m = np.zeros(len(mask), bool); m[keep] = True
    return m

mask_L = subsample(channel == 'L', 8_000)   # thin voids — least structure
mask_B = subsample(channel == 'B', 10_000)  # keep filaments dense
mask_M = channel == 'M'                      # keep all clusters (~5k)
print(f"  Downsampled: L={mask_L.sum()}  B={mask_B.sum()}  M={mask_M.sum()}")

# ── 6. Plot ───────────────────────────────────────────────────
print("Rendering …")

def make_trace(mask, name, colour, size, opacity):
    return go.Scatter3d(
        x=x[mask], y=y[mask], z=z_cart[mask],
        mode='markers',
        name=name,
        marker=dict(size=size, color=colour, opacity=opacity,
                    line=dict(width=0)),
    )

fig = go.Figure(data=[
    make_trace(mask_L, f'Light  (1-r)²={W_L:.4f}',    LIGHT_COL,    1.2, 0.30),
    make_trace(mask_B, f'Boundary 2r(1-r)={W_B:.4f}', BOUNDARY_COL, 1.8, 0.60),
    make_trace(mask_M, f'Matter r²={W_M:.4f}',         MATTER_COL,   2.8, 0.95),
])

fig.update_layout(
    title=dict(
        text='SDSS Cosmic Web  ·  c² = c + 1',
        font=dict(color='#ff8c00', family='monospace', size=16),
        x=0.02, y=0.97,
    ),
    paper_bgcolor='#050510',
    plot_bgcolor ='#050510',
    scene=dict(
        bgcolor='#050510',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title='', showbackground=False, showspikes=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title='', showbackground=False, showspikes=False),
        zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title='', showbackground=False, showspikes=False),
        camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)),
    ),
    legend=dict(
        x=0.01, y=0.12,
        font=dict(color='#cccccc', family='monospace', size=11),
        bgcolor='rgba(0,0,0,0)',
        borderwidth=0,
    ),
    margin=dict(l=0, r=0, t=0, b=0),
)

# Annotation panel — no box, just neon text
fig.add_annotation(
    text=(
        '<span style="color:#ff8c00">φ = 1.6180  r = 0.3090</span><br>'
        f'<span style="color:#4fc3f7">Light    (1-r)² = {W_L:.4f}</span><br>'
        f'<span style="color:#ffd54f">Boundary 2r(1-r)= {W_B:.4f} ←</span><br>'
        f'<span style="color:#ef5350">Matter   r²     = {W_M:.4f}</span><br>'
        f'<br><span style="color:#555">{len(result):,} SDSS galaxies · z = 0.02–0.15</span>'
    ),
    xref='paper', yref='paper',
    x=0.01, y=0.40,
    showarrow=False,
    align='left',
    font=dict(family='monospace', size=11),
    bgcolor='rgba(0,0,0,0)',
    borderwidth=0,
)

out = 'figures/07_sdss_cosmic_web.html'
fig.write_html(out, include_plotlyjs='cdn')
print(f"Saved → {out}")
print("Open in browser — drag to rotate, scroll to zoom.")
