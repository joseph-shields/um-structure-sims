"""
SDSS cosmic web — classic 2D redshift slice (pie / cone diagram).

Thin Dec band ±5° around the celestial equator, plotted as a fan:
  radial = comoving distance (Mpc)
  angular = RA

Coloured by UM channel density classification.
"""

import numpy as np
from astroquery.sdss import SDSS
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from scipy.spatial import cKDTree
import plotly.graph_objects as go

PHI = (1 + np.sqrt(5)) / 2
R   = 1 / (2 * PHI)
W_L = (1 - R) ** 2
W_B = 2 * R * (1 - R)
W_M = R ** 2
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

LIGHT_COL    = '#4fc3f7'
BOUNDARY_COL = '#ffd54f'
MATTER_COL   = '#ef5350'

print("Querying SDSS DR17 (thin Dec slice) …")
query = """
SELECT TOP 80000
    p.ra, p.dec, s.z
FROM PhotoObj AS p
JOIN SpecObj  AS s ON s.bestObjID = p.objID
WHERE s.class = 'GALAXY'
  AND s.z BETWEEN 0.01 AND 0.20
  AND s.zWarning = 0
  AND p.mode = 1
  AND p.dec BETWEEN -5 AND 5
"""
result = SDSS.query_sql(query, data_release=17)
print(f"  {len(result)} galaxies in ±5° Dec band")

ra  = np.array(result['ra'],  dtype=float)
dec = np.array(result['dec'], dtype=float)
z   = np.array(result['z'],   dtype=float)

dc = COSMO.comoving_distance(z).to(u.Mpc).value

# Polar coords for the fan plot
ra_r = np.radians(ra)
px = dc * np.cos(ra_r)
py = dc * np.sin(ra_r)

# Density classification (same logic as 07)
pts_2d = np.column_stack([px, py])
tree   = cKDTree(pts_2d)
dists, _ = tree.query(pts_2d, k=16)
density  = 1.0 / dists[:, 15]
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

def trace(mask, name, col, size, opacity):
    return go.Scattergl(          # WebGL — 10x faster than SVG Scatter
        x=px[mask], y=py[mask],
        mode='markers',
        name=name,
        marker=dict(size=size, color=col, opacity=opacity, line=dict(width=0)),
    )

fig = go.Figure(data=[
    trace(channel=='L', f'Light  (1-r)²={W_L:.4f}',    LIGHT_COL,    1.5, 0.3),
    trace(channel=='B', f'Boundary 2r(1-r)={W_B:.4f}',  BOUNDARY_COL, 2.2, 0.6),
    trace(channel=='M', f'Matter r²={W_M:.4f}',          MATTER_COL,   3.5, 0.9),
])

fig.update_layout(
    title=dict(
        text='SDSS Cosmic Web  ·  2D Redshift Slice  ·  φ² = φ + 1',
        font=dict(color='#ff8c00', family='monospace', size=15),
        x=0.02, y=0.97,
    ),
    paper_bgcolor='#000000',
    plot_bgcolor ='#000000',
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='y'),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    legend=dict(
        font=dict(color='#ccc', family='monospace', size=11),
        bgcolor='rgba(0,0,0,0)',
        borderwidth=0,
        x=0.01, y=0.15,
    ),
    margin=dict(l=10, r=10, t=40, b=10),
)

fig.add_annotation(
    text=(
        '<span style="color:#ff8c00">φ = 1.6180  r = 0.3090</span><br>'
        f'<span style="color:#4fc3f7">Light    (1-r)² = {W_L:.4f}</span><br>'
        f'<span style="color:#ffd54f">Boundary 2r(1-r)= {W_B:.4f} ←</span><br>'
        f'<span style="color:#ef5350">Matter   r²     = {W_M:.4f}</span><br>'
        f'<br><span style="color:#444">Dec ±5°  ·  {len(result):,} galaxies</span>'
    ),
    xref='paper', yref='paper', x=0.01, y=0.95,
    showarrow=False, align='left',
    font=dict(family='monospace', size=11),
    bgcolor='rgba(0,0,0,0)', borderwidth=0,
)

out = 'figures/07b_sdss_slice.html'
fig.write_html(out, include_plotlyjs='cdn')
print(f"Saved → {out}")
