"""
Synthetic CMB sphere coloured by UM channels.

Uses go.Surface (single GPU mesh) instead of 196k scatter points —
renders in a fraction of the time at higher visual quality.
"""

import numpy as np
import healpy as hp
import plotly.graph_objects as go

PHI = (1 + np.sqrt(5)) / 2
R   = 1 / (2 * PHI)
W_L = (1 - R) ** 2
W_B = 2 * R * (1 - R)
W_M = R ** 2

LIGHT_COL    = '#4fc3f7'
BOUNDARY_COL = '#ffd54f'
MATTER_COL   = '#ef5350'

# ── 1. Planck 2018 best-fit power spectrum ────────────────────
NSIDE = 128
LMAX  = 3 * NSIDE - 1
ell   = np.arange(LMAX + 1, dtype=float); ell[0] = 1

def planck_cl(l):
    ns = 0.965; As = 2.2e-9
    cl = 2 * np.pi * As * (l / 10) ** (ns - 1) / (l * (l + 1))
    peaks = (
        0.60 * np.exp(-0.5 * ((l -  220) / 60) ** 2)
      + 0.25 * np.exp(-0.5 * ((l -  540) / 70) ** 2)
      + 0.12 * np.exp(-0.5 * ((l -  800) / 80) ** 2)
      + 0.06 * np.exp(-0.5 * ((l - 1050) / 90) ** 2)
    )
    return cl * (1 + 40 * peaks) * np.exp(-(l / 1500) ** 2)

Cl = planck_cl(ell); Cl[0] = Cl[1] = 0

print(f"Generating synthetic CMB map  NSIDE={NSIDE} …")
np.random.seed(42)
T_map = hp.synfast(Cl, nside=NSIDE, lmax=LMAX)

# ── 2. Classify pixels ────────────────────────────────────────
npix = hp.nside2npix(NSIDE)
tn = (T_map - T_map.min()) / (T_map.max() - T_map.min())
thresh_L = np.percentile(tn, W_L * 100)
thresh_M = np.percentile(tn, (1 - W_M) * 100)
channel = np.full(npix, 1.0)          # boundary = 1.0
channel[tn <= thresh_L] = 0.0         # light     = 0.0
channel[tn >= thresh_M] = 2.0         # matter    = 2.0

n_l = (channel==0).sum(); n_b = (channel==1).sum(); n_m = (channel==2).sum()
print(f"  L={n_l/npix:.3f}  B={n_b/npix:.3f}  M={n_m/npix:.3f}")

# ── 3. Build surface mesh ─────────────────────────────────────
# Sample sphere on a regular theta/phi grid, look up healpix pixel
NTHETA, NPHI = 256, 512
theta_1d = np.linspace(1e-6, np.pi - 1e-6, NTHETA)
phi_1d   = np.linspace(0, 2 * np.pi, NPHI)
THETA, PHI_G = np.meshgrid(theta_1d, phi_1d, indexing='ij')

pixels    = hp.ang2pix(NSIDE, THETA.ravel(), PHI_G.ravel())
color_grid = channel[pixels].reshape(NTHETA, NPHI)

SX = np.sin(THETA) * np.cos(PHI_G)
SY = np.sin(THETA) * np.sin(PHI_G)
SZ = np.cos(THETA)

# Discrete step colorscale: light → boundary → matter
colorscale = [
    [0.00, LIGHT_COL],    [0.33, LIGHT_COL],
    [0.34, BOUNDARY_COL], [0.66, BOUNDARY_COL],
    [0.67, MATTER_COL],   [1.00, MATTER_COL],
]

print("Rendering surface …")
fig = go.Figure(data=[go.Surface(
    x=SX, y=SY, z=SZ,
    surfacecolor=color_grid,
    cmin=0, cmax=2,
    colorscale=colorscale,
    showscale=False,
    lighting=dict(ambient=0.9, diffuse=0.3, specular=0.0),
    lightposition=dict(x=1, y=1, z=1),
)])

fig.update_layout(
    title=dict(
        text='UM CMB Sphere  ·  c² = c + 1',
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
        camera=dict(eye=dict(x=1.5, y=1.0, z=0.8)),
        aspectmode='data',
    ),
    margin=dict(l=0, r=0, t=0, b=0),
)

fig.add_annotation(
    text=(
        '<span style="color:#ff8c00">UM CMB  ·  c² = c + 1</span><br>'
        '<span style="color:#ff8c00">φ = 1.6180  r = 0.3090</span><br><br>'
        f'<span style="color:#4fc3f7">Light    cold voids  {W_L:.4f}</span><br>'
        f'<span style="color:#ffd54f">Boundary interface   {W_B:.4f} ←</span><br>'
        f'<span style="color:#ef5350">Matter   hot spots   {W_M:.4f}</span><br>'
        f'<br><span style="color:#444">synthetic CMB  ·  NSIDE={NSIDE}</span>'
    ),
    xref='paper', yref='paper', x=0.01, y=0.88,
    showarrow=False, align='left',
    font=dict(family='monospace', size=11),
    bgcolor='rgba(0,0,0,0)', borderwidth=0,
)

out = 'figures/09_cmb_um.html'
fig.write_html(out, include_plotlyjs='cdn')
print(f"Saved → {out}")
