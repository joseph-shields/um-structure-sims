# Claude Design Brief — UM Interactive Visuals
## Two self-contained HTML files. No server. No dependencies. Just open in browser.

---

# BRIEF 1: Wolfram Hypergraph Viewer

## What it is
A 3D interactive hypergraph — nodes as spheres, 3-body edges as semi-transparent
triangles — styled like the Wolfram Physics Project's own renders but coloured by
Unified Mechanics channel (light / boundary / matter).

## Data source
Embed the JSON below directly in the HTML as a const. Do not fetch it.

```
[paste full contents of figures/wolfram_graph.json here]
```

## Visual spec
- **Background**: #050510 (near-black blue)
- **Nodes**: spheres, coloured by generation using a gradient:
  - gen=0 (earliest): rgb(30, 120, 240) — deep blue
  - gen=mid: rgb(255, 200, 40) — gold
  - gen=max (latest): rgb(250, 40, 40) — red
  - Node BORDER colour = channel: cyan (#4fc3f7) = Light, gold (#ffd54f) = Boundary, red (#ef5350) = Matter
- **Edges** (3-body hyperedges): filled triangles, same gen-gradient colour, opacity 0.08–0.12
- **Wireframe lines**: connecting triangle vertices, opacity 0.3, width 1px
- **Camera**: orbiting, mouse drag to rotate, scroll to zoom, auto-slow-rotate on idle

## Overlay (top-left corner, small)
```
φ² = φ + 1
r = 1/(2φ) = 0.3090

Light    (1-r)² = 0.4775
Boundary 2r(1-r) = 0.4271  ←
Matter   r²     = 0.0955

425 nodes · 428 edges
Wolfram standard rule
```

## Legend (bottom-right, small)
Three coloured dots: cyan = Light, gold = Boundary, red = Matter
Colour bar: blue → gold → red = early → late generation

## Interaction
- Mouse drag: rotate
- Scroll: zoom
- Click node: show tooltip (node id, generation, channel)
- Double-click: reset camera
- Idle (3s): slow auto-rotate

## Technology
Three.js (CDN). InstancedMesh for nodes, custom BufferGeometry for triangles.
Single HTML file, no external assets except Three.js CDN.

---

# BRIEF 2: CMB Sphere — Procedural WebGL

## What it is
An interactive 3D sphere showing a procedurally generated CMB-like temperature map
derived from the three UM channel weights. No pre-computed tiles. No server.
The GPU computes everything from the equation φ² = φ + 1.

## The physics
Three channels from φ² = φ + 1:
```
φ = (1 + √5) / 2 = 1.6180339887
r = 1 / (2φ)     = 0.3090169944

Light    (1-r)²   = 0.4775  — smooth, low frequency, propagating
Boundary 2r(1-r)  = 0.4271  — labyrinthine, medium frequency, interface
Matter   r²       = 0.0955  — dense spots, high frequency, accumulating
```

## Fragment shader spec
The sphere texture is generated entirely in GLSL. For each point on the sphere:

1. **Determine channel zone** by the point's position (use a combination of
   spherical harmonics weighted by channel fractions — light dominates the
   "smooth open" areas, boundary forms the labyrinthine mid-zones, matter
   forms dense compact spots)

2. **Generate noise per zone**:
   - Light zone: large-scale smooth simplex noise (low frequency ~3–5 octaves)
   - Boundary zone: medium-scale turbulent noise (labyrinthine, ~6–8 octaves,
     think reaction-diffusion texture)
   - Matter zone: small-scale spot noise (voronoi-ish, tight clusters)

3. **Mix zones** using the channel weights as blend factors

4. **Apply CMB colormap** (standard false-colour):
   - cold (T < mean): deep blue → cyan
   - mean: black
   - hot (T > mean): orange → red → white
   - ΔT range: ±300 μK (just for visual, scale doesn't matter)

## Visual spec
- **Background**: #000000
- **Sphere**: radius 1.0, 512×512 segments minimum for smooth shading
- **Rotation**: slow auto-rotate on idle (0.05°/frame), mouse drag to spin
- **Zoom**: scroll wheel

## Overlay (top-left)
```
UM CMB  ·  φ² = φ + 1
φ = 1.6180  r = 0.3090

Light    0.4775  smooth
Boundary 0.4271  interface  ←
Matter   0.0955  dense
```

## Buttons (bottom centre, minimal dark style)
- **UM** / **Planck** toggle (Planck = real CMB colormap from embedded data,
  just a placeholder greyed-out button is fine if not implementing dual mode)
- **Rotate** on/off
- **Wireframe** (shows the spherical harmonic decomposition grid)

## Technology
Three.js + custom ShaderMaterial (GLSL fragment shader).
All noise functions implemented in GLSL (no external noise library needed —
include classic Perlin/simplex noise functions inline).
Single HTML file, Three.js from CDN only.

---

# Shared style notes
- Font: system-ui or -apple-system, monospace for numbers
- All overlays: rgba(8,8,12,0.82) background, 1px solid #2a2a30 border,
  8px border-radius, backdrop-filter blur
- Colour language: cyan = light, gold = boundary, red = matter — consistent
  across both files
- Both files should open by double-clicking. No python, no npm, no server.
