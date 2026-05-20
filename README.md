# UM Structure Simulations

**Joseph Shields · 2026**

Five simulations projecting Unified Mechanics' three-channel recursion onto different
physical substrates. Every figure in this repo is generated from one equation: `φ² = φ + 1`.

```python
phi = (1 + 5**0.5) / 2
r   = 1 / (2 * phi)          # 0.3090...

light    = (1 - r)**2         # 0.4775 — propagation
boundary = 2 * r * (1 - r)   # 0.4271 — cross-term, interface
matter   = r**2               # 0.0955 — accumulation
```

The boundary channel is the cross-term. It cannot be seeded — it emerges at the
interface between the light and matter structures whenever both are present. These
simulations show that across five structurally unrelated substrates, the same
three-channel geometry appears.

---

## Simulations

| # | File | Substrate | What you see |
|---|------|-----------|-------------|
| 1 | `01_gray_scott.py` | Reaction-diffusion PDE | Three Turing morphologies in one field — coral (L), labyrinth (B), spots (M) |
| 2 | `02_attractors.py` | Clifford strange attractors | Three phase geometries, one per channel — open basin, tight interface, dense core |
| 3 | `03_hypernetx_voids.py` | Hypergraph topology | Channel-coloured hyperedges, connected component tracking (topological voids) |
| 4 | `04_dla.py` | Diffusion-limited aggregation | Single fractal object, three branch textures by sticking-probability zone |
| 5 | `05_hypergraph_causal.py` | Wolfram hypergraph rewriting | Causal event graph coloured by channel — boundary events visible at the interface |

---

## Run all

```bash
pip install -r requirements.txt
python3 01_gray_scott.py
python3 02_attractors.py
python3 03_hypernetx_voids.py
python3 04_dla.py
python3 05_hypergraph_causal.py
# figures saved to figures/
```

---

## The key result

Sim 5 is the direct numerical test. Seed a Wolfram hypergraph with light and matter
structures at their exact UM weights. Run the Wolfram standard rule. The boundary
channel emerges at the interface:

```
B emerged = 0.328   target = 0.4271
```

The gap closes as simulation scale increases. The Wolfram standard rule is uniquely
the closest rule to the UM target across a 12-rule sweep — it is also the rule known
to generate ~3D spatial topology. These are the same fact.

---

## Related

- [boundary-channel-planck](https://github.com/joseph-shields/boundary-channel-planck) — the full derivation paper
- [unified-mechanics](https://github.com/joseph-shields/unified-mechanics) — the UM series
