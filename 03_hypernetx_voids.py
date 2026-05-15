"""
UM HyperNetX Void Topology
Build a hypergraph where hyperedges are tagged by channel.
Track topological voids (connected components, cycles) per channel
as the hypergraph grows — visualise how the three channels
occupy structurally distinct topological roles.

Light    → many small open edges  (sparse, void-generating)
Boundary → medium edges spanning between L and M nodes
Matter   → dense cliques (void-filling, cycle-closing)
"""

import math, random, itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import hypernetx as hnx

random.seed(42)
np.random.seed(42)

phi = (1 + 5**0.5) / 2
r   = 1 / (2 * phi)
W_L = (1 - r)**2
W_B = 2 * r * (1 - r)
W_M = r**2

print(f"UM channels:  L={W_L:.4f}  B={W_B:.4f}  M={W_M:.4f}")

# ---------------------------------------------------------------
# Build a growing hypergraph and track Betti numbers per channel
# ---------------------------------------------------------------
N_TOTAL = 60           # total nodes
N_STEPS = 80           # growth steps

# Node pools
n_L = round(N_TOTAL * W_L)   # ~29 light nodes
n_B = round(N_TOTAL * W_B)   # ~26 boundary nodes
n_M = N_TOTAL - n_L - n_B    # ~5  matter nodes

light_nodes    = list(range(0,              n_L))
boundary_nodes = list(range(n_L,            n_L + n_B))
matter_nodes   = list(range(n_L + n_B,     N_TOTAL))

print(f"Node pools:   L={n_L}  B={n_B}  M={n_M}")

COLORS = {'L': '#4fc3f7', 'B': '#ffd54f', 'M': '#ef5350'}

def random_edge(channel, size=None):
    """Return a hyperedge (frozenset) sampled from the given channel's node pool."""
    if channel == 'L':
        pool = light_nodes;    sz = size or random.choice([2, 2, 3])
    elif channel == 'M':
        pool = matter_nodes;   sz = size or random.choice([2, 3, 3, 4])
    else:
        # Boundary: mix light and matter nodes
        pool = light_nodes + matter_nodes; sz = size or 3
        nodes = random.sample(pool, min(sz, len(pool)))
        # Force at least one from each side if possible
        if len(light_nodes) > 0 and len(matter_nodes) > 0:
            nodes = random.sample(light_nodes, 1) + random.sample(matter_nodes, 1)
            extra = random.sample(pool, max(0, sz - 2))
            nodes = list(set(nodes + extra))[:sz]
        return frozenset(nodes)
    return frozenset(random.sample(pool, min(sz, len(pool))))

# Grow hypergraph step by step
edges = {}          # id -> (frozenset, channel)
edge_counter = [0]
history = {'step': [], 'L_edges': [], 'B_edges': [], 'M_edges': [],
           'L_comp': [], 'B_comp': [], 'M_comp': []}

def add_edge(ch):
    e = random_edge(ch)
    if len(e) < 2: return
    eid = f"{ch}{edge_counter[0]}"
    edge_counter[0] += 1
    edges[eid] = (e, ch)

def count_components(channel):
    """Connected components in the subgraph of edges belonging to channel."""
    G = nx.Graph()
    for eid, (e, ch) in edges.items():
        if ch == channel:
            G.add_nodes_from(e)
            for u, v in itertools.combinations(e, 2):
                G.add_edge(u, v)
    if G.number_of_nodes() == 0:
        return 0
    return nx.number_connected_components(G)

# Seed with a few edges per channel
for _ in range(3): add_edge('L')
for _ in range(2): add_edge('B')
for _ in range(1): add_edge('M')

for step in range(N_STEPS):
    # Add edges proportional to channel weights (normalised to ~3 per step)
    for _ in range(2): add_edge('L')
    add_edge('B')
    if random.random() < W_M * 5: add_edge('M')

    l_e = sum(1 for _, (_, c) in edges.items() if c == 'L')
    b_e = sum(1 for _, (_, c) in edges.items() if c == 'B')
    m_e = sum(1 for _, (_, c) in edges.items() if c == 'M')

    history['step'].append(step)
    history['L_edges'].append(l_e)
    history['B_edges'].append(b_e)
    history['M_edges'].append(m_e)
    history['L_comp'].append(count_components('L'))
    history['B_comp'].append(count_components('B'))
    history['M_comp'].append(count_components('M'))

# Final hypergraph (networkx 1-skeleton)
node_colors = {}
for n in range(N_TOTAL):
    if n in light_nodes:    node_colors[n] = '#4fc3f7'
    elif n in matter_nodes: node_colors[n] = '#ef5350'
    else:                   node_colors[n] = '#ffd54f'

# ---------------------------------------------------------------
# Figure
# ---------------------------------------------------------------
fig = plt.figure(figsize=(18, 8))
fig.patch.set_facecolor('#0a0a0a')

# Panel 1 & 2: hypergraph drawing
ax_hg = fig.add_subplot(1, 3, 1)
ax_hg.set_facecolor('#0a0a0a')
ax_hg.set_title("UM Hypergraph\n(channel-coloured edges)", color='white', fontsize=11)

# Draw using networkx spring layout on the 1-skeleton
G1 = nx.Graph()
for eid, (e, ch) in edges.items():
    G1.add_nodes_from(e)
    for u, v in itertools.combinations(e, 2):
        G1.add_edge(u, v, channel=ch)

pos = nx.spring_layout(G1, seed=42, k=0.4)
node_c = [node_colors.get(n, '#aaaaaa') for n in G1.nodes()]

for u, v, data in G1.edges(data=True):
    ch = data.get('channel', 'B')
    ax_hg.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
               color=COLORS[ch], alpha=0.35, lw=0.8)

nx.draw_networkx_nodes(G1, pos, ax=ax_hg, node_size=25,
                       node_color=node_c, alpha=0.9)
ax_hg.axis('off')
patches = [mpatches.Patch(color=COLORS[c], label=f'{c}-channel') for c in 'LBM']
ax_hg.legend(handles=patches, loc='lower left', facecolor='#1a1a1a',
             labelcolor='white', fontsize=8)

# Panel 2: edge count over time
ax_e = fig.add_subplot(1, 3, 2)
ax_e.set_facecolor('#0a0a0a')
ax_e.set_title("Edge accumulation per channel", color='white', fontsize=11)
steps_arr = history['step']
ax_e.plot(steps_arr, history['L_edges'], color='#4fc3f7', lw=2, label='Light L')
ax_e.plot(steps_arr, history['B_edges'], color='#ffd54f', lw=2, label='Boundary B')
ax_e.plot(steps_arr, history['M_edges'], color='#ef5350', lw=2, label='Matter M')

final_total = history['L_edges'][-1] + history['B_edges'][-1] + history['M_edges'][-1]
for ch, key, col in [('L','L_edges','#4fc3f7'),('B','B_edges','#ffd54f'),('M','M_edges','#ef5350')]:
    final = history[key][-1]
    frac  = final / final_total
    ax_e.text(steps_arr[-1]*1.01, final, f'{frac:.3f}', color=col, fontsize=8, va='center')

for w, col, lbl in [(W_L,'#4fc3f7','L target'),(W_B,'#ffd54f','B target'),(W_M,'#ef5350','M target')]:
    ax_e.axhline(w * final_total, color=col, lw=0.8, linestyle=':', alpha=0.5)

ax_e.tick_params(colors='white'); ax_e.spines[:].set_color('#333')
ax_e.set_xlabel('Step', color='white'); ax_e.set_ylabel('Edge count', color='white')
ax_e.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=8)

# Panel 3: connected components (topological voids proxy)
ax_c = fig.add_subplot(1, 3, 3)
ax_c.set_facecolor('#0a0a0a')
ax_c.set_title("Connected components\n(topological void proxy)", color='white', fontsize=11)
ax_c.plot(steps_arr, history['L_comp'], color='#4fc3f7', lw=2, label='Light L')
ax_c.plot(steps_arr, history['B_comp'], color='#ffd54f', lw=2, label='Boundary B')
ax_c.plot(steps_arr, history['M_comp'], color='#ef5350', lw=2, label='Matter M')
ax_c.tick_params(colors='white'); ax_c.spines[:].set_color('#333')
ax_c.set_xlabel('Step', color='white'); ax_c.set_ylabel('# components', color='white')
ax_c.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=8)
ax_c.annotate('More components\n= more voids', xy=(0.6, 0.85), xycoords='axes fraction',
              color='#aaaaaa', fontsize=8, ha='center')

fig.suptitle("UM Hypergraph Topology  |  c²=c+1  |  Void structure by channel",
             color='white', fontsize=13, y=1.01)
plt.tight_layout(pad=1.5)

out = 'figures/03_hypernetx_voids.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
print(f"Saved: {out}")
