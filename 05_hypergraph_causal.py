"""
UM Hypergraph Causal Growth
Our own hypergraph rewriting engine (Wolfram standard rule) seeded with
UM channel weights. Visualises the causal graph as it grows, colouring
each event node by the channel type of the rewriting event (L/B/M).

This is the direct visualisation of the channel emergence experiment:
you can see the boundary channel growing at the interface between the
light and matter regions in real time.
"""

import math, random, itertools
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

random.seed(42)

phi = (1 + 5**0.5) / 2
r_  = 1 / (2 * phi)
W_L = (1 - r_)**2
W_B = 2 * r_ * (1 - r_)
W_M = r_**2

COLORS = {'L': '#4fc3f7', 'B': '#ffd54f', 'M': '#ef5350', None: '#555555'}

# ---------------------------------------------------------------
# Seed: light chain + matter hub (from channel_emergence.py)
# ---------------------------------------------------------------
def build_seed(n_light=19, n_matter_nodes=5):
    graph = []
    light_nodes  = set()
    matter_nodes = set()
    for i in range(n_light):
        e = frozenset({i, i+1, i+2})
        graph.append(e); light_nodes.update(e)
    interface_node = n_light + 1
    light_nodes.add(interface_node)
    hub  = interface_node
    sats = list(range(n_light + 2, n_light + 2 + n_matter_nodes - 1))
    matter_nodes.update([hub] + sats)
    for a, b in itertools.combinations(sats, 2):
        graph.append(frozenset({hub, a, b}))
    for i in range(len(sats)-1):
        graph.append(frozenset({hub, sats[i], sats[i+1]}))
    nc = [n_light + 2 + n_matter_nodes]
    return graph, light_nodes, matter_nodes, interface_node, nc

def classify_edge(edge, L, M):
    il = all(n in L for n in edge)
    im = all(n in M for n in edge)
    if il and im: return 'B'
    if il: return 'L'
    if im: return 'M'
    return 'B'

def apply_wolfram(graph, nc, L, M):
    """Wolfram standard rule: {a,b,c},{a,d,e} → {b,c,d},{c,a,e},{d,b,e}"""
    node_edges = defaultdict(list)
    for i, e in enumerate(graph):
        if len(e) == 3:
            for n in e: node_edges[n].append(i)
    for i1, e1 in enumerate(graph):
        if len(e1) != 3: continue
        for shared in list(e1):
            for i2 in node_edges[shared]:
                if i2 <= i1: continue
                e2 = graph[i2]
                if len(e2) != 3: continue
                s = e1 & e2
                if len(s) != 1: continue
                a  = list(s)[0]
                bc = list(e1-s); b,c = bc[0],bc[1]
                de = list(e2-s); d,e_ = de[0],de[1]

                t1 = classify_edge(e1, L, M)
                t2 = classify_edge(e2, L, M)
                etype = 'B' if (t1 == 'B' or t2 == 'B' or t1 != t2) else t1

                # Wolfram standard (no new node): {b,c,d},{c,a,e},{d,b,e}
                produced = [frozenset({b,c,d}), frozenset({c,a,e_}), frozenset({d,b,e_})]
                produced = [e for e in produced if len(e)==3]

                new_graph = [e for i,e in enumerate(graph) if i not in (i1,i2)]
                new_graph.extend(produced)
                return new_graph, nc, etype, (e1,e2), produced
    return None, nc, None, None, None

# ---------------------------------------------------------------
# Run and record causal graph
# ---------------------------------------------------------------
N_STEPS = 250

graph, L, M, inode, nc = build_seed(19, 5)
event_channel   = []    # channel type of each event
edge_produced_by = {}   # frozenset → event index
causal_parents   = defaultdict(set)

for step in range(N_STEPS):
    graph, nc, etype, consumed, produced = apply_wolfram(graph, nc, L, M)
    if graph is None: break

    for e in (consumed or []):
        if e in edge_produced_by:
            causal_parents[step].add(edge_produced_by[e])
    for e in (produced or []):
        edge_produced_by[e] = step

    event_channel.append(etype)

n_events = len(event_channel)
counts   = defaultdict(int)
for ch in event_channel: counts[ch] += 1
total = sum(counts.values())
print(f"Events: {n_events}   "
      f"L={counts['L']/total:.3f}  B={counts['B']/total:.3f}  M={counts['M']/total:.3f}")
print(f"UM target:        L={W_L:.3f}  B={W_B:.3f}  M={W_M:.3f}")

# ---------------------------------------------------------------
# Build causal DAG for visualisation
# ---------------------------------------------------------------
G_causal = nx.DiGraph()
for i in range(n_events):
    G_causal.add_node(i, channel=event_channel[i])
for child, parents in causal_parents.items():
    for p in parents:
        G_causal.add_edge(p, child)

# Layered layout: x = step, y = jitter by channel
pos = {}
channel_y = {'L': 0.8, 'B': 0.0, 'M': -0.8, None: 0.0}
for i in range(n_events):
    ch = event_channel[i]
    jitter = random.uniform(-0.25, 0.25)
    pos[i] = (i * 0.4, channel_y.get(ch, 0) + jitter)

node_colors = [COLORS.get(event_channel[i], '#555') for i in range(n_events)]
node_sizes  = [20 if event_channel[i] == 'B' else 12 for i in range(n_events)]

# ---------------------------------------------------------------
# Figure
# ---------------------------------------------------------------
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('#080808')

# Top panel: causal DAG
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_facecolor('#080808')
ax1.set_title("Causal Event Graph — Wolfram standard rule, UM seed",
              color='white', fontsize=12, pad=8)

nx.draw_networkx_nodes(G_causal, pos, ax=ax1, node_size=node_sizes,
                       node_color=node_colors, alpha=0.85)
nx.draw_networkx_edges(G_causal, pos, ax=ax1, width=0.4, alpha=0.2,
                       edge_color='#888888', arrows=False)

for label, y, col in [('Light L', 0.8, '#4fc3f7'),
                       ('Boundary B', 0.0, '#ffd54f'),
                       ('Matter M', -0.8, '#ef5350')]:
    ax1.axhline(y, color=col, lw=0.6, alpha=0.25, linestyle=':')
    ax1.text(-2, y, label, color=col, fontsize=8, va='center')

ax1.axis('off')
ax1.set_xlim(-5, n_events * 0.4 + 2)
ax1.set_ylim(-1.4, 1.4)

patches = [mpatches.Patch(color=COLORS[c], label=f'{c}-channel events') for c in 'LBM']
ax1.legend(handles=patches, loc='upper right', facecolor='#111',
           labelcolor='white', fontsize=9)

# Bottom panel: running channel fractions
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_facecolor('#080808')
ax2.set_title("Channel fractions over time", color='white', fontsize=12, pad=8)

run_L, run_B, run_M = [], [], []
rl, rb, rm = 0, 0, 0
for ch in event_channel:
    if ch == 'L': rl += 1
    elif ch == 'B': rb += 1
    elif ch == 'M': rm += 1
    tot = rl + rb + rm
    run_L.append(rl/tot); run_B.append(rb/tot); run_M.append(rm/tot)

steps_range = range(len(run_B))
ax2.plot(steps_range, run_L, color='#4fc3f7', lw=1.8, label='Light L')
ax2.plot(steps_range, run_B, color='#ffd54f', lw=2.2, label='Boundary B')
ax2.plot(steps_range, run_M, color='#ef5350', lw=1.8, label='Matter M')

for w, col, lbl in [(W_L,'#4fc3f7',f'L target {W_L:.4f}'),
                    (W_B,'#ffd54f',f'B target {W_B:.4f}'),
                    (W_M,'#ef5350',f'M target {W_M:.4f}')]:
    ax2.axhline(w, color=col, lw=1, linestyle='--', alpha=0.5, label=lbl)

if run_B:
    final_B = run_B[-1]
    ax2.text(len(run_B)*0.7, final_B+0.02,
             f'B = {final_B:.4f}', color='#ffd54f', fontsize=10)

ax2.set_xlabel('Event step', color='white', fontsize=10)
ax2.set_ylabel('Fraction of events', color='white', fontsize=10)
ax2.set_ylim(0, 1)
ax2.tick_params(colors='white')
ax2.spines[:].set_color('#333')
ax2.legend(facecolor='#111', labelcolor='white', fontsize=8,
           loc='upper right', ncol=2)

fig.suptitle(
    f"UM Hypergraph Causal Growth  |  c²=c+1  |  r={r_:.4f}  |  "
    f"B emerged={run_B[-1]:.4f}  target={W_B:.4f}",
    color='white', fontsize=13, y=1.01
)
plt.tight_layout(pad=1.5)

out = 'figures/05_hypergraph_causal.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#080808')
print(f"Saved: {out}")
