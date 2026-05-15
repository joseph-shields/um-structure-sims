"""
UM Wolfram-Style Hypergraph Render
3D interactive visualization of the growing spatial hypergraph,
styled like the Wolfram Physics Project's own renders.

  Nodes:      spheres, colored by generation (blue → red)
  2-edges:    thin lines
  3-hyperedges: filled semi-transparent triangles
  Channel tag: node border color L=cyan / B=gold / M=red

Outputs both an interactive HTML (open in browser) and a static PNG.
"""

import math, random, itertools
from collections import defaultdict
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

random.seed(42)
np.random.seed(42)

phi = (1 + 5**0.5) / 2
r_  = 1 / (2 * phi)
W_L = (1 - r_)**2
W_B = 2 * r_ * (1 - r_)
W_M = r_**2

# ---------------------------------------------------------------
# Build hypergraph via Wolfram standard rule
# ---------------------------------------------------------------
def build_seed(n_light=14, n_matter_nodes=4):
    graph = []
    light_nodes  = set()
    matter_nodes = set()
    for i in range(n_light):
        e = frozenset({i, i+1, i+2})
        graph.append(e); light_nodes.update(e)
    interface = n_light + 1
    light_nodes.add(interface)
    hub  = interface
    sats = list(range(n_light+2, n_light+2+n_matter_nodes-1))
    matter_nodes.update([hub]+sats)
    for a,b in itertools.combinations(sats,2):
        graph.append(frozenset({hub,a,b}))
    for i in range(len(sats)-1):
        graph.append(frozenset({hub,sats[i],sats[i+1]}))
    nc = [n_light+2+n_matter_nodes]
    return graph, light_nodes, matter_nodes, nc

def classify(e, L, M):
    il = all(n in L for n in e)
    im = all(n in M for n in e)
    if il and im: return 'B'
    if il: return 'L'
    if im: return 'M'
    return 'B'

def step_wolfram(graph, nc, L, M):
    """Cross-mixing rule with new node: {b,c,d},{c,a,e},{d,b,f} — grows the hypergraph."""
    node_edges = defaultdict(list)
    for i,e in enumerate(graph):
        if len(e)==3:
            for n in e: node_edges[n].append(i)
    for i1,e1 in enumerate(graph):
        if len(e1)!=3: continue
        for sh in list(e1):
            for i2 in node_edges[sh]:
                if i2<=i1: continue
                e2=graph[i2]
                if len(e2)!=3: continue
                s=e1&e2
                if len(s)!=1: continue
                a=list(s)[0]
                bc=list(e1-s); b,c=bc[0],bc[1]
                de=list(e2-s); d,e_=de[0],de[1]
                f=nc[0]; nc[0]+=1
                t1=classify(e1,L,M); t2=classify(e2,L,M)
                ch='B' if (t1=='B' or t2=='B' or t1!=t2) else t1
                if ch=='L': L.add(f)
                elif ch=='M': M.add(f)
                # Cross-mixing rule with new node
                produced=[frozenset({b,c,d}),frozenset({c,a,e_}),frozenset({d,b,f})]
                produced=[e for e in produced if len(e)==3]
                ng=[e for i,e in enumerate(graph) if i not in (i1,i2)]
                ng.extend(produced)
                return ng,nc,ch,(e1,e2),produced
    return None,nc,None,None,None

N_STEPS = 300
graph,L,M,nc = build_seed(14,4)

all_edges    = []   # list of frozensets currently in graph
edge_gen     = {}   # frozenset → step it was created
node_gen     = {}   # node → step first seen
node_channel = {}   # node → channel of event that created it

# Record initial edges
for e in graph:
    edge_gen[e] = 0
    for n in e:
        node_gen[n] = 0
        ch = classify(frozenset({n}), L, M)
        node_channel[n] = ch

print(f"Running {N_STEPS} steps...")
for step in range(1, N_STEPS+1):
    graph,nc,ch,consumed,produced = step_wolfram(graph,nc,L,M)
    if graph is None: break
    for e in (produced or []):
        edge_gen[e] = step
        for n in e:
            if n not in node_gen:
                node_gen[n] = step
                node_channel[n] = ch or 'B'
    if step % 20 == 0:
        print(f"  step {step}  nodes={len(node_gen)}  edges={len(graph)}")

# Final hyperedges
final_edges = list(graph)
all_nodes   = sorted(node_gen.keys())
max_gen     = max(node_gen.values()) if node_gen else 1
print(f"Final: {len(all_nodes)} nodes, {len(final_edges)} hyperedges")

# ---------------------------------------------------------------
# 3D layout — spring layout on 1-skeleton
# ---------------------------------------------------------------
G1 = nx.Graph()
G1.add_nodes_from(all_nodes)
for e in final_edges:
    lst = list(e)
    for u,v in itertools.combinations(lst,2):
        G1.add_edge(u,v)

# 3D spring layout via iterative force-directed
print("Computing 3D layout...")
pos2d = nx.spring_layout(G1, seed=42, k=0.5, iterations=80)
# Lift into 3D by adding z = sin of angle from centre
pos3d = {}
for n,(x,y) in pos2d.items():
    z = math.sin(math.atan2(y,x)*3 + node_gen.get(n,0)*0.05)
    pos3d[n] = np.array([x, y, z*0.4])

# ---------------------------------------------------------------
# Colour maps
# ---------------------------------------------------------------
CH_COLORS = {'L':'#4fc3f7','B':'#ffd54f','M':'#ef5350'}

def gen_color(g, max_g):
    """Blue (early) → gold (mid) → red (late)"""
    t = g / max(max_g, 1)
    if t < 0.5:
        t2 = t*2
        r,gr,b = int(30+t2*220), int(120+t2*80), int(240-t2*200)
    else:
        t2 = (t-0.5)*2
        r,gr,b = int(250), int(200-t2*160), int(40-t2*30)
    return f'rgb({r},{gr},{b})'

# ---------------------------------------------------------------
# PLOTLY interactive figure
# ---------------------------------------------------------------
print("Building Plotly figure...")
traces = []

# Hyperedge triangles (3-body)
tri_nodes = [e for e in final_edges if len(e)==3]
for e in tri_nodes:
    lst = list(e)
    pts = [pos3d[n] for n in lst]
    mid_gen = np.mean([node_gen.get(n,0) for n in lst])
    col = gen_color(mid_gen, max_gen)
    # Draw as filled triangle: 3 vertices + close
    xs = [p[0] for p in pts] + [pts[0][0]]
    ys = [p[1] for p in pts] + [pts[0][1]]
    zs = [p[2] for p in pts] + [pts[0][2]]
    traces.append(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines',
        line=dict(color=col, width=1),
        opacity=0.3,
        showlegend=False,
        hoverinfo='skip',
    ))

# Edges (1-skeleton lines)
for u,v in G1.edges():
    pu,pv = pos3d[u], pos3d[v]
    g_avg = (node_gen.get(u,0)+node_gen.get(v,0))/2
    col   = gen_color(g_avg, max_gen)
    traces.append(go.Scatter3d(
        x=[pu[0],pv[0],None], y=[pu[1],pv[1],None], z=[pu[2],pv[2],None],
        mode='lines',
        line=dict(color=col, width=1.5),
        opacity=0.4,
        showlegend=False,
        hoverinfo='skip',
    ))

# Nodes
node_xs = [pos3d[n][0] for n in all_nodes]
node_ys = [pos3d[n][1] for n in all_nodes]
node_zs = [pos3d[n][2] for n in all_nodes]
node_cols = [CH_COLORS.get(node_channel.get(n,'B'),'#ffd54f') for n in all_nodes]
node_gens  = [node_gen.get(n,0) for n in all_nodes]
gen_cols   = [gen_color(g,max_gen) for g in node_gens]

traces.append(go.Scatter3d(
    x=node_xs, y=node_ys, z=node_zs,
    mode='markers',
    marker=dict(
        size=4,
        color=gen_cols,
        line=dict(color=node_cols, width=1.5),
        opacity=0.9,
    ),
    text=[f"node {n}  gen={node_gen.get(n,0)}  ch={node_channel.get(n,'?')}"
          for n in all_nodes],
    hoverinfo='text',
    showlegend=False,
))

# Legend traces
for ch, col, lbl in [('L','#4fc3f7',f'Light (1-r)²={W_L:.4f}'),
                      ('B','#ffd54f',f'Boundary 2r(1-r)={W_B:.4f}'),
                      ('M','#ef5350',f'Matter r²={W_M:.4f}')]:
    traces.append(go.Scatter3d(
        x=[None],y=[None],z=[None],
        mode='markers',
        marker=dict(size=8,color=col),
        name=lbl,
    ))

fig_plotly = go.Figure(data=traces)
fig_plotly.update_layout(
    title=dict(
        text=f'UM Wolfram Hypergraph  |  c²=c+1  |  {len(all_nodes)} nodes  {len(final_edges)} hyperedges<br>'
             f'<span style="font-size:12px">Blue=early generation → Red=late  |  Node border = channel  |  Wolfram standard rule</span>',
        font=dict(color='white', size=14),
        x=0.5,
    ),
    scene=dict(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
        zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
        bgcolor='#050510',
    ),
    paper_bgcolor='#050510',
    plot_bgcolor='#050510',
    font=dict(color='white'),
    legend=dict(
        bgcolor='rgba(10,10,30,0.8)',
        bordercolor='#444',
        font=dict(color='white'),
    ),
    margin=dict(l=0,r=0,t=80,b=0),
)

html_out = 'figures/06_wolfram_render.html'
fig_plotly.write_html(html_out)
print(f"Saved interactive: {html_out}")

# Static PNG via plotly
png_out = 'figures/06_wolfram_render.png'
try:
    fig_plotly.write_image(png_out, width=1600, height=900, scale=1.5)
    print(f"Saved static:      {png_out}")
except Exception:
    # Fallback: matplotlib 3D
    print("Plotly image export unavailable — rendering with matplotlib...")
    fig3d = plt.figure(figsize=(16, 10))
    fig3d.patch.set_facecolor('#050510')
    ax3d  = fig3d.add_subplot(111, projection='3d')
    ax3d.set_facecolor('#050510')

    # Draw 1-skeleton edges
    for u,v in G1.edges():
        pu,pv = pos3d[u],pos3d[v]
        g_avg = (node_gen.get(u,0)+node_gen.get(v,0))/2
        t = g_avg/max(max_gen,1)
        col = (t*0.9, 0.3+t*0.4, 1-t*0.9)
        ax3d.plot([pu[0],pv[0]],[pu[1],pv[1]],[pu[2],pv[2]],
                  color=col, lw=0.5, alpha=0.25)

    # Draw hyperedge triangles
    tris  = []
    tcols = []
    for e in tri_nodes:
        lst = list(e)
        pts = np.array([pos3d[n] for n in lst])
        tris.append(pts)
        mid_gen = np.mean([node_gen.get(n,0) for n in lst])
        t = mid_gen/max(max_gen,1)
        tcols.append((t*0.9, 0.3+t*0.4, 1-t*0.9, 0.08))

    if tris:
        poly = Poly3DCollection(tris, alpha=0.08)
        poly.set_facecolor(tcols)
        poly.set_edgecolor('none')
        ax3d.add_collection3d(poly)

    # Draw nodes
    for n in all_nodes:
        p  = pos3d[n]
        g  = node_gen.get(n,0)
        t  = g/max(max_gen,1)
        nc = (t*0.9, 0.3+t*0.4, 1-t*0.9)
        ec = CH_COLORS.get(node_channel.get(n,'B'),'#ffd54f')
        ax3d.scatter(*p, s=18, color=[nc], edgecolors=ec, linewidths=0.6, alpha=0.9)

    ax3d.set_axis_off()
    ax3d.set_title(
        f'UM Wolfram Hypergraph  |  {len(all_nodes)} nodes  |  Wolfram standard rule\n'
        f'Node border = channel (cyan=L, gold=B, red=M)  |  Color = generation',
        color='white', fontsize=11, pad=10)

    plt.tight_layout()
    plt.savefig(png_out, dpi=150, bbox_inches='tight', facecolor='#050510')
    print(f"Saved static: {png_out}")
