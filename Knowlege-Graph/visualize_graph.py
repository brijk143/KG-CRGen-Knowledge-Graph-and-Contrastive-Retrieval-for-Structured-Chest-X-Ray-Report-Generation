import json
from collections import defaultdict

GRAPH_JSON  = '/Users/bkishor/Desktop/kg_new/Knowlege-Graph/output/graph_adj.json'
OUTPUT_HTML = '/Users/bkishor/Desktop/kg_new/Knowlege-Graph/knowledge_graph_interactive.html'

RELATION_COLORS = {
    'located_in':        '#e74c3c',
    'suggests':          '#2ecc71',
    'is_symptom_of':     '#3498db',
    'is_finding_of':     '#f39c12',
    'associated_with':   '#9b59b6',
    'contradicts':       '#e91e63',
    'strongly_suggests': '#00bcd4',
    'requires':          '#ff9800',
    'confirmed_by':      '#8bc34a',
    'absence_weakens':   '#607d8b',
    'weakly_suggests':   '#795548',
}

NODE_KEYWORDS = {
    'lobe':         '#1abc9c',
    'pneumonia':    '#e74c3c',
    'effusion':     '#3498db',
    'opacity':      '#f39c12',
    'nodule':       '#9b59b6',
    'fracture':     '#e67e22',
    'cardiac':      '#e91e63',
    'pleural':      '#00bcd4',
    'cardiomegaly': '#ff5722',
}

def node_color(name):
    nl = name.lower()
    for kw, col in NODE_KEYWORDS.items():
        if kw in nl:
            return col
    return '#b0bec5'

# Load graph
with open(GRAPH_JSON) as f:
    graph_data = json.load(f)

nodes_dict = {}
edges_list = []
edge_info  = defaultdict(int)

for source, relations in graph_data.items():
    nodes_dict.setdefault(source, 0)
    if not relations:
        continue
    for pair in relations:
        if len(pair) == 2:
            rel, target = pair
            nodes_dict.setdefault(target, 0)
            nodes_dict[source] = nodes_dict.get(source, 0) + 1
            nodes_dict[target] = nodes_dict.get(target, 0) + 1
            edges_list.append((source, rel, target))
            edge_info[rel] += 1

max_deg = max(nodes_dict.values()) if nodes_dict else 1

# Build JS arrays — escape quotes so JSON stays valid
def esc(s):
    return s.replace('\\', '\\\\').replace('"', '\\"')

js_nodes = []
for nid, deg in nodes_dict.items():
    size = 8 + 22 * (deg / max_deg)
    col  = node_color(nid)
    e    = esc(nid)
    js_nodes.append(
        f'{{"id":"{e}","label":"{e}","size":{size:.1f},'
        f'"color":{{"background":"{col}","border":"#fff",'
        f'"highlight":{{"background":"#f1c40f","border":"#fff"}},'
        f'"hover":{{"background":"#f1c40f","border":"#fff"}}}},'
        f'"font":{{"color":"#fff","size":11,"strokeWidth":2,"strokeColor":"#000"}},'
        f'"title":"<b>{e}</b><br>Degree: {deg}"}}'
    )

js_edges = []
for i, (src, rel, tgt) in enumerate(edges_list):
    col = RELATION_COLORS.get(rel, '#90a4ae')
    s, t, r = esc(src), esc(tgt), esc(rel)
    js_edges.append(
        f'{{"id":{i},"from":"{s}","to":"{t}",'
        f'"label":"{r}","title":"<b>{r}</b>",'
        f'"color":{{"color":"{col}","highlight":"#f1c40f","hover":"#f1c40f"}},'
        f'"font":{{"size":9,"color":"{col}","strokeWidth":0}},'
        f'"arrows":"to","width":1.5,'
        f'"smooth":{{"type":"curvedCW","roundness":0.15}}}}'
    )

legend_edges = ''.join(
    f'<div class="row"><span class="line" style="background:{c}"></span>{r}</div>'
    for r, c in RELATION_COLORS.items()
)
legend_nodes = ''.join(
    f'<div class="row"><span class="dot" style="background:{c}"></span>{k}</div>'
    for k, c in NODE_KEYWORDS.items()
)

nodes_js = ',\n'.join(js_nodes)
edges_js = ',\n'.join(js_edges)

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Medical Knowledge Graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:#1a1a2e; font-family:'Segoe UI',sans-serif; overflow:hidden; }}
  #graph {{ width:100vw; height:100vh; }}
  #panel {{
    position:fixed; top:12px; left:12px; z-index:999;
    background:rgba(15,15,35,0.93); border-radius:10px;
    padding:12px 16px; color:#fff; width:220px;
    box-shadow:0 4px 20px rgba(0,0,0,0.6);
    max-height:95vh; overflow-y:auto;
  }}
  #panel h3 {{ font-size:13px; letter-spacing:1px; text-transform:uppercase; color:#f1c40f; margin-bottom:8px; }}
  .section {{ font-size:10px; color:#aaa; margin:8px 0 4px; text-transform:uppercase; }}
  .row {{ display:flex; align-items:center; font-size:11px; margin:3px 0; }}
  .dot  {{ width:12px; height:12px; border-radius:50%; margin-right:7px; flex-shrink:0; }}
  .line {{ width:20px; height:3px; border-radius:2px; margin-right:7px; flex-shrink:0; }}
  #stats {{ margin-top:8px; padding-top:8px; border-top:1px solid #333; font-size:11px; color:#ccc; }}
  #search {{ margin-top:8px; padding-top:8px; border-top:1px solid #333; }}
  #search input {{ width:100%; padding:5px 8px; border-radius:5px; border:1px solid #444; background:#2a2a4a; color:#fff; font-size:11px; }}
  #search input::placeholder {{ color:#666; }}
  #sres {{ font-size:10px; color:#f1c40f; margin-top:3px; min-height:14px; }}
  #physBtn {{ margin-top:8px; padding-top:8px; border-top:1px solid #333; }}
  #physBtn button {{ width:100%; padding:5px; border-radius:5px; border:none; background:#2a2a4a; color:#fff; font-size:11px; cursor:pointer; }}
  #physBtn button:hover {{ background:#3a3a6a; }}
</style>
</head>
<body>
<div id="graph"></div>
<div id="panel">
  <h3>🧠 Medical KG</h3>
  <div id="stats">
    <div>📌 Nodes: <b>{len(nodes_dict)}</b></div>
    <div>🔗 Edges: <b>{len(edges_list)}</b></div>
  </div>
  <div id="search">
    <input type="text" placeholder="🔍 Search node…" oninput="doSearch(this.value)">
    <div id="sres"></div>
  </div>
  <div id="physBtn"><button onclick="togglePhys()">⚡ Toggle Physics</button></div>
  <div class="section">Edge Types</div>
  {legend_edges}
  <div class="section">Node Categories</div>
  {legend_nodes}
</div>
<script>
var nodes = new vis.DataSet([{nodes_js}]);
var edges = new vis.DataSet([{edges_js}]);
var options = {{
  physics: false,
  layout: {{ randomSeed: 42 }},
  interaction: {{ hover:true, tooltipDelay:100, navigationButtons:true, keyboard:true }},
  nodes: {{ shape:'dot', borderWidth:1.5 }},
  edges: {{ selectionWidth:3 }}
}};
var network = new vis.Network(document.getElementById('graph'), {{nodes:nodes, edges:edges}}, options);
var physOn = false;
function togglePhys() {{ physOn = !physOn; network.setOptions({{physics:physOn}}); }}
function doSearch(q) {{
  var el = document.getElementById('sres');
  if (!q) {{ el.textContent=''; network.unselectAll(); return; }}
  var m = nodes.get().find(n => n.id.toLowerCase().includes(q.toLowerCase()));
  if (m) {{
    network.selectNodes([m.id]);
    network.focus(m.id, {{scale:1.8, animation:{{duration:600}}}});
    el.textContent = '✓ ' + m.id;
  }} else {{
    network.unselectAll(); el.textContent = '✗ Not found';
  }}
}}
</script>
</body>
</html>"""

with open(OUTPUT_HTML, 'w') as f:
    f.write(html)

print(f"✓ Saved → {OUTPUT_HTML}")
print(f"  Nodes: {len(nodes_dict)}  |  Edges: {len(edges_list)}")
