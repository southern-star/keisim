"""Build the graph of an OSM box, report the drops by reason and draw kept / dropped edges (scratch tool)."""
import json, math, sys, collections, time, numpy as np, cv2
import keisim.realworld as rw
from keisim.realworld import Graph, road_graph, graph_town, town_config
osm = json.load(open(sys.argv[1])); bbox = tuple(float(v) for v in sys.argv[2:6]); out = sys.argv[6]
log = []
orig_remove = Graph.remove_edge
def rec(self, eid, why=None):
    e = self.edges[eid]
    if why is not None: log.append((why, e["pts"].copy(), e["oneway"]))
    orig_remove(self, eid, why)
Graph.remove_edge = rec
t0 = time.time()
G, proj = road_graph(osm, bbox)
print(f"graph in {time.time()-t0:.1f} s")
km = collections.defaultdict(float)
for why, pts, ow in log:
    km[(why, "oneway" if ow else "twoway")] += float(np.hypot(*np.diff(pts, axis=0).T).sum())
for k, v in sorted(km.items(), key=lambda kv: -kv[1]): print(f"  {k}: {v/1000:.2f} km")
kept = sum(G.length(e) for e in G.edges)
print("kept", round(kept/1000, 2), "km; one-way", round(sum(G.length(e) for e in G.edges if G.edges[e]["oneway"])/1000, 2),
      "nodes", len(G.adj), "signals", len(G.signals))
x0, y0 = proj.xy(bbox[0], bbox[1]); x1, y1 = proj.xy(bbox[2], bbox[3])
sc = 1.2; W, H = int((x1 - x0 + 120) * sc), int((y1 - y0 + 120) * sc)
img = np.full((H, W, 3), 255, np.uint8)
def px(p): return (int((p[0] - x0 + 60) * sc), int(H - (p[1] - y0 + 60) * sc))
colors = {"dead end": (200, 200, 200), "other component": (255, 180, 60), "acute arm": (255, 0, 255), "outside the box": (230, 230, 255),
          "short edge (contracted)": (0, 200, 200), "carriageway (merged)": (0, 128, 0), "self-loop": (0, 0, 0), "duplicate way": (0,0,0)}
for why, pts, ow in log:
    c = colors.get(why, (0, 0, 0))
    cv2.polylines(img, [np.array([px(p) for p in pts], np.int32)], False, c, 2 if ow else 1)
    if ow and why == "dead end":
        cv2.polylines(img, [np.array([px(p) for p in pts], np.int32)], False, (120, 120, 255), 2)
for e in G.edges.values():
    cv2.polylines(img, [np.array([px(p) for p in e["pts"]], np.int32)], False, (0, 0, 255) if e["oneway"] else (0, 0, 0), 2)
    if e["oneway"]:
        p, q = e["pts"][-2], e["pts"][-1]; cv2.arrowedLine(img, px(p), px(q), (0, 0, 255), 2, tipLength=0.6)
for n in G.signals: cv2.circle(img, px(G.nodes[n]), 5, (0, 160, 0), 2)
cv2.rectangle(img, px((x0, y1)), px((x1, y0)), (0, 160, 0), 1)
cv2.imwrite(out, img)
if len(sys.argv) > 7:
    town = graph_town(G, town_config(), name="t")
    print(town.summary(), "max radius", round(max(J.radius for J in town.junctions), 1))
