"""Decode the road LOD3 tiles over a box (Draco via DracoPy) and report the fraction of their vertices inside the
box, the box's share of the 1 km area that has geometry (on a 50 m grid), and the height range.
   dracoenv/bin/python draco_verify.py <ward> S W N E"""
import json, math, struct, sys, urllib.request, urllib.parse, os
import numpy as np, DracoPy
UA = {"User-Agent": "keisim-site-survey/0.1"}
req = urllib.request.Request("https://api.plateauview.mlit.go.jp/datacatalog/plateau-datasets", headers=UA)
d = json.loads(urllib.request.urlopen(req, timeout=90).read())["datasets"]
sets = {(x.get("ward") or x.get("city")): x["url"] for x in d if x.get("pref") == "東京都" and x.get("type_en") == "tran" and str(x.get("lod")) == (sys.argv[6] if len(sys.argv) > 6 else "3") and x.get("year") == 2025 and x.get("url")}
os.makedirs("b3dm", exist_ok=True)
def get(url):
    fn = os.path.join("b3dm", urllib.parse.quote(url, safe=""))[:200]
    if os.path.exists(fn): return open(fn, "rb").read()
    buf = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300).read(); open(fn, "wb").write(buf); return buf
def leaf_tiles(url, depth=0, out=None):
    out = [] if out is None else out
    ts = json.loads(get(url))
    def walk(t, base):
        reg = t.get("boundingVolume", {}).get("region")
        c = t.get("content") or (t.get("contents") or [None])[0]
        if c and c.get("uri", c.get("url", "")):
            uri = urllib.parse.urljoin(base, c.get("uri", c.get("url")))
            if uri.endswith(".json") and depth < 3: leaf_tiles(uri, depth + 1, out)
            elif reg: out.append(([math.degrees(v) for v in reg[:4]], uri, t.get("transform")))
        for ch in t.get("children", []): walk(ch, base)
    walk(ts["root"], url); return out
def ecef_to_llh(xyz):
    a, e2 = 6378137.0, 0.00669437999014
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lon = np.arctan2(y, x); p = np.hypot(x, y); lat = np.arctan2(z, p * (1 - e2))
    for _ in range(5):
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2); h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + h)))
    return np.degrees(lat), np.degrees(lon), h
def positions(buf, transform):
    hl = struct.unpack("<7I", buf[:28]); ftj, ftb, btj, btb = hl[3:7]
    ft = json.loads(buf[28:28 + ftj] or b"{}"); glb = buf[28 + ftj + ftb + btj + btb:]
    jl = struct.unpack("<II", glb[12:20])[0]; gj = json.loads(glb[20:20 + jl]); off = 20 + jl
    bl = struct.unpack("<II", glb[off:off + 8])[0]; bin_ = glb[off + 8:off + 8 + bl]
    rtc = np.array(gj.get("extensions", {}).get("CESIUM_RTC", {}).get("center", ft.get("RTC_CENTER", [0, 0, 0])), float)
    pts = []
    for m in gj.get("meshes", []):
        for p in m.get("primitives", []):
            dr = p.get("extensions", {}).get("KHR_draco_mesh_compression")
            if dr:
                bv = gj["bufferViews"][dr["bufferView"]]
                mesh = DracoPy.decode(bytes(bin_[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]))
                pts.append(np.asarray(mesh.points, float).reshape(-1, 3))
            else:
                acc = gj["accessors"][p["attributes"]["POSITION"]]; bv = gj["bufferViews"][acc["bufferView"]]
                st = bv.get("byteStride", 12); start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
                pts.append(np.frombuffer(bin_, np.float32, count=acc["count"] * (st // 4), offset=start).reshape(-1, st // 4)[:, :3].astype(float))
    P = np.concatenate(pts) if pts else np.zeros((0, 3))
    P = P[:, [0, 2, 1]] * np.array([1, -1, 1]) if len(P) else P        # glTF y-up -> z-up
    if transform:
        M = np.array(transform, float).reshape(4, 4).T; P = (M[:3, :3] @ P.T).T + M[:3, 3]
    else: P = P + rtc
    return P
ward = sys.argv[1]; s, w, n, e = [float(v) for v in sys.argv[2:6]]
tl = leaf_tiles(sets[ward])
hit = [t for t in tl if not (t[0][2] < w or t[0][0] > e or t[0][3] < s or t[0][1] > n)]
allP = []
for reg, uri, tr in hit:
    try: allP.append(positions(get(uri), tr))
    except Exception as ex: print("  tile failed:", uri.split("/")[-1], ex)
P = np.concatenate(allP) if allP else np.zeros((0, 3))
la, lo, h = ecef_to_llh(P) if len(P) else (np.zeros(0),)*3
ins = (la >= s) & (la <= n) & (lo >= w) & (lo <= e)
g = 0.05 / 111.0
cells = set(zip(((la[ins] - s) / g).astype(int), ((lo[ins] - w) / (g / math.cos(math.radians(35.7)))).astype(int)))
ncell = int((n - s) / g) * int((e - w) / (g / math.cos(math.radians(35.7))))
print(f"{ward} box {s},{w},{n},{e}: tiles {len(hit)}, vertices {len(P)}, inside box {ins.sum()} ({100*ins.mean() if len(P) else 0:.0f}%), 50 m cells with geometry {len(cells)}/{ncell} ({100*len(cells)/max(ncell,1):.0f}%), height {np.percentile(h[ins],5) if ins.any() else float('nan'):.1f}-{np.percentile(h[ins],95) if ins.any() else float('nan'):.1f} m")
