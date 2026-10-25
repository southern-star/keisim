"""For each (ward, box): the road LOD3 leaf tiles whose region intersects the box, their total size, and the batch
tables' feature counts by class / function (uncompressed, so no Draco needed)."""
import json, math, struct, sys, urllib.request, urllib.parse, collections, os
UA = {"User-Agent": "keisim-site-survey/0.1"}
req = urllib.request.Request("https://api.plateauview.mlit.go.jp/datacatalog/plateau-datasets", headers=UA)
d = json.loads(urllib.request.urlopen(req, timeout=90).read())["datasets"]
sets = {(x.get("ward") or x.get("city")): x["url"] for x in d if x.get("pref") == "東京都" and x.get("type_en") == "tran" and str(x.get("lod")) == "3" and x.get("year") == 2025 and x.get("url")}
os.makedirs("b3dm", exist_ok=True)
def get(url, cache=True):
    fn = os.path.join("b3dm", urllib.parse.quote(url, safe=""))[:200]
    if cache and os.path.exists(fn): return open(fn, "rb").read()
    buf = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300).read()
    if cache: open(fn, "wb").write(buf)
    return buf
def leaf_tiles(url, depth=0, out=None):
    out = [] if out is None else out
    ts = json.loads(get(url))
    def walk(t, base):
        reg = t.get("boundingVolume", {}).get("region")
        c = t.get("content") or (t.get("contents") or [None])[0]
        if c and c.get("uri", c.get("url", "")):
            uri = urllib.parse.urljoin(base, c.get("uri", c.get("url")))
            if uri.endswith(".json") and depth < 3: leaf_tiles(uri, depth + 1, out)
            elif reg: out.append(([math.degrees(v) for v in reg[:4]], uri))
        for ch in t.get("children", []): walk(ch, base)
    walk(ts["root"], url); return out
def batch_table(buf):
    if buf[:4] != b"b3dm": return {}, 0
    hl = struct.unpack("<7I", buf[:28]); ftj, ftb, btj, btb = hl[3:7]
    ft = json.loads(buf[28:28 + ftj] or b"{}"); bt = json.loads(buf[28 + ftj + ftb:28 + ftj + ftb + btj] or b"{}")
    return bt, ft.get("BATCH_LENGTH", 0)
boxes = json.load(open(sys.argv[1]))
tiles_cache = {}
for name, ward, (s, w, n, e) in boxes:
    if ward not in tiles_cache: tiles_cache[ward] = leaf_tiles(sets[ward])
    hit = [t for t in tiles_cache[ward] if not (t[0][2] < w or t[0][0] > e or t[0][3] < s or t[0][1] > n)]
    size, feats, cls, keys = 0, 0, collections.Counter(), collections.Counter()
    for reg, uri in hit:
        buf = get(uri); size += len(buf)
        bt, n_b = batch_table(buf); feats += n_b
        for k in bt: keys[k] += 1
        for k in ("function", "tran:function", "class", "tran:class", "gml:name", "feature_type", "usage", "tran:usage"):
            if k in bt and isinstance(bt[k], list):
                for v in bt[k]: cls[(k, str(v)[:24])] += 1
                break
    area_km2 = (e - w) * 111 * math.cos(math.radians(35.7)) * (n - s) * 111
    print(f"{name:22s} tiles {len(hit):3d}  {size/1e6:5.1f} MB  features {feats:6d}  ({feats/area_km2:6.0f}/km2)  classes {dict(cls.most_common(5))}", flush=True)
    if name == boxes[0][0]: print("   batch table keys:", sorted(keys)[:25])
