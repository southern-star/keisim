"""Road LOD3 feature density on a 1 km grid per ward: every leaf tile's BATCH_LENGTH (from the b3dm header, fetched
with a Range request) is spread over the grid cells its region overlaps, in proportion to the overlap."""
import json, math, struct, sys, urllib.request, urllib.parse, collections, os
from concurrent.futures import ThreadPoolExecutor
UA = {"User-Agent": "keisim-site-survey/0.1"}
tl_all = json.load(open("lod3_tiles_tokyo.json"))
req = urllib.request.Request("https://api.plateauview.mlit.go.jp/datacatalog/plateau-datasets", headers=UA)
d = json.loads(urllib.request.urlopen(req, timeout=90).read())["datasets"]
sets = {(x.get("ward") or x.get("city")): x["url"] for x in d if x.get("pref") == "東京都" and x.get("type_en") == "tran" and str(x.get("lod")) == "3" and x.get("year") == 2025 and x.get("url")}
def get(url, rng=None):
    h = dict(UA); 
    if rng: h["Range"] = rng
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120).read()
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
def batch_len(uri):
    try:
        buf = get(uri, "bytes=0-27")
        if buf[:4] != b"b3dm": return 0
        return struct.unpack("<7I", buf[:28])[2] and json.loads(get(uri, f"bytes=28-{28 + struct.unpack('<7I', buf[:28])[3] - 1}")).get("BATCH_LENGTH", 0)
    except Exception as e:
        return -1
wards = sys.argv[1:]
KM = 1 / 111.0
results = {}
for ward in wards:
    tl = leaf_tiles(sets[ward])
    with ThreadPoolExecutor(12) as ex: bl = list(ex.map(batch_len, [t[1] for t in tl]))
    bad = sum(1 for b in bl if b < 0)
    w0 = min(t[0][0] for t in tl); s0 = min(t[0][1] for t in tl)
    dlon = KM / math.cos(math.radians(35.7)); dlat = KM
    cells = collections.defaultdict(float)
    for (w, s, e, n), b in zip((t[0] for t in tl), bl):
        if b <= 0: continue
        area = max((e - w) * (n - s), 1e-12)
        i0, i1 = int((w - w0) / dlon), int((e - w0) / dlon); j0, j1 = int((s - s0) / dlat), int((n - s0) / dlat)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                cw, cs = w0 + i * dlon, s0 + j * dlat
                ov = max(0, min(e, cw + dlon) - max(w, cw)) * max(0, min(n, cs + dlat) - max(s, cs))
                if ov > 0: cells[(i, j)] += b * ov / area
    top = sorted(cells.items(), key=lambda kv: -kv[1])[:8]
    print(f"== {ward}: {len(tl)} tiles, {sum(b for b in bl if b > 0)} features, {bad} header errors; top 1 km cells (features/km2, box S W N E):")
    for (i, j), v in top:
        cw, cs = w0 + i * dlon, s0 + j * dlat
        print(f"   {v:6.0f}  {cs:.3f} {cw:.3f} {cs + dlat:.3f} {cw + dlon:.3f}")
        results.setdefault(ward, []).append((round(v), [round(cs, 3), round(cw, 3), round(cs + dlat, 3), round(cw + dlon, 3)]))
    sys.stdout.flush()
json.dump(results, open("lod3_grid.json", "w"), ensure_ascii=False, indent=1)
