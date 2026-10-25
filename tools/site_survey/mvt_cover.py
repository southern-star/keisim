"""Road LOD1 / LOD2 MVT coverage of a box: fetch the z=16 tiles over it and count features per layer.
   python mvt_cover.py <ward> S W N E"""
import json, math, sys, urllib.request
import mapbox_vector_tile
UA = {"User-Agent": "keisim-site-survey/0.1"}
d = json.loads(urllib.request.urlopen(urllib.request.Request("https://api.plateauview.mlit.go.jp/datacatalog/plateau-datasets", headers=UA), timeout=90).read())["datasets"]
ward = sys.argv[1]; s, w, n, e = [float(v) for v in sys.argv[2:6]]
def tile(lat, lon, z):
    x = int((lon + 180) / 360 * 2 ** z); y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * 2 ** z)
    return x, y
z = 16
x0, y1 = tile(s, w, z); x1, y0 = tile(n, e, z)
for lod in (1, 2):
    u = next((x["url"] for x in d if x.get("pref") == "東京都" and (x.get("ward") or x.get("city")) == ward and x["type_en"] == "tran" and str(x.get("lod")) == str(lod) and x.get("year") == 2025), None)
    if not u: print(ward, "LOD", lod, "none"); continue
    feats, tiles, got = {}, 0, 0
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            tiles += 1
            try:
                buf = urllib.request.urlopen(urllib.request.Request(u.format(z=z, x=x, y=y), headers=UA), timeout=60).read()
            except Exception as ex:
                continue
            got += 1
            for layer, body in mapbox_vector_tile.decode(buf).items():
                for f in body["features"]:
                    key = (layer, str(f["properties"].get("tran:function", f["properties"].get("function", "")))[:12])
                    feats[key] = feats.get(key, 0) + 1
    print(f"{ward} road LOD{lod}: tiles with data {got}/{tiles}  features {dict(sorted(feats.items(), key=lambda kv: -kv[1])[:6])}")
