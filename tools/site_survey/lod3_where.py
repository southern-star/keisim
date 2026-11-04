import json, math, urllib.request, urllib.parse
UA = {"User-Agent": "keisim-site-survey/0.1"}
req = urllib.request.Request("https://api.plateauview.mlit.go.jp/datacatalog/plateau-datasets", headers=UA)
d = json.loads(urllib.request.urlopen(req, timeout=90).read())["datasets"]
sets = [(x.get("ward") or x.get("city"), x["url"]) for x in d if x.get("pref") == "東京都" and x.get("type_en") == "tran" and str(x.get("lod")) == "3" and x.get("year") == 2025 and x.get("url")]
cache = {}
def get(url):
    if url not in cache:
        cache[url] = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read())
    return cache[url]
def tiles(url, depth=0, out=None):
    out = [] if out is None else out
    ts = get(url)
    def walk(t, base):
        reg = t.get("boundingVolume", {}).get("region")
        c = t.get("content") or (t.get("contents") or [None])[0]
        if c and c.get("uri", c.get("url", "")):
            uri = urllib.parse.urljoin(base, c.get("uri", c.get("url")))
            if uri.endswith(".json") and depth < 3:
                try: tiles(uri, depth + 1, out)
                except Exception as e: pass
            elif reg:
                out.append([math.degrees(v) for v in reg[:4]])
        for ch in t.get("children", []): walk(ch, base)
    walk(ts["root"], url)
    return out
res = {}
for name, url in sorted(sets):
    try: tl = tiles(url)
    except Exception as e: print(name, "ERR", e); continue
    if not tl: print(f"{name:8s} no region tiles"); continue
    w = min(t[0] for t in tl); s = min(t[1] for t in tl); e = max(t[2] for t in tl); n = max(t[3] for t in tl)
    area = sum((t[2]-t[0])*111*math.cos(math.radians(35.7))*(t[3]-t[1])*111 for t in tl)
    res[name] = tl
    print(f"{name:8s} tiles {len(tl):4d}  union lat {s:.3f}-{n:.3f} lon {w:.3f}-{e:.3f}  tile area sum {area:6.1f} km2  median tile {sorted((t[2]-t[0])*111*math.cos(math.radians(35.7))*1000 for t in tl)[len(tl)//2]:.0f} m wide", flush=True)
    if len(tl) <= 24:
        for t in sorted(tl, key=lambda t: (t[1], t[0])):
            print(f"         lat {t[1]:.4f}-{t[3]:.4f} lon {t[0]:.4f}-{t[2]:.4f}  ({(t[2]-t[0])*111*math.cos(math.radians(35.7))*1000:.0f} x {(t[3]-t[1])*111000:.0f} m)")
json.dump(res, open("lod3_tiles_tokyo.json", "w"))
