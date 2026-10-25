import json, sys, urllib.request, urllib.parse, collections, math, time
boxes = {
 "札幌 東区 (参考)":            (43.088, 141.360, 43.097, 141.372),
 "江戸川区 瑞江・南篠崎":       (35.683, 139.895, 35.695, 139.910),
 "江戸川区 船堀・一之江":       (35.685, 139.870, 35.697, 139.885),
 "江東区 南砂・東砂":           (35.670, 139.825, 35.682, 139.840),
 "板橋区 高島平":               (35.785, 139.655, 35.797, 139.670),
 "北区 浮間":                   (35.790, 139.690, 35.800, 139.705),
 "大田区 萩中・南六郷":         (35.550, 139.715, 35.562, 139.730),
 "立川市 錦町・羽衣町":         (35.690, 139.405, 35.702, 139.420),
 "多摩市 永山・諏訪":           (35.625, 139.430, 35.637, 139.445),
}
KEEP = {"primary","secondary","tertiary","residential","unclassified","living_street"}
def q(query):
    for attempt in range(3):
        try:
            req = urllib.request.Request("https://overpass-api.de/api/interpreter", data=urllib.parse.urlencode({"data": query}).encode(), headers={"User-Agent": "keisim-site-survey/0.1"})
            return json.loads(urllib.request.urlopen(req, timeout=180).read())["elements"]
        except Exception as e:
            err = e; time.sleep(20)
    raise err
out = {}
for name,(s,w,n,e) in boxes.items():
    els = q(f'[out:json][timeout:120];(way["highway"]({s},{w},{n},{e});node["highway"~"traffic_signals|stop|crossing"]({s},{w},{n},{e}););out body;>;out skel qt;')
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in els if el["type"]=="node"}
    ways = [el for el in els if el["type"]=="way" and el.get("tags",{}).get("highway") in KEEP]
    deg = collections.Counter(nid for wy in ways for nid in wy["nodes"])
    def dist(a,b):
        la,lo=nodes[a]; lb,lob=nodes[b]
        return math.hypot((lb-la)*111000, (lob-lo)*111000*math.cos(math.radians(la)))
    def wlen(wy): return sum(dist(wy["nodes"][i-1], wy["nodes"][i]) for i in range(1,len(wy["nodes"])))
    tot = sum(wlen(wy) for wy in ways)
    one = [wy for wy in ways if wy["tags"].get("oneway") in ("yes","-1")]
    one_len = sum(wlen(wy) for wy in one)
    multi = [wy for wy in ways if wy["tags"].get("lanes","").split(";")[0].isdigit() and int(wy["tags"]["lanes"].split(";")[0])>=3]
    multi_len = sum(wlen(wy) for wy in multi)
    spac=[]
    for wy in ways:
        last=None; acc=0
        for i,nid in enumerate(wy["nodes"]):
            if i: acc+=dist(wy["nodes"][i-1],nid)
            if deg[nid]>=2 or i==0 or i==len(wy["nodes"])-1:
                if last is not None and acc>5: spac.append(acc)
                last=nid; acc=0
    spac.sort()
    tagged = [el for el in els if el["type"]=="node" and el.get("tags")]
    sig = sum(1 for el in tagged if el["tags"].get("highway")=="traffic_signals")
    stop = sum(1 for el in tagged if el["tags"].get("highway")=="stop")
    cross = sum(1 for el in tagged if el["tags"].get("highway")=="crossing")
    kinds = collections.Counter(wy["tags"]["highway"] for wy in ways)
    out[name] = dict(ways=len(ways), km=round(tot/1000,1), oneway_pct=round(100*one_len/max(tot,1)), multi_pct=round(100*multi_len/max(tot,1)),
                     signals=sig, stops=stop, crossings=cross, block_med=round(spac[len(spac)//2]) if spac else None, kinds=dict(kinds.most_common(4)))
    r = out[name]
    print(f"{name:22s} 道路 {r['km']:5.1f} km  一方通行 {r['oneway_pct']:3d}%  3車線以上 {r['multi_pct']:2d}%  信号 {r['signals']:3d}  止まれ {r['stops']:3d}  横断歩道 {r['crossings']:4d}  区画長中央値 {r['block_med']} m  {r['kinds']}", flush=True)
json.dump(out, open("tokyo_stats.json","w"), ensure_ascii=False, indent=1)
