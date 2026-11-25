#!/usr/bin/env python3
"""Publish the locked resolution A/B and its measurements on the exact two recorded pilot trajectories."""
import json
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from collect_reality_resolution import CFG, H, W, OUT, ROOT, TRACKS, open_fields, sweep
from fit_reality_resolution import FIT, read_head
from keireality.dataset import file_hash, write_json, writer_lock
from keireality.video import title_font
from keisim.config import SEM
from score_reality_resolution import Scores, SHEET, changes, quality_check, score_town


def detail_still(path, raw, t, before, after, seen):
    gt, seg = raw["rgb"][t,0], raw["labels"][t,0]
    paint = (seg==SEM["marking"]) & (gt.mean(-1)>=102) & (np.ptp(gt,axis=-1)<=41) & seen
    car = (seg==SEM["vehicle"]) & seen
    centers = []
    for idx,mask in enumerate((car,paint)):
        if idx:
            mask[:H//2] = False
        n,_,box,centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        if n<2:
            raise ValueError("No labeled object in the preselected comparison frame")
        centers.append(centroids[1+box[1:,cv2.CC_STAT_AREA].argmax()])
    w,h,zoom = 100,50,3
    header = 30
    board = Image.new("RGB",(3*w*zoom,2*(h*zoom+header)),(24,27,29))
    draw = ImageDraw.Draw(board)
    for row,(cx,cy) in enumerate(centers):
        x = int(np.clip(round(cx)-w//2,0,W-w)); y = int(np.clip(round(cy)-h//2,0,H-h))
        top = row*(h*zoom+header)
        for i,(im,name) in enumerate(zip((before,after,gt),("現行 256×128","試作 384×192","正解 384×192"),strict=True)):
            draw.text((i*w*zoom+8,top+3),("車" if row==0 else "白線")+"　"+name,font=title_font(),fill="white")
            tile = Image.fromarray(im[y:y+h,x:x+w]).resize((w*zoom,h*zoom),Image.Resampling.NEAREST)
            board.paste(tile,(i*w*zoom,top+header))
    board.save(path)


def bandwidth_check(town, movies, folder):
    raw = open_fields(OUT/"capture"/f"town{town}")
    features = open_fields(OUT/"features"/f"town{town}")
    reports = {}
    for slot,track in enumerate(TRACKS):
        score = Scores(["down256_up384"])
        t = 0
        for name in movies[f"town{town}-{track}"]:
            with Image.open(folder/name) as image:
                frames = np.array(image).reshape(-1,H,2*W,3)
            native = torch.from_numpy(frames[:,:,W:].copy()).permute(0,3,1,2).float()/255
            small = F.interpolate(native,size=(128,256),mode="bicubic",align_corners=False,antialias=True)
            limited = F.interpolate(small,size=(H,W),mode="bicubic",align_corners=False).clamp(0,1)
            rgb = (limited*255).round().byte().permute(0,2,3,1).numpy()
            for im,ab in zip(rgb,frames,strict=True):
                score.frame(im[None],raw["rgb"][t,slot],raw["labels"][t,slot],features["visible"][t,slot],
                            features["warp"][t,slot])
                if town==41 and slot==0 and t==83:
                    detail_still(folder/"town41-details.png",raw,t,ab[:,:W],ab[:,W:],features["visible"][t,slot])
                t += 1
        if t!=len(raw["pose"]):
            raise ValueError("Incomplete synchronized video")
        reports[track] = score.results()
    return reports


def evaluate():
    selection_path = FIT/"selection.json"
    selection = json.loads(selection_path.read_text())
    if (file_hash(FIT/"head.pt")!=selection["checkpoint"] or file_hash(ROOT/"keireality/resolution.py")!=selection["model"] or
            file_hash(Path(__file__).with_name("score_reality_resolution.py"))!=selection["scorer"]):
        raise ValueError("The frozen selection, model or scorer changed")
    strength = selection["strength"]
    head = read_head()
    n = CFG["validation_ticks"]
    clips = dict(towns=CFG["validation_towns"], width=W, height=H, angles=[sweep(t) for t in range(n)],
                 fps=10,perSheet=SHEET,windowed=True,defaultTrack="chase",labels=["現行 256×128","試作 384×192"],
                 liveModes=dict(sharp="stable",stable="stable384",wipe="stable384"),clips={})
    dest = OUT/"comparison"
    dest.mkdir(exist_ok=True)
    town_scores,checks,changes_by_town,source = {},{},{},{}
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        folder = Path(tmp)
        for town in CFG["validation_towns"]:
            src = OUT/"capture"/f"town{town}"/"capture.json"
            source[town] = dict(sha256=file_hash(src),**json.loads(src.read_text())["spec"])
            if source[town]["start"]!=0 or source[town]["count"]!=n:
                raise ValueError("Wrong control trajectory")
            results, movies = score_town(town,head,[strength],folder)
            limited = bandwidth_check(town,movies,folder)
            checks[town],changes_by_town[town] = {},{}
            for track,row in results.items():
                row["methods"]["native384"] = row["methods"].pop(str(strength))
                if row["counts"] != limited[track]["counts"]:
                    raise ValueError("Bandwidth control did not score identical teacher pixels")
                row["methods"].update(limited[track]["methods"])
                checks[town][track] = quality_check(row,"native384")
                changes_by_town[town][track] = changes(row,"native384")
            town_scores[town] = results
            clips["clips"].update(movies)
            print("validation",town,checks[town],flush=True)
        (folder/"clips.js").write_text("const clips = "+json.dumps(clips,ensure_ascii=False)+";\n",encoding="utf-8")
        shutil.copyfile(Path(__file__).with_name("reality_comparison.html"),folder/"index.html")
        for p in sorted(folder.iterdir(), key=lambda f:f.name=="index.html"):
            p.replace(dest/p.name)
    passed = all(all(row.values()) for tracks in checks.values() for row in tracks.values())
    write_json(OUT/"quality.json",dict(status="PASS" if passed else "HOLD",selection=file_hash(selection_path),
               strength=strength,source=source,baseline="256x128 stabilized output, bicubic to 384x192",
               methods=town_scores,relative_changes=changes_by_town,checks=checks,
               bandwidth_control="native384 reduced to 256x128 with antialiasing, then bicubic back to the common grid"))


if __name__=="__main__":
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    with writer_lock(OUT/"evaluation"):
        evaluate()
