#!/usr/bin/env python3
"""Put measured rotating-view outputs from both independent rollouts on the same video timeline."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from collect_reality_sweep import OUT, sweep
from keireality.dataset import file_hash, write_json
from keireality.video import japanese_header, write_video


def combine(town,dest):
    files=[OUT/d/f"town{town}-sweep.mp4" for d in ("val","val-stability")]
    caps=[cv2.VideoCapture(str(p)) for p in files]
    count=96
    try:
        for c in caps:
            if int(c.get(cv2.CAP_PROP_FRAME_COUNT))!=count or abs(c.get(cv2.CAP_PROP_FPS)-10)>.001:
                raise ValueError("mismatched source video")
        def panels():
            for tick in range(count):
                images=[]
                for c in caps:
                    ok,bgr=c.read()
                    if not ok or bgr.shape!=(434,768,3):
                        raise ValueError("incomplete source video")
                    images.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                a,b=images
                panes=(a[25:217,:384],a[242:,:384],a[242:,384:],b[242:,384:])
                names=(f"正解　町{town}　{tick/10:.1f}秒　{sweep(tick):+d}°","D　現在の表示",
                       "輪郭を優先した候補","ちらつきを抑えた候補")
                tiles=[np.vstack((japanese_header(label),p)) for label,p in zip(names,panes,strict=True)]
                image=np.vstack((np.hstack(tiles[:2]),np.hstack(tiles[2:])))
                if tick==12:
                    Image.fromarray(image).save(dest/f"town{town}-36deg.png")
                yield image
        if write_video(dest/f"town{town}-sweep.mp4",panels())!=count:
            raise ValueError("incomplete side-by-side")
    finally:
        for c in caps:
            c.release()


if __name__=="__main__":
    cv2.setNumThreads(1)
    dest=OUT/"comparison"
    dest.mkdir(exist_ok=True)
    for town in (40,41):
        combine(town,dest)
    write_json(dest/"media.json",{p.name:file_hash(p) for p in dest.iterdir() if p.suffix in (".mp4",".png")})
