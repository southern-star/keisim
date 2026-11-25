#!/usr/bin/env python3
"""Measure D and the frozen candidate through Chrome without taking over the user's live drive."""
import json
import subprocess
import threading
from http.server import ThreadingHTTPServer

import cv2
import torch

from eval_reality_blur_pair import release
from eval_reality_depth_refiner import OUT, RefinedDrive, read_refiner
from keireality.dataset import file_hash, write_json
from preview_reality_surface import LiveSimulation, Models, PreviewHandler, ROOT


class LiveRefinement(RefinedDrive):
    def render(self,view,slot=0):
        return (super().render(view,slot),)


class BenchmarkModels(Models):
    def new_drive(self):
        if self.use_depth:
            return LiveRefinement(self.phase2,self.detail,self.refiner,self.variant)
        return super().new_drive()


class LocalHandler(PreviewHandler):
    def log_request(self,code="-",size="-"):
        if isinstance(code,int) and code>=400:
            super().log_request(code,size)


def run():
    decision=release()
    choices={"E":decision["candidates"]["primary"]["method"],"S":decision["candidates"]["secondary"]["method"]}
    models=BenchmarkModels("cuda")
    models.refiner=read_refiner("cuda")
    results={m:{} for m in ("D","E","S")}
    with ThreadingHTTPServer(("127.0.0.1",0),LocalHandler) as server:
        server.simulation=LiveSimulation(models)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            for town,methods in ((40,("D","E","S")),(41,("S","E","D"))):
                for m in methods:
                    models.use_depth=m!="D"
                    models.variant=choices.get(m)
                    cmd=["node",str(ROOT/"web/tools/measure_reality_sweep.mjs"),f"http://127.0.0.1:{server.server_port}/",str(town)]
                    record=json.loads(subprocess.check_output(cmd,cwd=ROOT,text=True))
                    print(town,m,record,flush=True)
                    results[m][town]=record
        finally:
            server.shutdown()
            thread.join()
            server.simulation.env.close()
    write_json(OUT/"val/browser.json",dict(methods=choices,release=file_hash(OUT/"release.json"),results=results))


if __name__=="__main__":
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    run()
