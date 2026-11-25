#!/usr/bin/env python3
"""Enlarge the evaluated D clips and serve a loopback preview from live KeiSim cameras."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keireality.background_history import BackgroundHistoryDisplay
from keireality.dataset import file_hash, load_manifest, write_json
from keireality.encoder import extract_mem, frozen_pilot
from keireality.geometry import CHASE_CAMERA, ORBIT_ANGLES, orbit_camera, relative_pose, view_pose
from keireality.surface_observation import ObservedDetail
from keisim.config import EnvConfig, set_camera_rig
from keisim.env import KeiEnv
from diagnose_reality_surface_gaps import read_detail
from eval_reality_temporal import load_renderer
from prepare_reality_projection import checked_fields


RUN = ROOT / "runs/reality/p2_surface_history"
OUT = RUN / "preview"
TOWNS = {e["town_seed"]: e for e in json.loads((ROOT/"configs/reality/p2_episodes.json").read_text())
         if e["split"] == "val"}
HTML = Path(__file__).with_name("surface_preview.html")
FPS = 10
ORBIT_LIMIT = max(abs(angle) for angle in ORBIT_ANGLES)


class Models:
    def __init__(self, device):
        self.device = device
        validated = json.loads((RUN/"val/metrics.json").read_text())["contract"]
        for name in ("keireality/background_history.py", "keireality/surface_observation.py",
                     "keireality/scene_detail.py", "runs/keipilot_mc.pt", "runs/reference/mc14-depth.pt",
                     "runs/reality/p2/temporal/best.pt", "runs/reality/p2_scene_detail/train/readout.pt",
                     "configs/reality/p2_scene_detail.json"):
            if file_hash(ROOT/name) != validated[name]:
                raise ValueError(f"preview differs from the evaluated D model: {name}")
        self.pilot = frozen_pilot(ROOT/"runs/keipilot_mc.pt",device)
        self.head = frozen_pilot(ROOT/"runs/reference/mc14-depth.pt",device)
        self.phase2 = load_renderer(ROOT/"runs/reality/p2/temporal/best.pt",validated["runs/keipilot_mc.pt"],device)
        self.detail = read_detail(ObservedDetail,device)
        self.view = torch.tensor(view_pose(CHASE_CAMERA),device=device)[None]

    def new_drive(self):
        return BackgroundHistoryDisplay(self.phase2,self.detail)

    @torch.inference_mode()
    def frame(self, display, rgb, t, delta, episode, angle=0):
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(0,3,1,2)[None].to(self.device)
        features = extract_mem(self.pilot,image,pyramid=True,scene=True)
        for name in ("mem","c2","c3"):
            features[name] = features[name].half()
        features["estimated_depth"] = self.head.depth_metres(self.head.forward_depth(image)).reshape(1,2,80,160)
        display.observe(features,episode,t,delta)
        view = self.view if angle == 0 else torch.as_tensor(view_pose(orbit_camera(angle)), device=self.device)[None]
        pixels = display.render(view)[0][0].permute(1,2,0).cpu().numpy()
        return np.rint(pixels.clip(0,1)*255).astype(np.uint8)


def export_clips(models):
    data = ROOT/"data/reality/p2"
    rows = load_manifest(data)["shards"]
    OUT.mkdir(exist_ok=True)
    metadata = {}
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        stage = Path(tmp)
        for town,ep in TOWNS.items():
            display = models.new_drive()
            frames = []
            for row in (r for r in rows if r["split"]=="val" and r["town_seed"]==town):
                fields = checked_fields(data,row,("step","t_sim","delta_pose","rgb_in"))
                for step,t,delta,rgb in zip(fields["step"],fields["t_sim"],fields["delta_pose"],fields["rgb_in"],strict=True):
                    if int(step)!=ep["start_step"]+len(frames):
                        raise ValueError("nonconsecutive validation frames")
                    frames.append(models.frame(display,rgb.copy(),t,delta,row["episode_id"]))
            if len(frames)!=ep["frames"]:
                raise ValueError("incomplete validation clip")
            middle = frames[len(frames)//2]
            reference = np.asarray(Image.open(RUN/f"val/town{town}.png").convert("RGB"))[242:,384:]
            diff = np.abs(reference.astype(int)-cv2.resize(middle,(384,192),interpolation=cv2.INTER_NEAREST))
            if diff.mean()>.02:
                raise ValueError(f"town {town}: midpoint differs from the validated D panel")
            Image.fromarray(middle).resize((1024,512),Image.Resampling.NEAREST).save(stage/f"town{town}.png")
            subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-f","rawvideo","-pixel_format","rgb24",
                            "-video_size","256x128","-framerate",str(FPS),"-i","-","-vf",
                            "scale=1024:512:flags=neighbor","-c:v","libx264","-crf","14","-pix_fmt","yuv420p",
                            "-movflags","+faststart",str(stage/f"town{town}.mp4")],
                           input=np.stack(frames).tobytes(),check=True)
            metadata[town] = dict(frames=len(frames),fps=FPS,native=[256,128],display=[1024,512],
                                  midpoint_mean_difference_8bit=float(diff.mean()),midpoint_max_8bit=int(diff.max()))
        shutil.copyfile(HTML,stage/"index.html")
        write_json(stage/"clips.json",metadata)
        for path in stage.iterdir():
            path.replace(OUT/path.name)
    print(json.dumps(metadata),flush=True)


class ReplacedDrive(Exception):
    """The requesting browser no longer owns the active drive."""


class LiveSimulation:
    """Advance one tick per displayed frame. Reuse KeiView across pauses and town changes."""
    def __init__(self, models):
        self.models = models
        cfg = set_camera_rig(EnvConfig(renderer="keiview",weather="noon",keiview_gl="hw"),"wide_narrow")
        self.env = KeiEnv(cfg)
        self.lock = threading.Lock()
        self.session = None

    def start(self,town):
        ep = TOWNS[town]
        with self.lock:
            self.session = None
            self.env.reset(town_seed=town,episode_seed=ep["episode_seed"],render=False)
            for _ in range(ep["start_step"]):
                self.env.step(self.env.expert_action())
            self.display = self.models.new_drive()
            self.previous = None
            self.tick = 0
            self.session = secrets.token_hex(16)
            return self.session

    def frame(self,session,angle=0):
        with self.lock:
            if not session or session != self.session:
                raise ReplacedDrive
            if self.env.done:
                return None
            start = time.perf_counter()
            world = self.env.world
            pose = tuple(world.ego.pose)
            cameras = self.env.keiview().render_rig(pose,world.scene_state(include_ego=False),world.t,want_seg=False)
            rgb = np.stack([bgr[...,::-1] for bgr,_ in cameras])
            captured = time.perf_counter()
            frame = self.models.frame(self.display,rgb,world.t,relative_pose(self.previous,pose),session,angle)
            rendered = time.perf_counter()
            ok, png = cv2.imencode(".png",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,1])
            if not ok:
                raise RuntimeError("could not encode model output")
            stats = {"Tick":self.tick, "Speed":world.ego.v*3.6, "View-Angle":angle,
                     "Camera-Ms":(captured-start)*1000, "Model-Ms":(rendered-captured)*1000}
            self.previous = pose
            self.env.step(self.env.expert_action())
            self.tick += 1
            return png.tobytes(),stats


class PreviewHandler(BaseHTTPRequestHandler):
    def asset(self, path, content_type):
        if not path.is_file():
            self.send_error(404)
            return
        size = path.stat().st_size
        start, end = 0, size-1
        partial = self.headers.get("Range")
        if partial:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)",partial)
            if match and any(match.groups()):
                left,right = match.groups()
                start = int(left) if left else max(0,size-int(right))
                end = min(size-1,int(right)) if left and right else size-1
            else:
                start = size
            if not 0<=start<=end:
                self.send_response(416)
                self.send_header("Content-Range",f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type",content_type)
        if content_type.startswith("text/html"):
            self.send_header("Cache-Control","no-cache")
        self.send_header("Content-Length",str(end-start+1))
        self.send_header("Accept-Ranges","bytes")
        if partial:
            self.send_header("Content-Range",f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command!="HEAD":
            with path.open("rb") as source:
                source.seek(start)
                self.wfile.write(source.read(end-start+1))

    def handle_route(self):
        url = urlsplit(self.path)
        if url.path in ("/","/index.html"):
            return self.asset(HTML,"text/html; charset=utf-8")
        for town in TOWNS:
            for ext,content_type in (("png","image/png"),("mp4","video/mp4")):
                if url.path==f"/town{town}.{ext}":
                    return self.asset(OUT/f"town{town}.{ext}",content_type)
        self.send_error(404)

    def post_frame(self):
        url = urlsplit(self.path)
        allowed_hosts = {f"127.0.0.1:{self.server.server_port}",f"localhost:{self.server.server_port}"}
        if (self.headers.get("Host") not in allowed_hosts or self.headers.get("Sec-Fetch-Site")=="cross-site"
                or self.headers.get("X-KeiReality")!="preview"):
            self.send_error(403)
            return
        query = parse_qs(url.query)
        stats = {}
        if url.path=="/start":
            town = query.get("town",[])
            if len(town)!=1 or town[0] not in {str(t) for t in TOWNS}:
                self.send_error(400)
                return
            data = self.server.simulation.start(int(town[0])).encode("ascii")
            content_type = "text/plain"
        elif url.path=="/frame":
            try:
                angle = float(self.headers.get("X-KeiReality-Angle", "0"))
            except ValueError:
                self.send_error(400, "Invalid view angle")
                return
            if not -ORBIT_LIMIT <= angle <= ORBIT_LIMIT:
                self.send_error(400, "View angle outside the trained arc")
                return
            try:
                result = self.server.simulation.frame(self.headers.get("X-KeiReality-Session"),angle)
            except ReplacedDrive:
                self.send_error(409)
                return
            if result is None:
                self.send_response(204)
                self.end_headers()
                return
            data,stats=result
            content_type="image/png"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type",content_type)
        self.send_header("Cache-Control","no-store")
        self.send_header("Content-Length",str(len(data)))
        for key,value in stats.items():
            self.send_header("X-"+key,str(value))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        try:
            self.post_frame()
        except (BrokenPipeError,ConnectionResetError):
            pass
        except Exception as exc:
            print(f"live preview: {exc}",file=sys.stderr,flush=True)
            self.send_error(500)

    def log_request(self,code="-",size="-"):
        if self.path!="/frame" or code!=200:
            super().log_request(code,size)

    def do_GET(self):
        try:
            self.handle_route()
        except (BrokenPipeError,ConnectionResetError):
            pass

    do_HEAD = do_GET


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export",action="store_true",help="replay the validation camera inputs and save 4x clips")
    parser.add_argument("--port",type=int,default=8766)
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args=parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    models=Models(args.device)
    if args.export:
        export_clips(models)
    else:
        if not all((OUT/f"town{t}.mp4").is_file() for t in TOWNS):
            export_clips(models)
        else:
            shutil.copyfile(HTML,OUT/"index.html")
        with ThreadingHTTPServer(("127.0.0.1",args.port),PreviewHandler) as server:
            server.simulation=LiveSimulation(models)
            try:
                print(f"KeiReality http://127.0.0.1:{server.server_port}/",flush=True)
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.simulation.env.close()
