#!/usr/bin/env python3
"""Serve an independent local live view of the closed-loop controller and its display."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import threading
import traceback
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np
import torch

from reality_closed_loop import Drive, OUTPUT, Runtime, TOWNS
from keireality.drive_metrics import summarize


PAGE = Path(__file__).with_name("reality_closed_loop.html")
COMPARISON = OUTPUT.parent / "p2_resolution/comparison"
PUBLISHED = PAGE.resolve().parents[1] / "docs/keireality/384-v1"
REPORTS = {
    "/metrics.json": (OUTPUT / "metrics.json", PUBLISHED / "closed-loop-metrics.json"),
    "/browser.json": (OUTPUT / "browser.json", PUBLISHED / "closed-loop-browser.json"),
    "/resolution/timing.json": (COMPARISON.parent / "timing.json", PUBLISHED / "resolution-timing.json"),
    "/resolution/browser.json": (COMPARISON.parent / "browser.json", PUBLISHED / "resolution-browser.json"),
}


class Session:
    def __init__(self, runtime):
        self.runtime = runtime
        self.lock = threading.Lock()
        self.drive = self.thread = self.token = None

    def close_drive(self):
        if self.thread is not None:
            self.drive.stop.set()
            self.thread.join()

    def start(self, town, mode):
        with self.lock:
            self.close_drive()
            drive = Drive(self.runtime, town, mode)
            def run():
                try:
                    print(f"Starting closed loop on town {town} ({mode})", flush=True)
                    drive.run()
                    print(f"Finished town {town} ({mode}): {drive.result['stop_reason']}, {len(drive.steps)} controls", flush=True)
                except Exception:
                    traceback.print_exc()
            self.drive, self.token = drive, secrets.token_hex(16)
            self.thread = threading.Thread(target=run, name="keipilot-control")
            self.thread.start()
            return self.token

    def find(self, token):
        with self.lock:
            return self.drive if token and token == self.token else None


def recent_stats(steps, generated, tick):
    result = dict(latest_tick=steps[-1]["tick"] if steps else tick, control_hz=None, work_p95=None,
                  pilot_ms=None, generation_fps=None, latency_ms=None)
    if len(steps) >= 2:
        result.update(control_hz=(len(steps)-1)/(steps[-1]["start"]-steps[0]["start"]),
                      work_p95=float(np.percentile([r["work_ms"] for r in steps], 95)), pilot_ms=steps[-1]["pilot_ms"])
    if len(generated) >= 2:
        result.update(generation_fps=(len(generated)-1)/(generated[-1][1]-generated[0][1]), latency_ms=generated[-1][3])
    return result


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, content=b"", mime="application/json", meta=None):
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        if meta is not None:
            self.send_header("X-Kei-Frame", json.dumps(meta, separators=(",", ":"), allow_nan=False))
        self.end_headers()
        self.wfile.write(content)

    def frame(self, drive, after):
        with drive.condition:
            drive.condition.wait_for(lambda: drive.finished or (drive.latest and drive.latest[0].tick > after), timeout=1)
            if drive.error:
                return self.reply(500, b'{"error":"simulation_failed"}')
            if not drive.latest or drive.latest[0].tick <= after:
                if drive.finished:
                    timing = None
                    if len(drive.steps) >= 2:
                        timing = summarize(drive)
                        del timing["result"]
                    return self.reply(410, json.dumps(dict(**drive.result, timing=timing)).encode())
                return self.reply(204)
            frame, rgb = drive.latest
            steps, generated = drive.steps[-30:], drive.rendered[-30:]
        meta = dict(tick=frame.tick, sim_time=frame.sim_time, speed=frame.speed*3.6, target=frame.target*3.6,
                    action=frame.action, angle=frame.angle, distance=frame.distance,
                    **recent_stats(steps, generated, frame.tick))
        h,w = frame.wide.shape[:2]
        rh,rw = rgb.shape[:2] if rgb is not None else (0,0)
        meta["sizes"] = [[w,h],[w,h]] + ([[rw,rh]] if rgb is not None else [])
        atlas = np.zeros((max(h,rh), 2*w + rw, 3), np.uint8)
        atlas[:h, :w], atlas[:h, w:2*w] = frame.wide, frame.narrow
        if rgb is not None:
            atlas[:rh, 2*w:] = rgb[:,:,::-1]
        ok, png = cv2.imencode(".png", atlas, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise RuntimeError("PNG encoding failed")
        self.reply(200, png.tobytes(), "image/png", meta)

    def route(self, post):
        url = urlsplit(self.path)
        params = {k:v[0] for k,v in parse_qs(url.query).items() if len(v) == 1}
        if not post and url.path == "/":
            return self.reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if not post and url.path == "/modes.json":
            return self.reply(200, json.dumps(dict(modes=list(self.server.session.runtime.methods),
                              comparison=(COMPARISON/"index.html").is_file())).encode())
        if not post and url.path in REPORTS:
            local, published = REPORTS[url.path]
            path = local if local.is_file() else published
            return self.reply(200, path.read_bytes()) if path.is_file() else self.reply(404)
        if not post and url.path.startswith("/resolution/"):
            name = url.path.removeprefix("/resolution/") or "index.html"
            mime = {"index.html":"text/html; charset=utf-8","clips.js":"text/javascript; charset=utf-8"}.get(name)
            if name == "town41-details.png" or re.fullmatch(r"town(?:40|41)-(?:chase|sweep)-\d{3}\.png",name):
                mime = "image/png"
            if mime and (COMPARISON/name).is_file():
                return self.reply(200,(COMPARISON/name).read_bytes(),mime)
            return self.reply(404)
        if self.headers.get("Host") not in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"):
            return self.reply(403)
        if self.headers.get("X-KeiReality") != "closed-loop" or self.headers.get("Sec-Fetch-Site") == "cross-site":
            return self.reply(403)
        session = self.server.session
        if post and url.path == "/start":
            town, mode = params.get("town"), params.get("mode")
            if town not in {str(t) for t in TOWNS} or mode not in ("pilot", *session.runtime.methods):
                return self.reply(400)
            return self.reply(200, session.start(int(town), mode).encode(), "text/plain")
        drive = session.find(self.headers.get("X-Kei-Session"))
        if drive is None:
            return self.reply(409)
        if post and url.path == "/stop":
            drive.stop.set()
            return self.reply(204)
        if post and url.path == "/angle":
            angle = int(params.get("value", "1000"))
            if not -36 <= angle <= 36:
                return self.reply(400)
            drive.angle = angle
            return self.reply(204)
        if not post and url.path == "/frame":
            return self.frame(drive, int(params.get("after", "-1")))
        self.reply(404)

    def handle_route(self, post):
        try:
            self.route(post)
        except (ConnectionResetError, BrokenPipeError):
            pass
        except ValueError:
            self.reply(400)
        except Exception:
            traceback.print_exc()
            self.reply(500)

    def do_GET(self):
        self.handle_route(False)

    def do_POST(self):
        self.handle_route(True)

    def log_request(self, code="-", size="-"):
        if code not in (200, 204, 410):
            super().log_request(code, size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        runtime = Runtime()
        server.session = Session(runtime)
        try:
            print(f"KeiPilot × KeiReality http://127.0.0.1:{args.port}/", flush=True)
            server.serve_forever()
        finally:
            server.session.close_drive()
            runtime.env.close()
