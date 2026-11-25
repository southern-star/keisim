"""Run the normal mc14 driving policy and an optional latest-frame KeiReality display."""
from dataclasses import dataclass
from hashlib import sha256
import json
import threading
from time import perf_counter

import numpy as np
import torch

from diagnose_reality_blur import ROOT, OUT, WEIGHTS
from diagnose_reality_surface_gaps import read_detail
from eval_reality_depth_refiner import contract, read_refiner
from eval_reality_temporal import load_renderer
from keipilot.agent import KeiPilotAgent
from keireality.dataset import file_hash
from keireality.encoder import frozen_pilot
from keireality.geometry import orbit_camera, relative_pose, view_pose
from keireality.refined_display import RefinedDrive
from keireality.resolution import ResolutionDrive, ResolutionHead
from keireality.surface_observation import ObservedDetail
from keisim.config import EnvConfig, set_camera_rig
from keisim.env import KeiEnv


OUTPUT = ROOT / "runs/reality/closed_loop"
TOWNS = {r["town_seed"]: r["episode_seed"] for r in
         json.loads((ROOT / "configs/reality/p2_episodes.json").read_text()) if r["split"] == "val"}
YIELD_HOLD = (4, 1, 3.5, 4, .5, .5, 10)


class Runtime:
    def __init__(self):
        pair = json.loads((OUT / "development/paired-selection.json").read_text())
        if pair["weights_and_software"] != contract():
            raise ValueError("Validation weights or rendering code changed")
        self.methods = {"sharp": pair["primary"], "stable": pair["secondary"]}
        self.resolution_dir = ROOT / "runs/reality/p2_resolution/fit"
        self.resolution_head = None
        if (self.resolution_dir/"selection.json").is_file():
            self.methods["stable384"] = pair["secondary"]
        self.device = "cuda"
        self.agent = KeiPilotAgent(ROOT / WEIGHTS[0], self.device, brake_hold=True, red_hold=True, yield_hold=YIELD_HOLD)
        self.agent.model.requires_grad_(False)
        if self.agent.model.rig != "wide_narrow":
            raise ValueError("Expected the mc14 wide/narrow rig")
        self.depth = frozen_pilot(ROOT / WEIGHTS[1], self.device)
        self.renderer = load_renderer(ROOT / WEIGHTS[2], file_hash(ROOT / WEIGHTS[0]), self.device)
        self.detail = read_detail(ObservedDetail, self.device)
        self.refiner = read_refiner(self.device)
        cfg = set_camera_rig(EnvConfig(renderer="keiview", weather="noon", keiview_gl="hw"), self.agent.model.rig)
        self.env = KeiEnv(cfg)
        self.dt = cfg.dt * cfg.action_repeat
        self.control_stream = torch.cuda.Stream(priority=-1)
        torch.cuda.synchronize()

    def display(self, mode):
        baseline = RefinedDrive(self.renderer, self.detail, self.refiner, self.methods[mode])
        if mode != "stable384":
            return baseline
        if self.resolution_head is None:
            selection = json.loads((self.resolution_dir/"selection.json").read_text())
            path = self.resolution_dir/"head.pt"
            if (file_hash(path) != selection["checkpoint"] or
                    file_hash(ROOT/"keireality/resolution.py") != selection["model"]):
                raise ValueError("Native-resolution decoder differs from the selected model")
            head = ResolutionHead().to(self.device, memory_format=torch.channels_last)
            head.load_state_dict(torch.load(path,map_location="cpu")["model"])
            self.resolution_head = head.eval().requires_grad_(False)
            self.resolution_strength = selection["strength"]
            torch.cuda.synchronize()
        return ResolutionDrive(baseline, self.resolution_head, self.resolution_strength)


@dataclass
class Frame:
    tick: int
    sim_time: float
    pose: tuple
    speed: float
    target: float
    action: tuple
    distance: float
    angle: int
    wide: np.ndarray
    narrow: np.ndarray
    started: float


class DisplayWorker:
    """One GPU job in flight and one replaceable observation. The controller never joins a frame."""
    def __init__(self, runtime, mode, episode, publish):
        self.runtime = runtime
        self.episode = episode
        self.publish = publish
        self.display = runtime.display(mode)
        self.condition = threading.Condition()
        self.pending = None
        self.done = False
        self.error = None
        self.dropped = 0
        self.thread = threading.Thread(target=self.work, name="reality-display")
        self.thread.start()

    def submit(self, frame, features, ready):
        with self.condition:
            if self.error:
                raise RuntimeError("KeiReality display failed") from self.error
            if self.pending is not None:
                self.dropped += 1
            self.pending = (frame, features, ready)
            self.condition.notify()

    @torch.inference_mode()
    def work(self):
        try:
            stream = torch.cuda.Stream()
            previous = None
            views = {}
            with torch.cuda.stream(stream):
                while True:
                    with self.condition:
                        self.condition.wait_for(lambda: self.pending is not None or self.done)
                        if self.pending is None:
                            return
                        frame, raw, ready = self.pending
                        self.pending = None
                    start = perf_counter()
                    stream.wait_event(ready)
                    # Keep producer tensors alive until the CPU copy completes on this stream.
                    features = {key: raw[key].half() for key in ("mem", "c2", "c3")}
                    features.update(stem=raw["stem"], source_logits=raw["source_logits"])
                    depth = self.runtime.depth
                    features["estimated_depth"] = depth.depth_metres(depth.forward_depth(raw["rgb_in"])).reshape(1, 2, 80, 160)
                    self.display.observe(features, self.episode, frame.sim_time, relative_pose(previous, frame.pose))
                    if frame.angle not in views:
                        views[frame.angle] = torch.as_tensor(view_pose(orbit_camera(frame.angle)), device=self.runtime.device)[None]
                    pixels = self.display.render(views[frame.angle])[0].permute(1, 2, 0)
                    rgb = pixels.clamp(0, 1).mul(255).round().byte().contiguous().cpu().numpy()
                    self.publish(frame, rgb, start)
                    previous = frame.pose
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.pending = None
                self.condition.notify_all()

    def finish(self):
        with self.condition:
            self.done = True
            self.condition.notify()
        self.thread.join()
        if self.error:
            raise RuntimeError("KeiReality display failed") from self.error


class Drive:
    def __init__(self, runtime, town, mode, *, capture_trace=False):
        self.runtime = runtime
        self.town = town
        self.mode = mode
        self.capture_trace = capture_trace
        self.stop = threading.Event()
        self.condition = threading.Condition()
        self.angle = 0
        self.latest = None
        self.steps = []
        self.rendered = []
        self.dropped = 0
        self.finished = False
        self.error = None
        self.result = None

    def publish(self, frame, rgb, start):
        ended = perf_counter()
        with self.condition:
            if rgb is not None:
                self.rendered.append((frame.tick, ended, (ended-start)*1000, (ended-frame.started)*1000))
            self.latest = (frame, rgb)
            self.condition.notify_all()

    @torch.inference_mode()
    def run(self, limit=None):
        env, agent, dt = self.runtime.env, self.runtime.agent, self.runtime.dt
        worker = None
        try:
            agent.reset()
            env.reset(self.town, TOWNS[self.town], render=False)
            if env.keiview().gl != "hw":
                raise RuntimeError("Hardware KeiView cameras required for this benchmark")
            if self.mode != "pilot":
                worker = DisplayWorker(self.runtime, self.mode, (self.town, TOWNS[self.town]), self.publish)
            scheduled = perf_counter()
            while not env.done and (limit is None or len(self.steps) < limit):
                if self.stop.wait(max(0., scheduled - perf_counter())):
                    break
                start = perf_counter()
                # If a cycle overruns, start the next one immediately without replaying past time slots.
                scheduled = start + dt
                pose, sim_time, speed = tuple(env.world.ego.pose), env.elapsed, env.world.ego.v
                cameras = [bgr for bgr, _ in env.render_cameras(want_rgb=True, want_seg=False)]
                captured = perf_counter()
                with torch.cuda.stream(self.runtime.control_stream):
                    action, plan = agent.act(cameras, env.command(), env.target_point(), speed, dt,
                                             with_reality=worker is not None)
                    ready = self.runtime.control_stream.record_event() if worker else None
                decided = perf_counter()
                frame = Frame(len(self.steps), sim_time, pose, speed, plan["target_speed"], tuple(map(float, action)),
                              env.distance, self.angle, cameras[0], cameras[1], start)
                if worker:
                    worker.submit(frame, plan.pop("reality"), ready)
                env.step(action)
                ended = perf_counter()
                digest = None
                if self.capture_trace:
                    digest = sha256()
                    for camera in cameras:
                        digest.update(np.ascontiguousarray(camera))
                row = dict(tick=frame.tick, start=start, end=ended, camera_ms=1000*(captured-start),
                           pilot_ms=1000*(decided-captured), work_ms=1000*(ended-start), pose=pose,
                           speed=speed, target=frame.target, action=frame.action,
                           camera_hash=digest.hexdigest() if digest else None)
                with self.condition:
                    self.steps.append(row)
                if worker is None:
                    self.publish(frame, None, start)
            self.result = env.metrics()
            self.result["stop_reason"] = (env.status if env.done else "stopped" if self.stop.is_set() else "tick_limit")
            self.result["final_pose"] = list(env.world.ego.pose)
            self.result["final_speed"] = float(env.world.ego.v)
        except Exception as exc:
            self.error = exc
            raise
        finally:
            try:
                if worker is not None:
                    worker.finish()
                    self.dropped = worker.dropped
            except Exception as exc:
                self.error = exc
                raise
            finally:
                with self.condition:
                    self.finished = True
                    self.condition.notify_all()
