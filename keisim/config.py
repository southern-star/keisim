"""Configuration dataclasses and global constants for KeiSim."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------
# Semantic classes (shared by the camera segmentation, map texture and model)
# ----------------------------------------------------------------------------
SEM_NAMES = [
    "sky", "road", "marking", "sidewalk", "terrain", "building", "vegetation",
    "pole", "tl_red", "tl_yellow", "tl_green", "vehicle", "pedestrian",
]
SEM = {n: i for i, n in enumerate(SEM_NAMES)}
N_SEM = len(SEM_NAMES)

# BGR colours used for visualising semantic maps (Cityscapes-like)
SEM_COLORS = np.array([
    [180, 130, 70],    # sky
    [128, 64, 128],    # road
    [255, 255, 255],   # marking
    [232, 35, 244],    # sidewalk
    [152, 251, 152],   # terrain
    [70, 70, 70],      # building
    [35, 142, 107],    # vegetation
    [153, 153, 153],   # pole
    [0, 0, 255],       # tl red
    [0, 220, 255],     # tl yellow
    [0, 255, 0],       # tl green
    [142, 0, 0],       # vehicle
    [60, 20, 220],     # pedestrian
], dtype=np.uint8)

# Traffic light states
TL_RED, TL_YELLOW, TL_GREEN, TL_NONE = 0, 1, 2, 3
TL_NAMES = ["red", "yellow", "green", "none"]

# High-level navigation commands
CMD_LEFT, CMD_STRAIGHT, CMD_RIGHT = 0, 1, 2
CMD_NAMES = ["left", "straight", "right"]


@dataclass
class TownConfig:
    grid_min: int = 3
    grid_max: int = 5
    spacing_min: float = 70.0
    spacing_max: float = 100.0
    block_mode: str = "uniform"       # uniform: one block length per town | varied: one per grid row and column
    jitter: float = 0.12              # node jitter as a fraction of spacing
    edge_drop: float = 0.22           # probability of dropping a grid edge
    curve_prob: float = 0.4           # probability that a road is curved
    curve_max_deg: float = 22.0
    lane_width: float = 3.5
    shoulder: float = 0.25
    sidewalk_width: float = 2.5
    left_hand_traffic: bool = True    # Japan style (keep left)
    speed_limit: float = 10.0         # m/s (36 km/h)
    stop_line_setback: float = 5.0    # stop line distance before junction entry
    signal_green: tuple = (6.0, 9.0)
    signal_yellow: float = 2.5
    signal_allred: float = 1.5
    # split: every approach has its own green (no conflicts inside a junction). two_phase: opposite approaches
    # share the green (Japan's usual signal); right turns then wait inside the junction for a gap in the oncoming
    # traffic (see Lane.yields).
    signal_mode: str = "split"
    two_phase_green: tuple = (12.0, 18.0)
    two_phase_yellow: float = 3.0
    two_phase_allred: float = 2.0
    # Road types. uniform: every road is the two-lane street above (lane_width, shoulder, sidewalk_width,
    # speed_limit). mixed: `main_roads` interior grid lines become main roads and every other road is a street,
    # a residential road or an alley with the probabilities `road_mix` (ROAD_KINDS; drawn from a separate random
    # stream, so the layout is that of the uniform town with the same seed).
    road_types: str = "uniform"
    main_roads: int = 2
    road_mix: tuple = (0.3, 0.45, 0.25)
    building_prob: float = 0.85
    tree_prob: float = 0.55
    tex_res: float = 0.1              # metres per texel of the ground texture

    @property
    def road_half_width(self):
        return self.lane_width + self.shoulder

    @property
    def walk_outer(self):
        return self.road_half_width + self.sidewalk_width


# Road kinds of road_types="mixed" besides "street" (TownConfig's own lane / shoulder / sidewalk / speed limit).
# shoulder: from the lane's outer edge to the edge of the road surface; on roads without a sidewalk it is the
# 路側帯 where people walk. center: centre line (dashed: white, solid near signals; yellow: solid, no overtaking;
# none). edge: edge lines (classic: KeiSim's original, 0.35 m inside the road edge; lane: just outside the lane;
# none). Speed limits in m/s.
ROAD_KINDS = {
    "main": {"lane_width": 3.5, "shoulder": 0.75, "sidewalk": 3.5, "speed_limit": 40 / 3.6, "center": "yellow",
             "edge": "lane"},
    "residential": {"lane_width": 3.0, "shoulder": 0.75, "sidewalk": 0.0, "speed_limit": 30 / 3.6, "center": "none",
                    "edge": "lane"},
    # narrow (lanes under roadnet.NARROW_LANE): no trucks, and the outer turn of a bend waits for an oncoming car
    "alley": {"lane_width": 2.75, "shoulder": 0.25, "sidewalk": 0.0, "speed_limit": 20 / 3.6, "center": "none",
              "edge": "none"},
}


@dataclass
class TrafficConfig:
    vehicle_spacing: tuple = (32.0, 75.0)  # one NPC vehicle per this many metres of lane (sampled per episode)
    ped_spacing: tuple = (28.0, 60.0)      # one pedestrian per this many metres of road (sampled per episode)
    max_vehicles: int = 90
    max_peds: int = 70
    cross_rate: float = 1 / 90.0      # spontaneous crossing rate per pedestrian (1/s)
    ego_cross_rate: float = 0.12      # crossing-trigger rate for peds just ahead of ego (1/s)
    idm_a: float = 2.0
    idm_b: float = 3.0
    idm_T: float = 1.2
    idm_s0: float = 2.5
    lat_acc: float = 2.0              # comfortable lateral acceleration for curve speed
    # Box rule: wait at the stop line, even at green, while a vehicle from another approach is still inside the
    # junction (it may be crossing our path; two such vehicles stopped head to head never get out again).
    box_rule: bool = False
    # Deadlock release: NPCs standing still for `stuck_far_s` more than 60 m from the ego are moved elsewhere.
    # With release_hidden, so are those standing for `stuck_hidden_s` anywhere the ego camera cannot see
    # (beyond view_dist or more than view_half_deg off its heading); jams around the ego then clear up.
    stuck_far_s: float = 60.0
    release_hidden: bool = False
    stuck_hidden_s: float = 45.0
    view_half_deg: float = 60.0
    view_dist: float = 110.0


@dataclass
class CameraConfig:
    width: int = 320
    height: int = 160
    fov_deg: float = 100.0
    x: float = 0.6                    # forward offset from vehicle centre (m)
    y: float = 0.0
    z: float = 1.55                   # mount height (m)
    pitch_deg: float = 8.0            # downward pitch
    max_dist: float = 110.0           # far clipping distance for objects
    supersample: int = 2              # render at N x resolution then downsample


# Town styles (TownConfig overrides). "classic" is every suite up to KeiPilot v0.5; "varied" mixes block lengths
# of 70-200 m inside a town (longer blocks hold longer queues: half the deadlocks of the classic towns).
TOWN_STYLES = {
    "classic": {},
    "varied": {"block_mode": "varied", "spacing_min": 70.0, "spacing_max": 200.0},
    "twophase": {"block_mode": "varied", "spacing_min": 70.0, "spacing_max": 200.0, "signal_mode": "two_phase"},
    # twophase + road types: main roads (40 km/h, wide sidewalks, yellow centre line), streets, residential roads
    # without sidewalks (30 km/h, people walk in the 路側帯) and narrow alleys (20 km/h)
    "roadtypes": {"block_mode": "varied", "spacing_min": 70.0, "spacing_max": 200.0, "signal_mode": "two_phase",
                  "road_types": "mixed"},
}

# TownConfig fields added after town_tag() was introduced: part of the tag only when they differ from the
# default, so the tags (and exported town names) of the existing styles stay the same.
_TAG_LATER = ("road_types", "main_roads", "road_mix")


def town_config(style="classic"):
    cfg = TownConfig()
    for k, v in TOWN_STYLES[style].items():
        setattr(cfg, k, v)
    return cfg


def town_tag(cfg):
    """'' for the classic towns, else a short hash of the town settings (keeps exported towns apart)."""
    import hashlib
    from dataclasses import asdict
    d = asdict(cfg)
    base = asdict(TownConfig())
    if d == base:
        return ""
    for k in _TAG_LATER:
        if d[k] == base[k]:
            del d[k]
    if cfg.road_types != "uniform":          # the road kinds are settings of these towns too
        d["road_kinds"] = repr(sorted(ROAD_KINDS.items()))
    return hashlib.sha1(repr(sorted(d.items())).encode()).hexdigest()[:8]


def town_key(seed, cfg=None):
    """Name of an exported town: the seed, plus the settings tag for non-classic towns (e.g. '1010-3f2a9c1e')."""
    tag = town_tag(cfg) if cfg is not None else ""
    return f"{int(seed)}-{tag}" if tag else str(int(seed))


@dataclass
class EnvConfig:
    town: TownConfig = field(default_factory=TownConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    dt: float = 0.05                  # physics step (20 Hz)
    action_repeat: int = 2            # control at 10 Hz
    route_length: tuple = (450.0, 800.0)
    blocked_timeout: float = 90.0     # s without progress -> terminate
    route_deviation: float = 6.0      # m lateral from route -> terminate
    render_rgb: bool = True
    render_seg: bool = True
    render_bev: bool = False
    weather: str = "random"           # random | noon | cloudy | sunset | fog
    terminate_on_collision: bool = True
    renderer: str = "keisim"          # keisim (CPU, numpy + OpenCV) | keiview (web/, three.js on the GPU)
    keiview_quality: str = "medium"
    keiview_gl: str = "auto"          # auto (GPU, else SwiftShader on the CPU) | hw | soft
    keiview_roads: str = "plain"      # plain (the look of every KeiView dataset so far) | detail (markings, signs, ...)
    # The expert's give-way rule counts only the cars with priority that the ego camera can see right now (inside its
    # horizontal field of view, not behind a building; Traffic.visible_from), so its labels follow from the image:
    # with all of them, a car outside the 100 deg view made the expert wait where the image looked clear.
    expert_yield_visible: bool = False
