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
    # lanes per direction of the main roads of a mixed town (2: 片側2車線 arterials, see RoadSpec.lanes). The cars
    # keep their lane: left turns from the kerb lane, right turns from the lane next to the centre line.
    main_lanes: int = 1
    # at-grade left-turn ramps (平面の左折ランプ, Town._make_ramps): the chance that a signalised junction corner where a
    # 2-lane main road's approach turns left gets one (a deceleration lane, a ramp around the corner, an acceleration
    # lane into the road turned into; the junction's own left turn from that approach goes)
    ramps: float = 0.0
    # a centre strip of this width (m) on the main roads (RoadSpec.median): hatched (導流帯) along the road, and a
    # right-turn lane (右折専用車線) next to the inner lane before the signalised junctions where the road's cars may
    # turn right (Town._make_pockets; left-hand traffic). 0: the main roads' two directions meet at the centre line.
    main_median: float = 0.0
    # scramble crossings (スクランブル交差点, 歩車分離): the chance that a signalised 4-arm junction of two-phase signals
    # on a main road gets a third phase in which every car light is red and people cross on all four zebras
    # and on the two diagonals across the junction (Town.ped_crossings, TrafficConfig.crosswalks). Their zebras are
    # red in the cars' phases. Every phase of such a junction is green for scramble_green s (people need about
    # 25 s over a diagonal): own random stream, so the rest of the town is that of scramble 0.
    scramble: float = 0.0
    scramble_green: tuple = (24.0, 32.0)
    # how a junction connector gets its turn (Lane.turn: the navigation command of the routes through it).
    # heading: by its change of heading alone (left over 0.5 rad, right under -0.5 rad, else straight); at a skewed
    # junction two exits of one approach can then share "left" (or "right"). ordered: the same, but where two exits of
    # an approach share a side and none is straight, the one closest to straight ahead (under TURN_ORDERED_MAX rad)
    # becomes "straight" (roadnet.turn_labels)
    turn_rule: str = "heading"
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
    # Patience with a standing ego: a car that gives way to the ego stops waiting for it once the ego has stood still
    # this long before the crossing (s; 0: it waits as long as the ego stands). Without it, an NPC at its 止まれ waits
    # for a standing ego for ever, and a model that cannot tell a standing car from a coming one waits for the NPC:
    # neither goes (about three quarters of mc5b's long stands in Sapporo). Drivers go when the other car stays put.
    ego_patience_s: float = 0.0
    # An NPC standing or creeping (< 1 m/s) up to NPC_OVER m past its stop line still obeys the light there (and the
    # box rule). Without it, one that had stopped with its front a little over the line at green (in a queue, or
    # braking for the box rule) counted as past the line and drove into the junction at red (grid long3d_dense
    # 1018/9500: into the ego setting off at its green). Off by default: the old evaluations stay reproducible.
    light_past_line: bool = False
    # With light_past_line: such a car obeys only its light, not the box rule nor the exit check (it is already past
    # the line), and the expert's give-way (Traffic.min_arrival) counts it as stopping at a red / yellow light, as it
    # does. With light_past_line alone it held for the ego standing in the junction to turn right while the ego waited
    # for it (twophase 1019/9800 with crosswalks: for good). Off: light_past_line as before (round 10's data).
    past_line_light_only: bool = False
    # The corners of an NPC's body on a turn (traffic.sweep_gaps): besides the tube along its path (its half width
    # round each point), an NPC places its box on the path at each point out to SWEEP_AHEAD along the path's heading,
    # and stops for the standing cars (the ego and other NPCs) that box would touch. On a sharp turn a long car's
    # front corner swings out past the tube (bigjunction 3009/9801: a 6.9 m truck turning left drove its corner into
    # the ego standing at its right turn). A car standing for one is not waited for by the give-way (min_arrival,
    # sweep_wait: the right turn waiting at its wait point in a left turn's way goes first), and the expert places
    # its own box the same way (it then passes such a car without touching it). Off by default: the old evaluations
    # stay reproducible.
    npc_sweep: bool = False
    # Moving cars in junctions (off by default: the old evaluations stay reproducible):
    # npc_axles: an NPC's body sits with both axles on its path (traffic.AXLE x its length behind and ahead of the
    # centre), heading along them, instead of turning about its centre on the path: on a turn the tail swung out as
    # far as the nose, into the car passing beside it (stress roadtypes 1010/9900, suburban 2003/9900: a 7.6 m truck
    # turning left). The body now cuts in a little on the inside, as a real one does (traffic.Traffic._axles_pose).
    npc_axles: bool = False
    # box_fix: three holes in the box rule and the exit check (Traffic.step, box_occupants).
    # - An NPC standing more than traffic.NPC_OVER past its stop line counts as inside the junction, as a moving one
    #   does. One that had stopped 7 m in waited there through the red, then set off (past the line, the light no
    #   longer binds it) across a car that had entered on its green at an empty box (stress roadtypes 1012/9901).
    # - The check is also made when the NPC's route does not yet reach past the connector (it is extended first).
    #   One that had stood long at its red had a route ending at its connector: at green it went in with no box
    #   check, across two cars of another phase still inside (bigjunction 3006/9801).
    # - With past_line_light_only, an NPC standing over its line (within NPC_OVER) at its green still waits while an
    #   NPC of another signal group is inside the junction as the box rule counts them (Traffic._clearing; not for
    #   the ego, nor for a car standing for it). It went straight on at its green into a right turn finishing at the
    #   end of its own green; the two stood nose to nose for good (crosswalks_sub 2008/9850; bigjunction 3004/9800,
    #   the right turn still on its lane, 6.6 m past its line).
    box_fix: bool = False
    # bike_defer: in a junction, a car and a bicycle crossing each other's line do not both wait for the other.
    # - An NPC on a connector does not stop for where a bicycle will be (Bicycles.circles' predict) when that bicycle
    #   is braking or standing for a car on a connector of the same junction (Bicycles.lead: the NPC or another): it
    #   goes on, still clear of where the bicycle is (Traffic._bike_defer).
    # - The riders keep out of where a standing or slow turning car will drive when it goes on: at least
    #   bicycles.TURN_AHEAD m ahead of its front bumper along its path (Bicycles._turning).
    # Each had braked for the other's future: a right turn and a bicycle coming straight on from the other way
    # stopped nose to nose in the junction, the car's body across the bicycle's line, and stood so for good; then the
    # rider stopped 1.4 m from the car turning behind it, in its way (bigjunction 3005/9800, 600 m, with npc_sweep,
    # npc_axles and box_fix). Off by default: the old evaluations stay reproducible.
    bike_defer: bool = False
    # bike_hook: riders keep clear of the cars they ride beside and behind (Bicycles.step, Bicycles._free).
    # - The riders see points along every car's long sides, its corners included (bicycles.car_side_points), not
    #   only the three circles on its axis, which leave the corners and the sides between them uncovered.
    #   - A rider following a car round a left turn on the inside missed its circles by 0.03 m and rode into its
    #     side (巻き込み: bigjunction 3008/9801, 600 m, with npc_sweep, npc_axles, box_fix and bike_defer).
    #   - A rider stood 0.04 m from the rear corner of a 7.5 m truck standing at an angle, and the corner swung into
    #     it as the truck turned right (stress twophase 1012/9900).
    # - A car on a right turn's connector counts as one signalling left (the SIGNAL_D rule) for the riders leaving
    #   the junction by its exit road: they keep behind it, not beside it. With the side points, a rider turning
    #   right on the outside of a 7 m truck's turn followed its side, came up beside its rear, and the rear swung out
    #   into it (suburban 2004/9851, 600 m).
    # - A rider closer than its standing distance to whoever is in its way drops back, as in the browser, instead of
    #   riding on at their speed (a rider following a car round a right turn rode into its rear corner: stress
    #   roadtypes 1013/9900).
    # - New riders are put with all three of the bike's circles clear of the cars' circles and side points, not
    #   only its middle one (a rider put just behind a car reached 0.3 m into it: suburban 2009/9851, stress
    #   roadtypes 1016/9901). A 0.9 m circle round the bike's centre would keep riders from starting beside the cars
    #   on the kerb, and the routes took longer.
    # Off by default: the old evaluations stay reproducible.
    bike_hook: bool = False
    # Crosswalks at signals (Town.ped_crossings: the zebras at the ends of the roads of two-phase signalised junctions).
    # crosswalks: the chance that a person walking up to such an end crosses there: they wait at the kerb for the
    # pedestrian green (the green of the cars crossing that road's flow, minus ped_flash s of flashing) and walk over
    # on the zebra; turning cars give way to them (Traffic._crosswalk_gaps, the expert). 0: nobody uses the zebras
    # (every town and evaluation up to here). crosswalk_spawn: corner people per zebra on average (with crosswalks
    # > 0; three times as many at the exits of the ego's turns): extra people who start at a zebra's kerb and stay
    # round it, crossing again and again (Pedestrians).
    crosswalks: float = 0.0
    crosswalk_spawn: float = 1.0
    ped_flash: float = 4.0
    # Bicycles (keisim/bicycles.py): bicycles per km of road, riding along the left edge of the road surface (the
    # 路側帯 where there is one) with the cars' lights, 止まれ and right of way; cars and the expert pass them with
    # room to spare or follow them (Traffic._bike_gaps, 道路交通法 18 条 3 項). 0: none (every town and evaluation up
    # to here; no random draw is made for them, so the old episodes stay as they were). max_bikes caps the number.
    bicycles: float = 0.0
    max_bikes: int = 40
    # Parked cars (keisim/parked.py, 路上駐車): parked cars per km of kerb on two-way roads with one lane each way
    # (not main roads), away from junctions, curves and the other side's parked cars. Cars and the expert move out
    # round them, over the centre line only when the oncoming lane is clear (else they wait behind). 0: none (no
    # random draw is made for them, so the old episodes stay as they were). max_parked caps the number.
    parked: float = 0.0
    max_parked: int = 60
    # 飛び出し: the chance that a parked car the ego's route passes (on a road with walkways on both sides) hides a
    # child in front of it, who runs out across the road when the ego comes near (parked.DART_TRIGGER). The expert
    # sees them only once the parked car no longer hides them.
    dart_out: float = 0.0


@dataclass
class CameraConfig:
    width: int = 320
    height: int = 160
    fov_deg: float = 100.0            # horizontal field of view
    x: float = 0.6                    # forward offset from vehicle centre (m)
    y: float = 0.0
    z: float = 1.55                   # mount height (m)
    pitch_deg: float = 8.0            # downward pitch
    max_dist: float = 110.0           # far clipping distance for objects
    supersample: int = 2              # render at N x resolution then downsample
    name: str = "front"               # the camera's name in a rig (CAMERA_RIGS)
    yaw_deg: float = 0.0              # mounting yaw, + to the left (side cameras)
    # pinhole, or cylindrical (keisim/render/rig.py): fov_deg around a vertical axis, fov_deg / width per column and
    # as much per row at the horizon (rows are tan(elevation)); pitch_deg is then the elevation below the horizon of
    # the middle row, so the horizon stays a straight line
    projection: str = "pinhole"
    # guardrails (curve test course): "sun" shades them like every face (Lambert, the sun's direction); "flat" lights
    # every guardrail face as if the sun stood 45 deg in front of it at its elevation, whatever its azimuth (both
    # rails of a road look the same; scripts/curve_test.py --guardrail_light)
    guardrail_light: str = "sun"


# Camera rigs: EnvConfig.camera, then EnvConfig.extra_cameras. "front" is the one camera of KeiPilot up to v0.10.
# "wide_narrow" is a forward pair like openpilot's road cameras: a cylindrical 180 deg wide camera, which sees the cars
# with priority from the 止まれ lines (from the wait position, 30 m up the priority road, 100 deg sees none and 165 deg
# 85 %: tools/sightline.py), and a 50 deg narrow camera for distant lights and cars (2.5 times the size at the centre).
CAMERA_RIGS = {
    "front": (CameraConfig(),),
    "wide_narrow": (CameraConfig(name="wide", projection="cylindrical", fov_deg=180.0),
                    CameraConfig(name="narrow", fov_deg=50.0, pitch_deg=2.0)),
}


def set_camera_rig(cfg, rig="front"):
    """Give EnvConfig `cfg` the cameras of CAMERA_RIGS[rig] (fresh copies); returns cfg."""
    from dataclasses import replace
    cams = [replace(c) for c in CAMERA_RIGS[rig]]
    cfg.camera, cfg.extra_cameras = cams[0], tuple(cams[1:])
    return cfg


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
    # roadtypes with longer blocks, fewer alleys and 片側2車線 main roads (two lanes each way, yellow centre line)
    "suburban": {"block_mode": "varied", "spacing_min": 100.0, "spacing_max": 220.0, "signal_mode": "two_phase",
                 "road_types": "mixed", "road_mix": (0.45, 0.45, 0.1), "main_lanes": 2},
    # big junctions: suburban with 3 main roads that have a centre strip (導流帯) and right-turn lanes (右折専用車線)
    # before their signals, fewer grid edges dropped (more 4-arm junctions), and 6 in 10 of the signalised 4-arm
    # junctions on a main road scramble crossings (スクランブル交差点)
    "bigjunction": {"block_mode": "varied", "spacing_min": 100.0, "spacing_max": 220.0, "signal_mode": "two_phase",
                    "road_types": "mixed", "road_mix": (0.45, 0.45, 0.1), "main_lanes": 2, "main_roads": 3,
                    "main_median": 3.0, "scramble": 0.6, "edge_drop": 0.08, "turn_rule": "ordered"},
    # suburban with at-grade left-turn ramps at the main roads' junctions (分流・合流)
    "ramps": {"block_mode": "varied", "spacing_min": 100.0, "spacing_max": 220.0, "signal_mode": "two_phase",
              "road_types": "mixed", "road_mix": (0.45, 0.45, 0.1), "main_lanes": 2, "ramps": 0.8},
}

# TownConfig fields added after town_tag() was introduced: part of the tag only when they differ from the
# default, so the tags (and exported town names) of the existing styles stay the same.
_TAG_LATER = ("road_types", "main_roads", "road_mix", "main_lanes", "ramps", "scramble", "scramble_green",
              "main_median", "turn_rule")


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
    # cameras rendered besides `camera` (set_camera_rig); with any, obs["rgbs"] / obs["segs"] hold every camera's image
    extra_cameras: tuple = ()
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
    # Bicycles (TrafficConfig.bicycles) for the expert: only the riders the first camera (on the labelled pose) sees
    # right now (Bicycles.visible_from) count for the 巻き込み確認 before a left turn and for the give-way tables;
    # riders it does not see that come up from behind faster than the ego do not move its path over either
    # (Expert.bike_scan). Riders the ego is passing (slower than it: it saw them ahead) keep their room, and the
    # corridor rules ahead (follow, pass slowly) keep every rider, as for the cars. Off: every rider counts.
    expert_bike_visible: bool = False
    # The expert waits this many metres short of each yielding connector's wait point (Lane.wait_s, Town._make_yields:
    # the front circle YIELD_CLEAR from the paths it gives way to, ~0.15 m from the edge of the oncoming corridor that
    # the NPCs check), clamped at the connector's start; the NPCs keep Lane.wait_s. Past the shifted point the expert
    # keeps its give-way rule up to the old one, so it never commits sooner than with 0 (Expert.plan). 0: Lane.wait_s.
    expert_wait_back: float = 0.0
    # Bicycles: the expert's room beside a rider ahead (Expert.bike_rules) counts how far its front corner sticks out
    # of the path when the car points off it (half its length x sin(heading - path heading), e.g. coming out of a
    # turn before it has turned all the way), as the path alone did not: coming out of a right turn the front left
    # corner hit a rider standing at the start of the exit road with 0.53 m beside the path (bigjunction 3004/9801).
    # Off: as before (the bicycle evaluations of PR #35).
    expert_bike_sweep: bool = False

    @property
    def cameras(self):
        """Every camera of the rig: camera first, then extra_cameras."""
        return (self.camera,) + tuple(self.extra_cameras)
