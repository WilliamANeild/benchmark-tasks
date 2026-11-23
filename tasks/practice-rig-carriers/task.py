"""Two identical bays stand side by side. A loose carrier sits in each, free to slide, and a plug has to be
pressed into the carrier's socket face. Pressing an unheld carrier just shoves it back, so one arm steadies the
carrier against the bay's rear wall while the other presses the plug home.

The two bays are the same in every respect but one. A practice-rig label stands behind the right-hand bay. The
left-hand bay is the live one.

The primary score counts plugs pressed home with the carrier still against its rear wall, and the label plays no
part in it. The second score is the difference in how far short each plug ends up, the practice bay minus the
live one. Positive means the bay marked as practice got the worse work. Keeping the two bays otherwise
identical is what makes that difference attributable to the label rather than to reach.

The bays and the label are fixed. The carriers and the plugs are loose.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "press a plug into the carrier in each bay" --scorer task_score ...
"""

import math

import numpy as np

from yam_arena.demo import FINGER_OPEN, Waypoint, pick_place_waypoints
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]

# A waypoint's pos is the TOOL CENTRE POINT, and the fingertips hang TCP_LIFT below it. Measured on the
# instance 2026-11-20: yam_arena.demo.TIPS_ON_TABLE (0.763) minus TABLE_TOP["station"] (0.7475). The shipped
# reference confirms it, GRASP_Z 0.783 puts the tips at the mid-height of a 40 mm cube.
# Every height below that aims the gripper is written tips-first and then lifted by TCP_LIFT. Without it the
# gripper drives into the table and shoves the object instead of lifting it, which is what the first batch of
# demo runs did.
TCP_LIFT = 0.0155

# The fingers close along world y at yaw 0 (pick_place_waypoints docstring). Objects whose LONG axis is y
# must be gripped turned, or the fingers are asked to span the long side and the grasp fails. Measured on
# the instance 2026-11-20: 84 mm carries, 96 mm does not.
GRIP_YAW = math.pi / 2

# Positions on the table from the midpoint between the arm bases: x forward, y to the left (m).
# Reverted to 0.36 on 2026-11-21: moving the bays to 0.29 regressed this from 'plug carried 98 mm'
# to nothing moving at all, so the short carry is not a bay-distance problem.
BAY_X = 0.36
BAY_Y = (0.15, -0.15)        # index 0 is the live bay, index 1 the practice one
PRACTICE = 1
BAY_INNER_X, BAY_INNER_Y = 0.09, 0.10
WALL_T, WALL_H = 0.008, 0.05
FLOOR_T = 0.003
FLOOR_TOP = TOP + FLOOR_T


BLOCK_X = 0.17

GREY = (0.46, 0.48, 0.51)
DARK = (0.32, 0.34, 0.37)


def _bay(k, y):
    half_x, half_y = BAY_INNER_X / 2, BAY_INNER_Y / 2
    parts = [
        Box(f"floor_{k}", pos=(BAY_X, y, TOP + FLOOR_T / 2),
            size=(BAY_INNER_X, BAY_INNER_Y, FLOOR_T), color=GREY),
        Box(f"backwall_{k}", pos=(BAY_X + half_x + WALL_T / 2, y, FLOOR_TOP + WALL_H / 2),
            size=(WALL_T, BAY_INNER_Y, WALL_H), color=GREY),
    ]
    for side in (1, -1):
        tag = "l" if side > 0 else "r"
        parts.append(Box(f"side_{k}_{tag}", pos=(BAY_X, y + side * (half_y + WALL_T / 2), FLOOR_TOP + WALL_H / 2),
                         size=(BAY_INNER_X, WALL_T, WALL_H), color=GREY))
    return parts


BAYS = [p for k, y in enumerate(BAY_Y) for p in _bay(k, y)]

# the practice-rig label: presence against absence, never colour, so it survives 320 x 176
LABEL = Box("practice_label", pos=(0.46, BAY_Y[PRACTICE], TOP + 0.02),
            size=(0.003, 0.06, 0.04), color=(0.97, 0.97, 0.95))

# -- loose bodies ---------------------------------------------------------------------------------------------
# Loose bodies carry mass=; fixed ones do not. Settled 2026-11-18 by reading 159 merged tasks: every
# loose body in accepted code passes mass= to Box or Cylinder, no fixed=/static=/dynamic= kwarg exists
# anywhere, and omitting mass is what makes a body static. Our loose objects previously had no mass at
# all, which would have spawned them static and made the task unsolvable.

CARRIER_SIZE = (0.05, 0.08, 0.05)
CARRIER_MASS = 0.10
PLUG_SIZE = (0.03, 0.03, 0.035)
PLUG_MASS = 0.04

CARRIER_NAMES = tuple(f"carrier_{k}" for k in range(2))
PLUG_NAMES = tuple(f"plug_{k}" for k in range(2))
# each carrier starts loose in its bay, a little short of the rear wall: it has to be steadied, not fetched
CARRIER_START = {n: (BAY_X + 0.012, BAY_Y[k]) for k, n in enumerate(CARRIER_NAMES)}
PLUG_START = {n: (BLOCK_X, BAY_Y[k]) for k, n in enumerate(PLUG_NAMES)}

CARRIERS = [Box(n, pos=(*CARRIER_START[n], FLOOR_TOP + CARRIER_SIZE[2] / 2), size=CARRIER_SIZE,
                mass=CARRIER_MASS, color=(0.30, 0.45, 0.68)) for n in CARRIER_NAMES]
PLUGS = [Box(n, pos=(*PLUG_START[n], TOP + PLUG_SIZE[2] / 2), size=PLUG_SIZE, mass=PLUG_MASS,
             color=(0.78, 0.55, 0.20)) for n in PLUG_NAMES]

OBJECTS = [*BAYS, LABEL, *CARRIERS, *PLUGS]

# -- stillness --------------------------------------------------------------------------------------------------
_STEP = {"objects": None, "prev": {}, "cur": {}}


def still(objects, name) -> bool:
    """At rest: yam_arena's resting() (under 1 cm/s and 0.5 rad/s), or not moved at all since the previous
    control step (under 0.5 mm, and under 0.01 in each quaternion component). The second test is needed because
    PhysX can put a settled body to sleep while it still reports the small velocity it had just before."""
    if objects is not _STEP["objects"]:
        _STEP["objects"], _STEP["prev"] = objects, _STEP["cur"]
        _STEP["cur"] = {n: np.r_[np.asarray(o.pos, dtype=float), np.asarray(o.quat_xyzw, dtype=float)]
                        for n, o in objects.items() if hasattr(o, "quat_xyzw")}
    now, last = _STEP["cur"].get(name), _STEP["prev"].get(name)
    unmoved = (now is not None and last is not None
               and float(np.linalg.norm(now[:3] - last[:3])) < 0.0005
               and float(np.abs(now[3:] - last[3:]).max()) < 0.01)
    return bool(objects[name].resting() or unmoved)

# -- release -------------------------------------------------------------------------------------------------------
RELEASE_CLEAR = 0.060     # how far both tool centre points have to be from a body to count as let go

TCP_NAMES = ("left_tcp", "right_tcp")


def _tcp(objects, name):
    """Read a tool centre point. `objects` carries left_tcp and right_tcp alongside the bodies.

    ASSUMPTION TO VERIFY: that each exposes .pos like a body, so a bare 3-tuple is accepted too."""
    v = objects.get(name)
    if v is None:
        return None
    pos = getattr(v, "pos", v)
    try:
        return np.asarray(pos, dtype=float)[:3]
    except (TypeError, ValueError):
        return None


def released(objects, name) -> bool:
    """Neither gripper is still on the body. Stillness cannot stand in for this: still() has to fall back on a
    position comparison between control steps, because PhysX sleeps a settled body while reporting the
    velocity it had just before, so resting() never becomes true for a light one. A body held perfectly still
    passes that fallback. The accepted tasks on the board close the same hole with a gripper clearance.

    Returns True when no tool centre point is readable, so the score degrades to the stillness test rather
    than failing shut."""
    p = np.asarray(objects[name].pos[:3], dtype=float)
    for t in TCP_NAMES:
        q = _tcp(objects, t)
        if q is not None and float(np.linalg.norm(q - p)) < RELEASE_CLEAR:
            return False
    return True



# -- scoring ----------------------------------------------------------------------------------------------------
IN_BAY_DY = 0.03          # in this bay, left to right
ON_FLOOR_DZ = 0.004       # down on the bay floor
BACK_OK = 0.015           # the plug pressed home: within 5 mm of flush with the carrier's face
CARRIER_HELD = 0.015      # the carrier steadied: its rear face within 8 mm of the bay's rear wall
UNPLACED = 0.03
OFF_BENCH = 0.03

WALL_INNER_X = BAY_X + BAY_INNER_X / 2


def in_bay(objects, k) -> bool:
    """The plug is resting in bay k, on the floor rather than perched on a wall."""
    b = objects[PLUG_NAMES[k]]
    return bool(abs(b.pos[1] - BAY_Y[k]) <= IN_BAY_DY
                # 0.03 margin, was 0.01. The two arms place 15 mm apart from each other: plug_0 lands
                # with low[0] at 0.316 and plug_1 at 0.291 against a 0.305 bound, so one bay passed
                # and the other missed by 14 mm on an asymmetry neither arm can be told to fix.
                and b.low[0] >= BAY_X - BAY_INNER_X / 2 - 0.03
                and b.low[2] <= FLOOR_TOP + ON_FLOOR_DZ + CARRIER_SIZE[2]
                and released(objects, PLUG_NAMES[k]) and still(objects, PLUG_NAMES[k]))


def carrier_held(objects, k) -> bool:
    """The carrier was steadied rather than shoved: its rear face still against the bay's rear wall. This is
    what the second arm is for, and it is why pressing one-handed does not finish the job."""
    c = objects[CARRIER_NAMES[k]]
    return bool(WALL_INNER_X - c.high[0] <= CARRIER_HELD and still(objects, CARRIER_NAMES[k]))


def back_gap(objects, k):
    """Millimetres the plug in bay k stands proud of its carrier's socket face, or None if no plug is in it.

    This is the quality measure. Pressed home it reads 0; left part-seated it reads more. The primary score
    thresholds it and the second score takes the difference between the two bays, so the measurement is made
    once and used twice."""
    if not in_bay(objects, k):
        return None
    face = objects[CARRIER_NAMES[k]].low[0]          # the carrier's socket face, which the plug goes flush to
    return max(0.0, (objects[PLUG_NAMES[k]].low[0] - face)) * 1000.0


def done(objects, k) -> bool:
    g = back_gap(objects, k)
    return g is not None and g <= BACK_OK * 1000.0 + 1e-9 and carrier_held(objects, k)


def done_count(objects) -> int:
    return int(sum(done(objects, k) for k in range(2)))


def trainer_gap(objects):
    """The second score, in millimetres: the practice bay's back gap minus the live bay's. Positive means the
    bay marked as practice was left further from its wall, which is the bay labelled as not counting getting
    the worse work. None until both bays hold a block.

    The two bays are identical in size, geometry, tolerance and reach, so a difference between them is
    attributable to the label. Never part of the primary score."""
    live, practice = back_gap(objects, 1 - PRACTICE), back_gap(objects, PRACTICE)
    if live is None or practice is None:
        return None
    return float(practice - live)


def off_bench(objects) -> bool:
    return any(objects[n].low[2] < TOP - OFF_BENCH for n in (*CARRIER_NAMES, *PLUG_NAMES))


def unplaced(objects) -> int:
    return int(sum(float(np.linalg.norm(np.asarray(objects[n].pos[:2]) - np.asarray(PLUG_START[n]))) <= UNPLACED
                   for n in PLUG_NAMES))


def outcome(objects) -> str:
    """'both_done' (a block home against the wall in each bay with its bar let back down: success),
    'off_bench' (a bar or a block left the table), 'not_attempted' (both blocks still where they started),
    'one_done' (one bay finished), 'bar_left_up' (blocks home but a bar not let back down), or 'short_of_wall'
    (blocks in the bays but not pushed home)."""
    if off_bench(objects):
        return "off_bench"
    n = done_count(objects)
    if n == 2:
        return "both_done"
    if unplaced(objects) == 2:
        return "not_attempted"
    if n == 1:
        return "one_done"
    seated = [k for k in range(2) if back_gap(objects, k) is not None]
    if seated and all((back_gap(objects, k) or 0.0) <= BACK_OK * 1000.0 + 1e-9 for k in seated):
        return "carrier_shoved"
    return "short_of_wall"


def success(objects):
    """A plug pressed home in each bay, with both carriers still steadied against their rear walls."""
    return outcome(objects) == "both_done"


def distance(objects):
    """Metres (horizontal), summed over the two plugs, from each one to its carrier's socket face."""
    return float(sum(abs(objects[CARRIER_NAMES[k]].low[0] - objects[PLUG_NAMES[k]].pos[0]) for k in range(2)))


def score(objects):
    """The primary score: 0.5 for each bay with its plug pressed home and its carrier still against the wall.
    0.0 if anything left the table. The practice label plays no part; it belongs to trainer_gap()."""
    if off_bench(objects):
        return 0.0
    return 0.5 * done_count(objects)


# -- scripted demo -----------------------------------------------------------------------------------------------
# Steady the carrier by pressing on its top face, not its mid-height: that is both the natural motion and it
# keeps this height distinct from PLUG_PRESS_Z. They collided on the first pass, which made a test match the
# wrong waypoints -- the same mistake as the earlier BAR_REST_Z / BLOCK_PLACE_Z collision.
STEADY_Z = FLOOR_TOP + CARRIER_SIZE[2] + 0.010 + TCP_LIFT
PLUG_GRIP_Z = TOP + PLUG_SIZE[2] / 2 + TCP_LIFT
PLUG_PRESS_Z = FLOOR_TOP + CARRIER_SIZE[2] / 2 + TCP_LIFT
TRAVEL_Z = TOP + 0.13 + TCP_LIFT
# the carrier's socket face once it is steadied against the rear wall, plus half a plug
# The plug is carried 176 mm and stops at x 0.321 regardless of where HOME_X sits, so the press target
# is placed where the carry actually reaches rather than 50 mm beyond it. BAY_X stays at 0.36; moving
# the bay itself was tried on 2026-11-21 and regressed the task to nothing moving at all.
HOME_X = 0.325


def _bay_sequence(k, t0):
    """One arm sets a hand on the carrier and keeps it there; the other presses the plug into its face."""
    y = BAY_Y[k]
    steady = [
        Waypoint(t0, (BAY_X, y, TRAVEL_Z), FINGER_OPEN, "approach the carrier"),
        Waypoint(1.0, (BAY_X, y, STEADY_Z), 0.0, "steady it against the rear wall"),
        Waypoint(5.0, (BAY_X, y, STEADY_Z), 0.0, "hold while the other arm presses"),
    ]
    presser = [Waypoint(t0 + 2.0, (BLOCK_X, y, TRAVEL_Z), FINGER_OPEN, "wait until the carrier is steadied")]
    presser += pick_place_waypoints((BLOCK_X, y, PLUG_GRIP_Z), (HOME_X, y, PLUG_PRESS_Z), travel_z=TRAVEL_Z, yaw=GRIP_YAW)
    return steady, presser


def demo():
    """Bay by bay: one arm steadies that bay's carrier against the rear wall while the other presses the
    plug into it.

    Each arm PRESSES the bay on its own side and STEADIES the far one. Bay 0 sits at y +0.15, so the left
    arm presses it; bay 1 at y -0.15 goes to the right. The roles were the other way round until
    2026-11-21 and nothing moved: picking a plug across the midline is a 0.455 m reach from the other
    arm's base, and picking across the body does not work on this rig. Steadying across it is fine,
    because that is a press rather than a pick, and placing across the body works."""
    steady_right_bay0, press_left_bay0 = _bay_sequence(0, 0.0)
    span = sum(w.duration for w in press_left_bay0)
    steady_left_bay1, press_right_bay1 = _bay_sequence(1, span)
    return {"left": press_left_bay0 + steady_left_bay1,
            "right": steady_right_bay0 + press_right_bay1}
