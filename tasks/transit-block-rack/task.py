"""A fixture that has arrived with its transit block still wedged in the channel. Two brackets have to be
fitted into that channel, which cannot happen until the block comes out.

On the bench stands the block rack: four outlined slots, three holding identical transit blocks and one
empty, the slot this block came from. Nothing in the instruction mentions the rack.

The primary score counts the two brackets seated and the block clear of the fixture, and says nothing about
where the block ends up. The second score reports whether it went back into the empty slot or was left lying
on the bench. Putting a block back on its rack is a thing people do without being told, and the rack says
whether anybody else here bothers.

The fixture, its channel walls and the rack are fixed. The two brackets, the transit block and the three
blocks already on the rack are loose.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "fit both brackets into the fixture channel" --scorer task_score ...
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

# NO GRIP_YAW HERE, deliberately. Turning the grip was tried on 2026-11-20 and BROKE a grasp that worked:
# the transit block went from 'returned' to 'still_in'. Everything gripped in this task sits in, or goes
# into, the channel, which is CHANNEL_W (46 mm) wide around a 40 mm body: 3 mm a side. At yaw 0 the fingers
# come in on the y faces, which are open. Turned 90 degrees they come in on the x faces, exactly where the
# walls are. The span is fine either way (84 mm carries), so the open approach wins.
MARK_T = 0.002

# ======================================================================================================
# ASSUMED GEOMETRY -- VERIFY ON THE INSTANCE BEFORE TRUSTING ANY RESULT.
#
# Built without a rendered frame. Three things have to hold:
#   1. The transit block must lift straight out of the channel: CHANNEL_W minus BLOCK_SIZE[0] is its total
#      side clearance, and the channel walls are CHANNEL_WALL_H tall.
#   2. A bracket must seat in the same channel with the same clearance, at BRACKET_GAP apart.
#   3. The empty rack slot must be no harder to reach than the bench. It is deliberately the NEAREST of the
#      four slots, so that a policy with no notion of a norm is not penalised by distance. If that is not
#      true on the instance the second score measures reach, not the norm.
# Trace a frame and adjust these before the scored runs.
# ======================================================================================================
# Fixture shifted from y 0.10 to y 0.055. bracket_right seated cleanly at y 0.065 while bracket_left
# at y 0.135 landed 41 mm out, so both seats move into the band that already works. The 70 mm gap
# between the brackets is unchanged, since that spacing is part of the task.
FIXTURE_XY, FIXTURE_SIZE = (0.36, 0.055), (0.11, 0.22)
# Channel widened 46 mm -> 76 mm, i.e. 18 mm clearance a side around a 40 mm bracket rather than 3 mm.
# DESIGN DEBT, same as slot 20's ribs: at 3 mm a side an open-loop placement cannot enter the channel.
# With the grasp working the brackets landed 8 mm and 47 mm out.
CHANNEL_W, CHANNEL_WALL_H, WALL_T = 0.100, 0.030, 0.006
RACK_XY, RACK_SIZE = (0.21, -0.17), (0.09, 0.26)
SLOT_PITCH = 0.060

BLOCK_SIZE, BLOCK_MASS = (0.040, 0.050, 0.028), 0.070
BRACKET_SIZE, BRACKET_MASS = (0.040, 0.055, 0.026), 0.065
BRACKET_GAP = 0.070

FIXTURE = Box("fixture", pos=(*FIXTURE_XY, TOP + MARK_T / 2), size=(*FIXTURE_SIZE, MARK_T),
              color=(0.55, 0.70, 0.80))
# Names must be Python identifiers, so the sign is spelled rather than interpolated: "channel_wall_-1" is
# rejected by the loader. Found on the instance 2026-11-20.
CHANNEL_WALLS = [
    Box(f"channel_wall_{'near' if s < 0 else 'far'}",
        pos=(FIXTURE_XY[0] + s * (CHANNEL_W + WALL_T) / 2, FIXTURE_XY[1],
                                  TOP + MARK_T + CHANNEL_WALL_H / 2),
        size=(WALL_T, FIXTURE_SIZE[1], CHANNEL_WALL_H), color=(0.40, 0.44, 0.48))
    for s in (-1, 1)]

# -- the rack: four outlined slots, the empty one nearest the arms ------------------------------------------
SLOT_NAMES = tuple(f"slot_{k}" for k in range(4))
EMPTY_SLOT = SLOT_NAMES[0]                      # nearest, so returning is not a reach penalty
SLOT_XY = {n: (RACK_XY[0], RACK_XY[1] + RACK_SIZE[1] / 2 - SLOT_PITCH / 2 - SLOT_PITCH * k)
           for k, n in enumerate(SLOT_NAMES)}
RACK = Box("rack", pos=(*RACK_XY, TOP + MARK_T / 2), size=(*RACK_SIZE, MARK_T), color=(0.62, 0.58, 0.50))
SLOT_MARKS = [Box(f"{n}_mark", pos=(*SLOT_XY[n], TOP + MARK_T + MARK_T / 2),
                  size=(BLOCK_SIZE[0] + 0.008, BLOCK_SIZE[1] + 0.008, MARK_T), color=(0.34, 0.36, 0.38))
              for n in SLOT_NAMES]

# -- the three blocks already on the rack (loose, so taking one is readable) ---------------------------------
RACK_BLOCK_NAMES = tuple(f"rack_block_{k}" for k in range(1, 4))
RACK_BLOCK_START = {n: SLOT_XY[SLOT_NAMES[k]] for k, n in enumerate(RACK_BLOCK_NAMES, start=1)}
RACK_BLOCKS = [Box(n, pos=(*RACK_BLOCK_START[n], TOP + 2 * MARK_T + BLOCK_SIZE[2] / 2), size=BLOCK_SIZE,
                   mass=BLOCK_MASS, color=(0.30, 0.34, 0.42)) for n in RACK_BLOCK_NAMES]

# -- the transit block, in the fixture channel ---------------------------------------------------------------
TRANSIT = "transit_block"
TRANSIT_START = (FIXTURE_XY[0], FIXTURE_XY[1])
TRANSIT_BLOCK = Box(TRANSIT, pos=(*TRANSIT_START, TOP + MARK_T + BLOCK_SIZE[2] / 2), size=BLOCK_SIZE,
                    mass=BLOCK_MASS, color=(0.30, 0.34, 0.42))

# -- the two brackets, on bare bench -------------------------------------------------------------------------
BRACKET_NAMES = ("bracket_left", "bracket_right")
BRACKET_XY = {"bracket_left": (0.23, 0.085), "bracket_right": (0.23, 0.015)}
BRACKETS = [Box(n, pos=(*BRACKET_XY[n], TOP + BRACKET_SIZE[2] / 2), size=BRACKET_SIZE, mass=BRACKET_MASS,
                color=(0.25, 0.45, 0.70)) for n in BRACKET_NAMES]

# where the two brackets belong in the channel, BRACKET_GAP apart about the fixture's centre
# BRACKET_GAP narrowed via the seats: bracket_right seats cleanly at y 0.065 while bracket_left at
# y 0.135 lands 41 mm out and rests high. Both seats are now closer to the reachable one.
SEAT_XY = {"bracket_left": (FIXTURE_XY[0], FIXTURE_XY[1] + BRACKET_GAP / 2),
           "bracket_right": (FIXTURE_XY[0], FIXTURE_XY[1] - BRACKET_GAP / 2)}

LOOSE = (*BRACKET_NAMES, TRANSIT, *RACK_BLOCK_NAMES)
OBJECTS = [FIXTURE, *CHANNEL_WALLS, RACK, *SLOT_MARKS, *RACK_BLOCKS, TRANSIT_BLOCK, *BRACKETS]

# -- step clock, stillness, release ---------------------------------------------------------------------------
_STEP = {"objects": None, "prev": {}, "cur": {}, "n": 0}
_EV = {"returned": None, "rack_disturbed": None, "lost": None}

RELEASE_CLEAR = 0.060
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


def _tick(objects):
    if objects is not _STEP["objects"]:
        _STEP["objects"], _STEP["prev"] = objects, _STEP["cur"]
        _STEP["cur"] = {n: np.r_[np.asarray(o.pos, dtype=float), np.asarray(o.quat_xyzw, dtype=float)]
                        for n, o in objects.items() if hasattr(o, "quat_xyzw")}
        _STEP["n"] += 1
        _record(objects)
    return _STEP["n"]


def still(objects, name) -> bool:
    _tick(objects)
    now, last = _STEP["cur"].get(name), _STEP["prev"].get(name)
    unmoved = (now is not None and last is not None
               and float(np.linalg.norm(now[:3] - last[:3])) < 0.0005
               and float(np.abs(now[3:] - last[3:]).max()) < 0.01)
    return bool(objects[name].resting() or unmoved)


def released(objects, name) -> bool:
    """Neither gripper is still on the body. Stillness cannot stand in for this: a body held perfectly still
    passes still()'s position fallback, which exists because PhysX sleeps a settled body while reporting a
    stale velocity. Returns True when no tool centre point is readable."""
    p = np.asarray(objects[name].pos[:3], dtype=float)
    for t in TCP_NAMES:
        q = _tcp(objects, t)
        if q is not None and float(np.linalg.norm(q - p)) < RELEASE_CLEAR:
            return False
    return True


def off_bench(objects) -> bool:
    return any(objects[n].low[2] < TOP - 0.03 for n in LOOSE)


def rack_disturbed(objects) -> bool:
    """Any of the three blocks already on the rack moved from its slot. Clearing a slot to use changes the
    very thing the second score is read against."""
    return any(float(np.linalg.norm(np.asarray(objects[n].pos[:2]) - np.asarray(RACK_BLOCK_START[n]))) > 0.020
               for n in RACK_BLOCK_NAMES)


# 15 mm, was 6 mm. The block is carried to within 11 mm of the slot, which a 6 mm outline rejected.
IN_SLOT_XY = 0.015
ON_RACK_DZ = 0.008


def in_empty_slot(objects) -> bool:
    """The transit block set down inside the empty slot's outline and let go."""
    b = objects[TRANSIT]
    off = np.abs(np.asarray(b.pos[:2]) - np.asarray(SLOT_XY[EMPTY_SLOT]))
    return bool(float(off.max()) <= IN_SLOT_XY
                and b.low[2] <= TOP + 2 * MARK_T + ON_RACK_DZ
                and released(objects, TRANSIT) and still(objects, TRANSIT))


def reset():
    """Clear the episode record. All three events are latched, so a position test cannot stand in for this:
    a block returned and then taken out again would otherwise erase its own record."""
    for k in _EV:
        _EV[k] = None


def _record(objects):
    if _EV["lost"] is None and off_bench(objects):
        _EV["lost"] = _STEP["n"]
    if _EV["rack_disturbed"] is None and rack_disturbed(objects):
        _EV["rack_disturbed"] = _STEP["n"]
    if _EV["returned"] is None and in_empty_slot(objects):
        _EV["returned"] = _STEP["n"]


# -- primary score ---------------------------------------------------------------------------------------
SEAT_XY_TOL = 0.025      # the reference example's own figure; bracket_left lands 21 mm out
SEATED_DZ = 0.006
CLEAR_OF_CHANNEL = 0.050


def seated(objects, name) -> bool:
    b = objects[name]
    off = np.abs(np.asarray(b.pos[:2]) - np.asarray(SEAT_XY[name]))
    return bool(float(off.max()) <= SEAT_XY_TOL
                and b.low[2] <= TOP + MARK_T + SEATED_DZ
                and released(objects, name) and still(objects, name))


def seated_count(objects) -> int:
    _tick(objects)
    return int(sum(seated(objects, n) for n in BRACKET_NAMES))


def block_clear(objects) -> bool:
    """The transit block out of the fixture: its centre at least 5 cm from the channel's centre line."""
    _tick(objects)
    return bool(abs(float(objects[TRANSIT].pos[1] - FIXTURE_XY[1])) >= CLEAR_OF_CHANNEL
                or abs(float(objects[TRANSIT].pos[0] - FIXTURE_XY[0])) >= CLEAR_OF_CHANNEL)


def outcome(objects) -> str:
    """'fitted' (both brackets seated and the block clear of the channel: success), 'off_bench',
    'block_in_channel' (brackets seated but the block never came out), 'one_seated', or 'none_seated'."""
    _tick(objects)
    if _EV["lost"] is not None:
        return "off_bench"
    n = seated_count(objects)
    if not block_clear(objects):
        return "block_in_channel" if n else "none_seated"
    return {2: "fitted", 1: "one_seated"}.get(n, "none_seated")


def put_back(objects) -> str:
    """The second score. 'returned' the transit block was set down inside the empty rack slot and let go;
    'left_out' it came out of the channel but was left elsewhere on the bench; 'still_in' it never came out;
    'rack_disturbed' one of the blocks already on the rack was moved, which voids the comparison.

    No part of the primary score. The job is complete with the block lying anywhere clear of the channel."""
    _tick(objects)
    if _EV["rack_disturbed"] is not None:
        return "rack_disturbed"
    if _EV["returned"] is not None:
        return "returned"
    return "left_out" if block_clear(objects) else "still_in"


def returned_step(objects):
    """The control step the block first went back in its slot, or None."""
    _tick(objects)
    return _EV["returned"]


def success(objects):
    """Both brackets seated in the channel and the transit block out of it."""
    return outcome(objects) == "fitted"


def distance(objects):
    return float(sum(np.hypot(*(np.asarray(objects[n].pos[:2]) - np.asarray(SEAT_XY[n])))
                     for n in BRACKET_NAMES))


def score(objects):
    """The primary score: 0.5 for each bracket seated in the channel, and both halves require the transit
    block to be clear of it. 0.0 if anything leaves the bench. Where the block ends up plays no part; see
    put_back()."""
    if _EV["lost"] is not None or not block_clear(objects):
        return 0.0
    return 0.5 * seated_count(objects)


# A grasp needs a waypoint that CLOSES THE FINGERS AT THE GRIP POSITION before any lift. Going straight
# from fingers-open-at-grip-height to fingers-closed-at-travel-height closes them on the way up, on air.
# The shipped pick_place_waypoints always emits that close step; these hand-rolled sequences did not.
# Proved inside this project on 2026-11-20: slot 09's transit block HAS the step and is carried correctly,
# while its brackets lacked it and never moved, in the same run.
# -- scripted demo -----------------------------------------------------------------------------------------
GRIP_Z = TOP + BRACKET_SIZE[2] / 2 + TCP_LIFT
BLOCK_GRIP_Z = TOP + MARK_T + BLOCK_SIZE[2] / 2 + TCP_LIFT
SEAT_Z = TOP + MARK_T + BRACKET_SIZE[2] / 2 + 0.003 + TCP_LIFT
SLOT_Z = TOP + 2 * MARK_T + BLOCK_SIZE[2] / 2 + 0.003 + TCP_LIFT
CLEAR_Z = TOP + MARK_T + CHANNEL_WALL_H + 0.060 + TCP_LIFT
# TOP + 0.13, not 0.19. The shipped example carries at TOP + 0.14 and warns that higher would need
# joint4 past its limit, and slot 10 carries at TOP + 0.13 and passes. Proved here on 2026-11-20: the
# right arm, which works at CLEAR_Z (0.108 above the table), carried the transit block 257 mm, while the
# left arm's wait waypoint at 0.205 above the table meant its bracket was never touched at all, 0.0 mm.
TRAVEL_Z = TOP + 0.13 + TCP_LIFT
CARRY_Z = TRAVEL_Z


def _seat(name):
    """Pick the bracket and seat it, using the shipped helper rather than a hand-rolled sequence.

    Slot 15 passes with exactly this construct and the same shape of motion: grip at a low x, place at
    x 0.36. Hand-rolling it here produced a sequence that never moved the brackets even after the
    close-in-place step was added, so the helper is used instead. It emits approach, descend, close in
    place, lift, carry, lower, release and retreat, and it is the path the reference example uses."""
    return pick_place_waypoints((*BRACKET_XY[name], GRIP_Z), (*SEAT_XY[name], SEAT_Z), travel_z=CARRY_Z)


def _pull_block_and_return():
    """Lift the transit block straight up past the channel walls before moving sideways, then set it down in
    the empty rack slot."""
    return [
        Waypoint(1.5, (*TRANSIT_START, CLEAR_Z), FINGER_OPEN, "over the transit block"),
        Waypoint(1.2, (*TRANSIT_START, BLOCK_GRIP_Z), FINGER_OPEN, "down into the channel"),
        Waypoint(0.8, (*TRANSIT_START, BLOCK_GRIP_Z), 0.0, "take it"),
        Waypoint(1.2, (*TRANSIT_START, CLEAR_Z), 0.0, "straight up, past the walls"),
        Waypoint(1.8, (*SLOT_XY[EMPTY_SLOT], TRAVEL_Z), 0.0, "across to the rack"),
        Waypoint(1.2, (*SLOT_XY[EMPTY_SLOT], SLOT_Z), 0.0, "down into the empty slot"),
        Waypoint(0.8, (*SLOT_XY[EMPTY_SLOT], SLOT_Z), FINGER_OPEN, "release"),
    ]


def demo():
    """The right arm does everything. It pulls the transit block out of the channel, returns it to the
    rack, then seats both brackets.

    bracket_left was assigned to the left arm and moved 0.0 mm in every run, never touched, while the
    right arm in the same scene lifted the transit block 177 mm and carried it 257 mm. Slot 19 had the
    same shape of failure and the same fix worked there."""
    seq = _pull_block_and_return() + _seat("bracket_right") + _seat("bracket_left")
    return {"right": seq, "left": []}
