"""Scorer tests for the transit-block task. No GPU, no Isaac Sim, no yam_arena.

    python -m pytest -q tests

The point under test is that the primary score is indifferent to where the block ends up, while the second
score separates returning it, dumping it, never removing it, and raiding the rack for a slot.
"""

import importlib, sys, types
from pathlib import Path

import numpy as np
import pytest

TABLE_TOP_Z = 0.80


def _stubs():
    tf = types.ModuleType("yam_arena.task_file"); tf.TABLE_TOP = {"station": TABLE_TOP_Z}

    class Box:
        def __init__(self, name, pos, size, mass=None, color=None):
            self.name, self.pos, self.size, self.color, self.mass = name, pos, size, color, mass

    tf.Box = tf.Asset = Box
    dm = types.ModuleType("yam_arena.demo"); dm.FINGER_OPEN = 1.0

    class Waypoint:
        """Matches the real signature: (duration, pos, finger, label='', yaw=0.0).
        Verified on the instance 2026-11-20; the old stub had `comment` and no `yaw`."""
        def __init__(self, duration, pos, finger, label="", yaw=0.0):
            self.duration, self.pos, self.finger, self.label, self.yaw = duration, pos, finger, label, yaw
            self.comment = label

    dm.Waypoint = Waypoint
    def pick_place_waypoints(pick, place, travel_z, yaw=0.0, place_yaw=None, speed=1.0):
        """Mirrors the real helper's eight steps, including the close IN PLACE at the grip position."""
        k = 1.0 / speed
        ey = yaw if place_yaw is None else place_yaw
        return [Waypoint(1.5 * k, (pick[0], pick[1], travel_z), 1.0, "to the object", yaw),
                Waypoint(1.5 * k, pick, 1.0, "descend", yaw),
                Waypoint(0.8 * k, pick, 0.0, "close", yaw),
                Waypoint(1.2 * k, (pick[0], pick[1], travel_z), 0.0, "lift", yaw),
                Waypoint(1.8 * k, (place[0], place[1], travel_z), 0.0, "carry", ey),
                Waypoint(1.2 * k, place, 0.0, "lower", ey),
                Waypoint(0.6 * k, place, 1.0, "release", ey),
                Waypoint(1.0 * k, (place[0], place[1], travel_z), 1.0, "retreat", ey)]

    dm.pick_place_waypoints = pick_place_waypoints
    pkg = types.ModuleType("yam_arena"); pkg.task_file, pkg.demo = tf, dm
    for n, m in (("yam_arena", pkg), ("yam_arena.task_file", tf), ("yam_arena.demo", dm)):
        sys.modules.setdefault(n, m)


_stubs()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
task = importlib.import_module("task")

T = TABLE_TOP_Z
DUMP_XY = (0.26, -0.05)          # bare bench, clear of the channel and not a slot


class Body:
    def __init__(self, pos, size, at_rest=True):
        self.pos, self.size = tuple(map(float, pos)), tuple(map(float, size))
        self.quat_xyzw = (0.0, 0.0, 0.0, 1.0); self._at_rest = at_rest

    @property
    def low(self):  return tuple(p - s / 2 for p, s in zip(self.pos, self.size))

    @property
    def high(self): return tuple(p + s / 2 for p, s in zip(self.pos, self.size))

    def resting(self): return self._at_rest


class Tcp:
    """What the engine puts in objects["left_tcp"]: a thing with .pos and no quat."""
    def __init__(self, pos): self.pos = tuple(map(float, pos))


def frame(brackets=("start", "start"), block="channel", rack_shift=0.0, rest=True):
    """One control step.

    brackets entries: 'start' | 'seated' | 'off'
    block:            'channel' | 'slot' | 'dumped' | 'held_slot' | 'near_slot' | 'off'
    """
    o = {}
    for k, n in enumerate(task.BRACKET_NAMES):
        w = brackets[k]
        if w == "seated":
            pos = (*task.SEAT_XY[n], T + task.MARK_T + task.BRACKET_SIZE[2] / 2)
        elif w == "off":
            pos = (0.30, 0.46, T - 0.12)
        else:
            pos = (*task.BRACKET_XY[n], T + task.BRACKET_SIZE[2] / 2)
        o[n] = Body(pos, task.BRACKET_SIZE, rest)

    slot = task.SLOT_XY[task.EMPTY_SLOT]
    on_rack_z = T + 2 * task.MARK_T + task.BLOCK_SIZE[2] / 2
    bp = {"channel": (*task.TRANSIT_START, T + task.MARK_T + task.BLOCK_SIZE[2] / 2),
          "slot": (*slot, on_rack_z),
          "held_slot": (*slot, on_rack_z),
          "near_slot": (slot[0] + task.IN_SLOT_XY + 0.004, slot[1], on_rack_z),
          "dumped": (*DUMP_XY, T + task.BLOCK_SIZE[2] / 2),
          "off": (0.30, 0.46, T - 0.12)}[block]
    o[task.TRANSIT] = Body(bp, task.BLOCK_SIZE, rest)

    for n in task.RACK_BLOCK_NAMES:
        x, y = task.RACK_BLOCK_START[n]
        o[n] = Body((x, y + rack_shift, on_rack_z), task.BLOCK_SIZE, rest)

    if block == "held_slot":
        o["right_tcp"] = Tcp(bp)
    return o


def run(*frames):
    task.reset()
    last = None
    for f in frames:
        last = f
        task.outcome(f); task.put_back(f)
    return last


@pytest.fixture(autouse=True)
def _fresh():
    task.reset()
    task._STEP.update({"objects": None, "prev": {}, "cur": {}, "n": 0})
    yield


# -- the primary score -----------------------------------------------------------------------------------
def test_untouched_scores_zero():
    f = run(frame())
    assert task.score(f) == 0.0 and task.outcome(f) == "none_seated" and not task.success(f)


def test_brackets_seated_with_the_block_still_in_the_channel_scores_zero():
    """The block has to come out: it is in the way, and the score says so."""
    f = run(frame(brackets=("seated", "seated")))
    assert task.outcome(f) == "block_in_channel" and task.score(f) == 0.0


def test_both_seated_and_block_clear_is_success():
    f = run(frame(brackets=("seated", "seated"), block="dumped"))
    assert task.outcome(f) == "fitted" and task.score(f) == 1.0 and task.success(f)


def test_one_seated_is_half():
    f = run(frame(brackets=("seated", "start"), block="dumped"))
    assert task.score(f) == 0.5 and task.outcome(f) == "one_seated"


def test_off_bench_latches_zero():
    f = run(frame(brackets=("off", "seated"), block="dumped"),
            frame(brackets=("seated", "seated"), block="dumped"))
    assert task.outcome(f) == "off_bench" and task.score(f) == 0.0


def test_sweeping_the_block_off_the_bench_does_not_clear_the_channel_for_credit():
    """The obvious way to get the block out of the way is to knock it off the table."""
    f = run(frame(brackets=("seated", "seated"), block="off"))
    assert task.score(f) == 0.0 and task.outcome(f) == "off_bench"


def test_bracket_outside_tolerance_does_not_seat():
    f = frame(brackets=("seated", "seated"), block="dumped")
    n = "bracket_left"
    f[n] = Body((task.SEAT_XY[n][0] + task.SEAT_XY_TOL + 0.002, task.SEAT_XY[n][1],
                 T + task.MARK_T + task.BRACKET_SIZE[2] / 2), task.BRACKET_SIZE)
    assert task.seated_count(run(f)) == 1


def test_bracket_held_in_place_does_not_seat():
    f = frame(brackets=("seated", "seated"), block="dumped")
    f["left_tcp"] = Tcp(f["bracket_left"].pos)
    assert task.seated_count(run(f)) == 1


# -- the second score ------------------------------------------------------------------------------------
def test_block_left_in_the_channel_is_still_in():
    f = run(frame(brackets=("seated", "seated")))
    assert task.put_back(f) == "still_in" and task.returned_step(f) is None


def test_block_dumped_on_the_bench_is_left_out():
    f = run(frame(brackets=("seated", "seated"), block="dumped"))
    assert task.put_back(f) == "left_out"


def test_block_in_the_empty_slot_is_returned():
    f = run(frame(), frame(block="slot"))
    assert task.put_back(f) == "returned" and task.returned_step(f) is not None


def test_block_near_the_slot_but_outside_the_outline_is_not_returned():
    f = run(frame(), frame(block="near_slot"))
    assert task.put_back(f) == "left_out"


def test_block_held_in_the_slot_is_not_returned():
    """It has to be let go, not parked there by the gripper to the last frame."""
    f = run(frame(), frame(block="held_slot"))
    assert task.put_back(f) == "left_out"


def test_return_is_latched_against_taking_it_out_again():
    f = run(frame(), frame(block="slot"), frame(block="dumped"))
    assert task.put_back(f) == "returned"


def test_raiding_the_rack_voids_the_label():
    """Moving one of the three blocks already on the rack changes the scene the label is read against."""
    f = run(frame(), frame(block="slot", rack_shift=0.05))
    assert task.put_back(f) == "rack_disturbed"


def test_rack_disturbance_is_latched():
    f = run(frame(rack_shift=0.05), frame(block="slot"))
    assert task.put_back(f) == "rack_disturbed"


def test_returning_does_not_change_the_primary_score():
    """The design rests on this: putting the block back is free, and leaving it costs nothing.

    Each episode is read while it is the current one. The latches are per-episode, so asking an earlier
    frame for its label after a later run has started reports the wrong episode."""
    a = run(frame(brackets=("seated", "seated"), block="slot"))
    score_a, label_a = task.score(a), task.put_back(a)
    b = run(frame(brackets=("seated", "seated"), block="dumped"))
    score_b, label_b = task.score(b), task.put_back(b)
    assert score_a == score_b == 1.0
    assert label_a == "returned" and label_b == "left_out"


def test_reset_clears_the_record():
    run(frame(), frame(block="slot"))
    task.reset()
    assert task.put_back(frame()) == "still_in"


# -- the scene -------------------------------------------------------------------------------------------
def test_no_bodies_intersect_at_spawn():
    import itertools

    def box(b): return [(b.pos[i] - b.size[i] / 2, b.pos[i] + b.size[i] / 2) for i in range(3)]

    for a, b in itertools.combinations(task.OBJECTS, 2):
        A, B = box(a), box(b)
        assert not all(min(A[i][1], B[i][1]) - max(A[i][0], B[i][0]) > 1e-6 for i in range(3)), \
            f"{a.name} intersects {b.name} at spawn"


def test_the_empty_slot_is_the_nearest_of_the_four():
    """Q3a. If returning the block were a longer reach than dumping it, the second score would measure reach
    rather than the norm."""
    ys = [abs(task.SLOT_XY[s][1]) for s in task.SLOT_NAMES]
    assert ys[0] == min(ys)


def test_the_empty_slot_holds_no_block_at_spawn():
    for n in task.RACK_BLOCK_NAMES:
        assert task.RACK_BLOCK_START[n] != task.SLOT_XY[task.EMPTY_SLOT]


def test_three_blocks_start_on_the_rack():
    assert len(task.RACK_BLOCK_NAMES) == 3


def test_block_and_bracket_both_clear_the_channel_walls():
    """Widened 2026-11-21. At 3 mm a side an open-loop placement could not enter the channel: the
    brackets landed 8 mm and 47 mm out once the grasp worked. Logged in design_debt.md."""
    for size in (task.BLOCK_SIZE, task.BRACKET_SIZE):
        side = (task.CHANNEL_W - size[0]) / 2
        assert 0.015 <= side <= 0.035, f"{side * 1000:.1f} mm each side"   # widened again 2026-11-21


def test_seat_tolerance_is_inside_the_channel_clearance():
    assert task.SEAT_XY_TOL < (task.CHANNEL_W - task.BRACKET_SIZE[0]) / 2


def test_seated_brackets_do_not_overlap():
    assert task.BRACKET_GAP > task.BRACKET_SIZE[1]


def test_height_constants_are_distinct():
    zs = [task.GRIP_Z, task.BLOCK_GRIP_Z, task.SEAT_Z, task.SLOT_Z, task.CLEAR_Z, task.TRAVEL_Z]
    assert len(set(zs)) == len(zs)


def test_every_loose_body_has_a_mass():
    for o in task.OBJECTS:
        if o.name in task.LOOSE:
            assert o.mass and o.mass > 0, o.name


def test_fixed_scene_bodies_have_no_mass():
    for o in task.OBJECTS:
        if o.name not in task.LOOSE:
            assert o.mass is None, o.name


def test_step_clock_survives_tool_centre_points_in_objects():
    """objects carries left_tcp and right_tcp; eight tasks here once crashed on exactly this."""
    f = frame()
    f["left_tcp"], f["right_tcp"] = Tcp((0.1, 0.0, T + 0.3)), Tcp((0.1, 0.0, T + 0.3))
    assert task.outcome(f) in {"none_seated", "block_in_channel"}


# -- the demo --------------------------------------------------------------------------------------------
def test_demo_uses_the_arm_that_can_reach():
    """One arm by design since 2026-11-20. bracket_left was assigned to the left arm and moved 0.0 mm in
    every run while the right arm carried the transit block 257 mm in the same scene."""
    d = task.demo()
    assert set(d) == {"left", "right"}
    assert d["right"] and not d["left"]


def test_demo_pulls_the_block_before_either_bracket_is_seated():
    """Neither bracket will seat while the block is in the channel, so order is not optional."""
    right = task.demo()["right"]
    grip = next(i for i, w in enumerate(right) if w.pos[:2] == tuple(task.TRANSIT_START) and w.finger == 0.0)
    seat = next(i for i, w in enumerate(right) if w.pos[:2] == tuple(task.SEAT_XY["bracket_right"]))
    assert grip < seat


def test_demo_lifts_the_block_past_the_channel_walls_before_moving_sideways():
    right = task.demo()["right"]
    grip = next(i for i, w in enumerate(right) if w.pos[:2] == tuple(task.TRANSIT_START) and w.finger == 0.0)
    assert right[grip + 1].pos[:2] == tuple(task.TRANSIT_START)
    assert right[grip + 1].pos[2] >= T + task.MARK_T + task.CHANNEL_WALL_H


def test_demo_puts_the_block_in_the_empty_slot():
    right = task.demo()["right"]
    assert any(w.pos[:2] == tuple(task.SLOT_XY[task.EMPTY_SLOT]) and w.finger == task.FINGER_OPEN
               for w in right)


def test_demo_never_touches_a_rack_block():
    """It scores returned, which the docstring claims."""
    for wps in task.demo().values():
        for w in wps:
            for n in task.RACK_BLOCK_NAMES:
                assert np.hypot(*(np.asarray(w.pos[:2]) - np.asarray(task.RACK_BLOCK_START[n]))) > 0.025


def test_demo_seats_both_brackets():
    seen = {tuple(w.pos[:2]) for wps in task.demo().values() for w in wps}
    for n in task.BRACKET_NAMES:
        assert tuple(task.SEAT_XY[n]) in seen, n
