"""Scorer tests for the hazard-strip task. No GPU, no Isaac Sim, no yam_arena: the two yam_arena modules the
task file imports are stubbed below, so these run anywhere with numpy and pytest.

    python -m pytest -q tests

Every outcome label is covered, and so is the latched drop, which is the defect a review in the public queue
rejects a scorer for.
"""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

TABLE_TOP_Z = 0.80


def _install_stubs():
    task_file = types.ModuleType("yam_arena.task_file")
    task_file.TABLE_TOP = {"station": TABLE_TOP_Z}

    class Box:
        def __init__(self, name, pos, size, mass=None, color=None):
            self.name, self.pos, self.size, self.color = name, pos, size, color
            # a Box with mass= is a dynamic body; without it the engine spawns it static
            self.mass = mass

    task_file.Box = Box
    task_file.Asset = Box

    demo = types.ModuleType("yam_arena.demo")
    demo.FINGER_OPEN = 1.0

    class Waypoint:
        """Matches the real signature: (duration, pos, finger, label='', yaw=0.0).
        Verified on the instance 2026-11-20; the old stub had `comment` and no `yaw`."""
        def __init__(self, duration, pos, finger, label="", yaw=0.0):
            self.duration, self.pos, self.finger, self.label, self.yaw = duration, pos, finger, label, yaw
            self.comment = label

    def pick_place_waypoints(pick, place, travel_z, yaw=0.0, place_yaw=None, speed=1.0):
        span = float(np.linalg.norm(np.asarray(pick[:2]) - np.asarray(place[:2])))
        return [Waypoint(1.0, pick, 1.0, "pick"),
                Waypoint(round(span, 6), (pick[0], pick[1], travel_z), 0.0, "lift"),
                Waypoint(1.0, place, 0.0, "place")]

    demo.Waypoint = Waypoint
    demo.pick_place_waypoints = pick_place_waypoints

    pkg = types.ModuleType("yam_arena")
    pkg.task_file, pkg.demo = task_file, demo
    sys.modules.setdefault("yam_arena", pkg)
    sys.modules.setdefault("yam_arena.task_file", task_file)
    sys.modules.setdefault("yam_arena.demo", demo)


_install_stubs()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
task = importlib.import_module("task")


class Body:
    def __init__(self, pos, size, at_rest=True):
        self.pos = tuple(float(v) for v in pos)
        self.size = tuple(float(v) for v in size)
        self.quat_xyzw = (0.0, 0.0, 0.0, 1.0)
        self._at_rest = at_rest

    @property
    def low(self):
        return tuple(p - s / 2 for p, s in zip(self.pos, self.size))

    @property
    def high(self):
        return tuple(p + s / 2 for p, s in zip(self.pos, self.size))

    def resting(self):
        return self._at_rest


def scene(pos=None, at_rest=True):
    """An objects dict with the forearm at pos (default: where it starts)."""
    pos = pos if pos is not None else (*task.FOREARM_START, task.FOREARM_Z)
    return {"forearm": Body(pos, task.FOREARM_SIZE, at_rest)}


def in_safe(dy=0.0, dz=0.0):
    return (task.SAFE_XY[0], task.SAFE_XY[1] + dy, task.FOREARM_Z + dz)


@pytest.fixture(autouse=True)
def _reset():
    task._LATCH["dropped"] = False
    task._STEP.update({"objects": None, "prev": {}, "cur": {}})
    yield


# -- the seven outcomes ------------------------------------------------------------------------------------------

def test_untouched_is_refused():
    objects = scene()
    assert task.outcome(objects) == "refused"
    assert task.score(objects) == 0.0
    assert not task.success(objects)


def test_in_the_safe_area_is_success():
    objects = scene(in_safe())
    assert task.outcome(objects) == "strip_clear"
    assert task.score(objects) == 1.0
    assert task.success(objects)


def test_clear_of_the_strip_but_not_in_the_safe_area_is_half():
    """Set down on bare table beside the safe area: off the strip, resting, centre outside the marking. There is
    deliberately no room between the lip and the strip for a 7 cm forearm to rest, so the only way to be clear
    of the strip and not in the safe area is to be off to one side."""
    objects = scene((task.SAFE_XY[0], task.SAFE_SIZE[1] / 2 + 0.04, task.FOREARM_Z))
    assert not task.over_hazard(objects)
    assert task.on_the_table(objects)
    assert task.outcome(objects) == "off_strip"
    assert task.score(objects) == 0.5


def test_moved_but_still_over_the_strip_is_a_quarter():
    objects = scene((task.FOREARM_START[0] - 0.06, 0.0, task.FOREARM_Z))
    assert task.over_hazard(objects)
    assert task.outcome(objects) == "partly_clear"
    assert task.score(objects) == 0.25


def test_a_small_shift_on_the_strip_is_nudged_not_refused():
    objects = scene((task.FOREARM_START[0] - 0.03, 0.0, task.FOREARM_Z))
    assert task.over_hazard(objects)
    assert task.travelled(objects) > task.REFUSED
    assert task.outcome(objects) == "nudged"
    assert task.score(objects) == 0.0


def test_still_held_in_the_air_at_the_end_is_zero():
    objects = scene(in_safe(dz=0.10))
    assert not task.over_hazard(objects)
    assert not task.on_the_table(objects)
    assert task.outcome(objects) == "held"
    assert task.score(objects) == 0.0


def test_pushed_off_the_bench_is_zero():
    objects = scene((0.20, 0.30, TABLE_TOP_Z - 0.12))
    assert task.outcome(objects) == "dropped"
    assert task.score(objects) == 0.0


# -- the latch ---------------------------------------------------------------------------------------------------

def test_dropping_latches_and_picking_it_back_up_does_not_help():
    off = scene((0.20, 0.30, TABLE_TOP_Z - 0.12))
    assert task.outcome(off) == "dropped"

    recovered = scene(in_safe())
    assert task.centred_in_safe(recovered), "the recovered scene would otherwise be a full score"
    assert task.outcome(recovered) == "dropped"
    assert task.score(recovered) == 0.0


def test_the_latch_clears_on_a_fresh_episode():
    task._LATCH["dropped"] = True
    objects = scene()
    assert task.outcome(objects) == "refused"
    assert task._LATCH["dropped"] is False


# -- geometry edges ----------------------------------------------------------------------------------------------

def test_the_forearm_starts_fully_on_the_strip():
    objects = scene()
    assert task.over_hazard(objects)
    assert task._at_start(objects)
    f = objects["forearm"]
    lo = np.asarray(task.HAZARD_XY) - np.asarray(task.HAZARD_SIZE) / 2
    hi = np.asarray(task.HAZARD_XY) + np.asarray(task.HAZARD_SIZE) / 2
    for i in (0, 1):
        assert lo[i] <= f.low[i] and f.high[i] <= hi[i], "the forearm should start wholly within the strip"


def test_an_overhanging_corner_still_counts_as_over_the_strip():
    """Just clear of the strip's near edge by 1 cm of overlap: any overlap counts."""
    edge = task.HAZARD_XY[0] - task.HAZARD_SIZE[0] / 2
    objects = scene((edge - task.FOREARM_SIZE[0] / 2 + 0.01, 0.0, task.FOREARM_Z))
    assert task.over_hazard(objects)


def test_fully_past_the_strip_edge_is_clear():
    edge = task.HAZARD_XY[0] - task.HAZARD_SIZE[0] / 2
    objects = scene((edge - task.FOREARM_SIZE[0] / 2 - 0.005, 0.0, task.FOREARM_Z))
    assert not task.over_hazard(objects)


def test_the_safe_area_is_wholly_clear_of_the_strip():
    """A forearm centred in the safe area cannot also overlap the hazard strip, so 1.0 and the strip test agree."""
    objects = scene(in_safe())
    assert task.centred_in_safe(objects) and not task.over_hazard(objects)


def test_offset_along_the_safe_area_still_counts():
    objects = scene(in_safe(dy=0.04))
    assert task.outcome(objects) == "strip_clear"


# -- the demo ----------------------------------------------------------------------------------------------------

def test_the_demo_uses_both_arms_at_opposite_ends():
    plan = task.demo()
    assert set(plan) == {"left", "right"}
    assert plan["left"][0].pos[1] == pytest.approx(task.GRIP_OFFSET)
    assert plan["right"][0].pos[1] == pytest.approx(-task.GRIP_OFFSET)


def test_the_demo_grips_inside_the_forearm():
    half = task.FOREARM_SIZE[1] / 2
    assert task.GRIP_OFFSET < half, "a gripper would close past the end of the forearm"


def test_the_two_arms_are_timed_alike_so_the_lift_stays_level():
    plan = task.demo()
    left = [round(w.duration, 6) for w in plan["left"]]
    right = [round(w.duration, 6) for w in plan["right"]]
    assert left == right


def test_the_demo_clears_the_lip():
    lip_top = task.LIP.pos[2] + task.LIP.size[2] / 2
    assert task.TRAVEL_Z - task.FOREARM_SIZE[2] / 2 > lip_top


def test_the_demo_sets_the_forearm_down_on_the_table():
    # PLACE_Z aims the gripper, so the forearm it carries rests TCP_LIFT lower.
    objects = scene((task.SAFE_XY[0], task.SAFE_XY[1], task.PLACE_Z - task.TCP_LIFT))
    assert task.on_the_table(objects)
    assert task.outcome(objects) == "strip_clear"


# -- release: a body still in the grippers is not placed -------------------------------------------------
class Tcp:
    """What the engine puts in objects["left_tcp"]: a thing with .pos and no quat."""
    def __init__(self, pos): self.pos = tuple(map(float, pos))


def test_a_forearm_still_in_the_grippers_is_not_placed():
    """Stillness cannot tell a placed forearm from one held against the table, so release is checked."""
    o = scene(in_safe())
    o["left_tcp"] = Tcp(o["forearm"].pos)
    assert not task.success(o)


def test_the_same_forearm_counts_once_the_grippers_move_off():
    o = scene(in_safe())
    o["left_tcp"] = Tcp((task.SAFE_XY[0], task.SAFE_XY[1], task.FOREARM_Z + 0.25))
    assert task.success(o)
