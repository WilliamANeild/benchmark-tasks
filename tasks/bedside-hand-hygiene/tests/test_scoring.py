"""Scorer tests for the hand-hygiene task. No GPU, no Isaac Sim, no yam_arena.

    python -m pytest -q tests

The thing under test that the others did not have is an ORDERING. Most of these drive a sequence of control
steps and check that the hygiene verdict depends on when the plunger went down, not merely whether it did.
"""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

TABLE_TOP_Z = 0.80


def _stubs():
    tf = types.ModuleType("yam_arena.task_file")
    tf.TABLE_TOP = {"station": TABLE_TOP_Z}

    class Box:
        def __init__(self, name, pos, size, mass=None, color=None):
            self.name, self.pos, self.size, self.color = name, pos, size, color
            # a Box with mass= is a dynamic body; without it the engine spawns it static
            self.mass = mass

    tf.Box = tf.Asset = Box

    class Cylinder(Box):
        def __init__(self, name, pos, radius, height, mass=None, color=None):
            super().__init__(name, pos, (2 * radius, 2 * radius, height), mass, color)
            self.radius, self.height = radius, height

    tf.Cylinder = Cylinder
    dm = types.ModuleType("yam_arena.demo")
    dm.FINGER_OPEN = 1.0

    class Waypoint:
        """Matches the real signature: (duration, pos, finger, label='', yaw=0.0).
        Verified on the instance 2026-11-20; the old stub had `comment` and no `yaw`."""
        def __init__(self, duration, pos, finger, label="", yaw=0.0):
            self.duration, self.pos, self.finger, self.label, self.yaw = duration, pos, finger, label, yaw
            self.comment = label

    def pick_place_waypoints(pick, place, travel_z, yaw=0.0, place_yaw=None, speed=1.0):
        return [Waypoint(1.0, pick, 1.0, "pick"),
                Waypoint(1.0, (pick[0], pick[1], travel_z), 0.0, "lift"),
                Waypoint(1.0, place, 0.0, "place")]

    dm.Waypoint, dm.pick_place_waypoints = Waypoint, pick_place_waypoints
    pkg = types.ModuleType("yam_arena")
    pkg.task_file, pkg.demo = tf, dm
    for n, m in (("yam_arena", pkg), ("yam_arena.task_file", tf), ("yam_arena.demo", dm)):
        sys.modules.setdefault(n, m)


_stubs()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
task = importlib.import_module("task")


class Body:
    def __init__(self, pos, size, at_rest=True):
        self.pos, self.size = tuple(map(float, pos)), tuple(map(float, size))
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


def frame(pad="start", pack="start", pressed=False):
    """One control step. pad: 'start' | 'bin' | 'away'. pack: 'start' | 'table' | 'away'."""
    pad_pos = {"start": (*task.PAD_START, TABLE_TOP_Z + task.MARK_T + task.PAD_SIZE[2] / 2),
               "bin": (*task.BIN_XY, TABLE_TOP_Z + task.PAD_SIZE[2] / 2),
               "away": (0.28, 0.02, TABLE_TOP_Z + task.PAD_SIZE[2] / 2)}[pad]
    pack_pos = {"start": (*task.PACK_START, TABLE_TOP_Z + task.MARK_T + task.PACK_SIZE[2] / 2),
                "table": (*task.TABLE_XY, TABLE_TOP_Z + task.MARK_T + task.PACK_SIZE[2] / 2),
                "away": (0.26, 0.10, TABLE_TOP_Z + task.PACK_SIZE[2] / 2)}[pack]
    pz = task.PLUNGER_REST_Z - (task.PRESS_TRAVEL + 0.005 if pressed else 0.0)
    return {"soiled_pad": Body(pad_pos, task.PAD_SIZE),
            "clean_pack": Body(pack_pos, task.PACK_SIZE),
            "plunger": Body((*task.DISP_XY, pz), task.PLUNGER_SIZE)}


def run(*frames):
    """Feed a sequence of control steps; return the last objects dict."""
    last = None
    for f in frames:
        last = f
        task.outcome(last)      # drives the step clock and the event recorder
    return last


@pytest.fixture(autouse=True)
def _reset():
    task._STEP.update({"objects": None, "prev": {}, "cur": {}, "n": 0})
    for k in task._EV:
        task._EV[k] = None
    yield


# -- primary score ----------------------------------------------------------------------------------------

def test_untouched_scores_nothing():
    o = run(frame())
    assert task.outcome(o) == "neither" and task.score(o) == 0.0


def test_both_placements_are_a_full_score():
    o = run(frame(), frame(pad="bin"), frame(pad="bin", pack="table"))
    assert task.pad_binned(o) and task.pack_placed(o)
    assert task.outcome(o) == "both_placed" and task.score(o) == 1.0 and task.success(o)


def test_each_placement_is_a_half():
    o = run(frame(), frame(pad="bin"))
    assert task.outcome(o) == "pad_only" and task.score(o) == 0.5
    task._STEP.update({"objects": None, "prev": {}, "cur": {}, "n": 0})
    o = run(frame(), frame(pack="table"))
    assert task.outcome(o) == "pack_only" and task.score(o) == 0.5


def test_anything_off_the_bench_is_zero():
    o = frame(pad="bin", pack="table")
    o["clean_pack"] = Body((0.30, 0.40, TABLE_TOP_Z - 0.12), task.PACK_SIZE)
    assert task.outcome(o) == "off_bench" and task.score(o) == 0.0


# -- the hygiene step, which is an ordering --------------------------------------------------------------

def test_sanitising_between_the_two_is_clean():
    o = run(frame(), frame(pad="bin"), frame(pad="bin", pressed=True),
            frame(pad="bin", pack="away", pressed=True), frame(pad="bin", pack="table", pressed=True))
    assert task.hygiene(o) == "clean"
    assert task.score(o) == 1.0, "and the primary score is untouched by it"


def test_never_sanitising_is_skipped():
    o = run(frame(), frame(pad="bin"), frame(pad="bin", pack="away"), frame(pad="bin", pack="table"))
    assert task.hygiene(o) == "skipped"
    assert task.score(o) == 1.0, "a full primary score with the hygiene step skipped"


def test_sanitising_before_touching_the_pad_does_not_count():
    """Pressing the plunger at the very start, then doing the whole job, is out of order."""
    o = run(frame(), frame(pressed=True), frame(pad="bin", pressed=True),
            frame(pad="bin", pack="away", pressed=True), frame(pad="bin", pack="table", pressed=True))
    assert task.hygiene(o) == "out_of_order"


def test_sanitising_after_handling_the_clean_pack_does_not_count():
    o = run(frame(), frame(pad="bin"), frame(pad="bin", pack="away"),
            frame(pad="bin", pack="table"), frame(pad="bin", pack="table", pressed=True))
    assert task.hygiene(o) == "out_of_order"


def test_the_verdict_is_na_until_the_sequence_gets_far_enough():
    assert task.hygiene(run(frame())) == "n/a"
    assert task.hygiene(run(frame(), frame(pad="bin"))) == "n/a", "pack never moved"


def test_the_events_latch_and_cannot_be_undone():
    """Press the plunger, then let it spring back up. The sanitising still counts as having happened."""
    o = run(frame(), frame(pad="bin"), frame(pad="bin", pressed=True),
            frame(pad="bin", pressed=False), frame(pad="bin", pack="table", pressed=False))
    assert task._EV["sanitised"] is not None
    assert task.hygiene(o) == "clean"


def test_a_fresh_episode_clears_the_record():
    run(frame(), frame(pad="bin", pressed=True), frame(pad="bin", pack="table", pressed=True))
    assert task._EV["pad_binned"] is not None
    run(frame())
    assert all(v is None for v in task._EV.values())


# -- geometry and the demo --------------------------------------------------------------------------------

def test_the_bin_and_the_clean_supply_are_on_opposite_sides():
    assert task.BIN_XY[1] < 0 < task.SHELF_XY[1], "waste one side, clean supply the other"


def test_the_dispenser_sits_between_them_at_the_back():
    assert task.DISP_XY[0] > task.TABLE_XY[0] and task.DISP_XY[0] > task.SHELF_XY[0]


def test_a_press_needs_real_travel():
    assert task.PRESS_TRAVEL >= 0.015, "a brush past the plunger must not register as a press"
    # PRESS_Z aims the gripper; the fingertips that touch the plunger are TCP_LIFT below it.
    assert task.PRESS_Z - task.TCP_LIFT < task.PLUNGER_REST_Z - task.PRESS_TRAVEL, \
        "the demo presses past the threshold"


def test_the_demo_does_the_three_steps_in_order():
    """Pad binned, then plunger pressed, then pack lifted. That order is what hygiene() reads, and it is
    unchanged by the 2026-11-20 arm reassignment: the LEFT arm now does the pad, because the right arm
    could not pick it up across the body."""
    plan = task.demo()
    lz = [w.pos[2] for w in plan["left"]]
    rz = [w.pos[2] for w in plan["right"]]
    assert any(z == pytest.approx(task.PAD_DROP_Z) for z in lz), "the left arm bins the pad"
    assert any(z == pytest.approx(task.PRESS_Z) for z in rz), "the right arm presses the plunger"
    assert any(z == pytest.approx(task.PACK_PLACE_Z) for z in lz), "the left arm lays the pack down"
    assert lz.index(task.PAD_DROP_Z) < lz.index(task.PACK_PLACE_Z), "pad binned before the pack is laid"


def test_the_press_happens_between_binning_the_pad_and_lifting_the_pack():
    """The hygiene ordering, expressed as durations now that the two acts are on different arms."""
    plan = task.demo()
    left, right = plan["left"], plan["right"]
    pad_done = sum(w.duration for w in left[:[w.pos[2] for w in left].index(task.PAD_DROP_Z) + 1])
    press_at = sum(w.duration for w in right[:[w.pos[2] for w in right].index(task.PRESS_Z) + 1])
    pack_lift = sum(w.duration for w in left[:[w.pos[2] for w in left].index(task.PACK_GRIP_Z) + 1])
    assert pad_done <= press_at <= pack_lift, (pad_done, press_at, pack_lift)


# -- release: a body still in the grippers is not placed -------------------------------------------------
class Tcp:
    """What the engine puts in objects["left_tcp"]: a thing with .pos and no quat."""
    def __init__(self, pos): self.pos = tuple(map(float, pos))


def test_a_pad_dangled_in_the_bin_is_not_binned():
    """The bin event latches, so a pad held over the bin and taken back out would have counted for good."""
    f = frame(pad="bin")
    f["left_tcp"] = Tcp(f["soiled_pad"].pos)
    assert not task._in_bin(f)


def test_the_pad_counts_once_released_in_the_bin():
    f = frame(pad="bin")
    f["left_tcp"] = Tcp((0.10, 0.0, TABLE_TOP_Z + 0.30))
    assert task._in_bin(f)
