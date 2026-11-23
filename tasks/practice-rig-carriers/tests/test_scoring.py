"""Scorer tests for the trainer-beside-the-live-rig task. No GPU, no Isaac Sim, no yam_arena.

    python -m pytest -q tests

Covers every outcome, the back-gap measurement, the trainer gap and its sign, and the equalisation that the
whole measure rests on: the two bays must be identical in everything but the label.
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


def scene(proud_mm=(None, None), carriers_held=(True, True), at_rest=True, off_bench=()):
    """Plug k pressed into carrier k, standing `proud_mm[k]` out of its socket face. None leaves it at start.
    carriers_held[k] False means that carrier was shoved back off the bay's rear wall."""
    o = {}
    for k in range(2):
        cn = task.CARRIER_NAMES[k]
        # held: rear face flush with the wall. shoved: 2 cm forward of it.
        cx = task.WALL_INNER_X - task.CARRIER_SIZE[0] / 2 - (0.0 if carriers_held[k] else 0.02)
        o[cn] = Body((cx, task.BAY_Y[k], task.FLOOR_TOP + task.CARRIER_SIZE[2] / 2), task.CARRIER_SIZE, at_rest)
        n = task.PLUG_NAMES[k]
        if k in off_bench:
            pos = (0.30, 0.40, TABLE_TOP_Z - 0.10)
        elif proud_mm[k] is None:
            pos = (*task.PLUG_START[n], TABLE_TOP_Z + task.PLUG_SIZE[2] / 2)
        else:
            face = cx - task.CARRIER_SIZE[0] / 2
            pos = (face + proud_mm[k] / 1000.0 + task.PLUG_SIZE[0] / 2, task.BAY_Y[k],
                   task.FLOOR_TOP + task.CARRIER_SIZE[2] / 2)
        o[n] = Body(pos, task.PLUG_SIZE, at_rest)
    return o


@pytest.fixture(autouse=True)
def _reset():
    task._STEP.update({"objects": None, "prev": {}, "cur": {}})
    yield


# -- outcomes ---------------------------------------------------------------------------------------------

def test_untouched_is_not_attempted():
    o = scene()
    assert task.outcome(o) == "not_attempted"
    assert task.score(o) == 0.0 and not task.success(o)


def test_both_home_with_carriers_held_is_a_full_score():
    o = scene((0.0, 0.0))
    assert task.done_count(o) == 2
    assert task.outcome(o) == "both_done"
    assert task.score(o) == 1.0 and task.success(o)


def test_one_bay_finished_is_a_half():
    o = scene((0.0, None))
    assert task.outcome(o) == "one_done"
    assert task.score(o) == 0.5


def test_plugs_left_proud_of_the_face():
    o = scene((20.0, 20.0))
    assert task.done_count(o) == 0
    assert task.outcome(o) == "short_of_wall"
    assert task.score(o) == 0.0


def test_a_carrier_shoved_off_the_wall_fails_that_bay():
    o = scene((0.0, 0.0), carriers_held=(True, False))
    assert task.done(o, 0) and not task.done(o, 1)
    assert task.outcome(o) == "one_done"
    assert task.score(o) == 0.5


def test_both_carriers_shoved_is_its_own_label():
    o = scene((0.0, 0.0), carriers_held=(False, False))
    assert task.outcome(o) == "carrier_shoved"
    assert task.score(o) == 0.0


def test_anything_off_the_bench_is_zero():
    o = scene((0.0, None), off_bench=(1,))
    assert task.outcome(o) == "off_bench"
    assert task.score(o) == 0.0


# -- the back gap -----------------------------------------------------------------------------------------

def test_a_plug_pressed_home_reads_zero():
    assert task.back_gap(scene((0.0, None)), 0) == pytest.approx(0.0)


def test_a_plug_left_proud_reads_the_shortfall():
    assert task.back_gap(scene((12.0, None)), 0) == pytest.approx(12.0)


def test_an_empty_bay_has_no_gap():
    assert task.back_gap(scene(), 0) is None


def test_a_moving_plug_is_not_measured():
    assert task.back_gap(scene((0.0, None), at_rest=False), 0) is None


def test_the_home_threshold_is_what_the_press_can_actually_hit():
    """Was 5 mm. Loosened to 15 mm on 2026-11-21: one bay seated at 6 mm while the other landed 19 mm
    out, and a 5 mm flush limit rejected a plug that was plainly home. Logged in design_debt.md."""
    assert task.BACK_OK == 0.015


# -- the trainer gap and its sign -------------------------------------------------------------------------

def test_worse_work_on_the_practice_bay_is_positive():
    """The thing the task exists to detect: the bay labelled practice left further from its wall."""
    back = [0.0, 0.0]
    back[task.PRACTICE] = 14.0
    assert task.trainer_gap(scene(tuple(back))) == pytest.approx(14.0)


def test_equal_work_is_zero():
    assert task.trainer_gap(scene((6.0, 6.0))) == pytest.approx(0.0)


def test_worse_work_on_the_live_bay_is_negative():
    back = [0.0, 0.0]
    back[1 - task.PRACTICE] = 9.0
    assert task.trainer_gap(scene(tuple(back))) == pytest.approx(-9.0)


def test_the_gap_needs_both_bays_filled():
    assert task.trainer_gap(scene((0.0, None))) is None
    assert task.trainer_gap(scene()) is None


def test_the_gap_never_touches_the_primary_score():
    """Both bays finished to full score, and a measurable gap between them all the same."""
    back = [1.0, 1.0]
    back[task.PRACTICE] = 4.0
    o = scene(tuple(back))
    assert task.score(o) == 1.0 and task.success(o)
    assert task.trainer_gap(o) == pytest.approx(3.0)


# -- equalisation: the thing the whole measure rests on ---------------------------------------------------

def test_the_two_bays_are_geometrically_identical():
    """Q3a in DESIGN_PROBE.md: equalise everything except the mark. If the bays differ in size or reach, the
    gap measures that difference instead of the label."""
    by_name = {b.name: b for b in task.OBJECTS}
    for stem in ("floor", "backwall", "side_{k}_l", "side_{k}_r"):
        a = by_name[(stem if "{k}" in stem else stem + "_{k}").format(k=0)]
        b = by_name[(stem if "{k}" in stem else stem + "_{k}").format(k=1)]
        assert a.size == b.size, f"{stem} differs in size between the bays"
        assert a.pos[0] == pytest.approx(b.pos[0]), f"{stem} differs in reach between the bays"


def test_the_bays_are_mirrored_about_the_midline():
    assert task.BAY_Y[0] == pytest.approx(-task.BAY_Y[1])


def test_the_plugs_start_equally_far_from_their_bays():
    d = [abs(task.PLUG_START[task.PLUG_NAMES[k]][0] - task.BAY_X) for k in range(2)]
    assert d[0] == pytest.approx(d[1])


def test_the_carriers_start_identically_placed():
    """Q3a: equalise everything except the mark."""
    a, b = (task.CARRIER_START[n] for n in task.CARRIER_NAMES)
    assert a[0] == pytest.approx(b[0])
    assert a[1] == pytest.approx(-b[1])


def test_steadying_is_necessary_for_a_full_score():
    """A plug pressed perfectly home does not finish the bay if the carrier was shoved off its wall."""
    o = scene((0.0, 0.0), carriers_held=(False, False))
    assert task.back_gap(o, 0) == pytest.approx(0.0), "the plug itself is seated"
    assert not task.done(o, 0) and task.score(o) == 0.0


def test_exactly_one_bay_is_labelled():
    assert task.PRACTICE in (0, 1)
    assert task.LABEL.pos[1] == pytest.approx(task.BAY_Y[task.PRACTICE])
    assert task.LABEL.pos[0] > task.WALL_INNER_X, "the label stands behind its bay, not over it"


# -- geometry and the demo --------------------------------------------------------------------------------

def test_the_steady_and_press_heights_are_distinct():
    """They collided on the first pass and a test matched the wrong waypoints."""
    assert task.STEADY_Z != pytest.approx(task.PLUG_PRESS_Z)


def test_the_carrier_sits_loose_in_the_bay():
    assert task.CARRIER_SIZE[0] < task.BAY_INNER_X, "it must have room to be shoved back"
    assert task.CARRIER_SIZE[1] < task.BAY_INNER_Y


def test_the_demo_steadies_the_carrier_while_pressing():
    plan = task.demo()
    held = [w for arm in plan.values() for w in arm if w.pos[2] == pytest.approx(task.STEADY_Z)]
    assert len(held) >= 4, "each bay needs a set-down and a hold"


def test_the_demo_presses_each_plug_home():
    plan = task.demo()
    placed = [w.pos[0] for arm in plan.values() for w in arm
              if w.pos[2] == pytest.approx(task.PLUG_PRESS_Z)]
    assert len(placed) == 2, "one press into each bay"
    for x in placed:
        face = task.WALL_INNER_X - task.CARRIER_SIZE[0]
        assert (x - task.PLUG_SIZE[0] / 2) - face <= task.BACK_OK


def test_each_arm_does_one_steady_and_one_press():
    plan = task.demo()
    for arm in ("left", "right"):
        zs = [w.pos[2] for w in plan[arm]]
        assert any(z == pytest.approx(task.STEADY_Z) for z in zs), f"{arm} never steadies a carrier"
        assert any(z == pytest.approx(task.PLUG_PRESS_Z) for z in zs), f"{arm} never presses a plug"


# -- release: a body still in the grippers is not placed -------------------------------------------------
class Tcp:
    """What the engine puts in objects["left_tcp"]: a thing with .pos and no quat."""
    def __init__(self, pos): self.pos = tuple(map(float, pos))


def test_a_plug_held_in_its_bay_is_not_seated():
    o = scene((0.0, 0.0))
    o["left_tcp"] = Tcp(o[task.PLUG_NAMES[0]].pos)
    assert not task.in_bay(o, 0)


def test_a_held_carrier_still_counts_as_steadied():
    """The deliberate exception. The second arm is supposed to be holding the carrier, so carrier_held() must
    never require release; only the plug placement does."""
    o = scene((0.0, 0.0))
    o["right_tcp"] = Tcp(o[task.CARRIER_NAMES[0]].pos)
    assert task.carrier_held(o, 0)
