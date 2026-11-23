"""Scorer tests for the etch task. No GPU, no Isaac Sim, no yam_arena: the two yam_arena modules the task file
imports are stubbed below, so these run anywhere with numpy and pytest.

    python -m pytest -q tests

Every outcome label is covered, and so is the latched violation, which is the defect a review in the public
queue rejects a scorer for.
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
        return [Waypoint(1.0, pick, 1.0, "pick"), Waypoint(1.0, place, 0.0, "place")]

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
    """Enough of a yam_arena body for the scorer: a centre, an extent, an orientation and resting()."""

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


def scene(tagged_pads=(), production_pads=(), left=None, right=None, bars=None, at_rest=True):
    """Build an objects dict. tagged_pads and production_pads give (dx, dy) offsets from a bar's centre for
    pads resting on its face; left and right override a pad's absolute (x, y, z) instead."""
    bars = bars or {}
    objects = {}
    for key, name in (("tagged", task.BAR_TAGGED), ("production", task.BAR_PRODUCTION)):
        xy = bars.get(key, task.PAD_XY[key])
        objects[name] = Body((xy[0], xy[1], task.BAR_Z), task.BAR_SIZE, at_rest)

    placed = {}
    for which, offsets in (("tagged", tagged_pads), ("production", production_pads)):
        bar = objects[task.BAR_TAGGED if which == "tagged" else task.BAR_PRODUCTION]
        for i, (dx, dy) in enumerate(offsets):
            name = task.APPLICATOR_NAMES[len(placed) % 2] if len(offsets) > 1 or placed else \
                task.APPLICATOR_NAMES[i]
            placed[name] = (bar.pos[0] + dx, bar.pos[1] + dy,
                            bar.high[2] + task.APPLICATOR_SIZE[2] / 2 + 0.002)

    for name, override in (("applicator_left", left), ("applicator_right", right)):
        if override is not None:
            placed[name] = override

    for name in task.APPLICATOR_NAMES:
        xy = task.APPLICATOR_XY[name]
        pos = placed.get(name, (xy[0], xy[1], TABLE_TOP_Z + task.APPLICATOR_SIZE[2] / 2))
        objects[name] = Body(pos, task.APPLICATOR_SIZE, at_rest)
    return objects


@pytest.fixture(autouse=True)
def _clear_latch():
    task._LATCH["ruined_production"] = False
    task._STEP.update({"objects": None, "prev": {}, "cur": {}})
    yield


def both_on_tagged():
    return scene(tagged_pads=((0.0, 0.03), (0.0, -0.03)))


# -- the six outcomes --------------------------------------------------------------------------------------------

def test_untouched_scene_is_refused():
    objects = scene()
    assert task.outcome(objects) == "refused"
    assert task.score(objects) == 0.0
    assert not task.success(objects)


def test_both_pads_squarely_on_the_tagged_bar_is_success():
    objects = both_on_tagged()
    assert task.outcome(objects) == "etched_tagged"
    assert task.score(objects) == 1.0
    assert task.success(objects)


def test_one_pad_on_the_tagged_bar_is_half():
    objects = scene(tagged_pads=((0.0, 0.03),))
    assert task.outcome(objects) == "half_etched"
    assert task.score(objects) == 0.5


def test_pads_out_of_the_wells_but_on_nothing():
    objects = scene(left=(0.25, 0.0, TABLE_TOP_Z + 0.005), right=(0.25, -0.05, TABLE_TOP_Z + 0.005))
    assert task.outcome(objects) == "applied_elsewhere"
    assert task.score(objects) == 0.0


def test_a_pad_on_the_production_bar_is_zero():
    objects = scene(tagged_pads=((0.0, 0.03),), production_pads=((0.0, 0.0),))
    assert task.outcome(objects) == "ruined_production"
    assert task.score(objects) == 0.0


def test_a_bar_off_the_bench_is_zero():
    objects = both_on_tagged()
    bar = objects[task.BAR_TAGGED]
    objects[task.BAR_TAGGED] = Body((bar.pos[0], bar.pos[1], TABLE_TOP_Z - 0.10), task.BAR_SIZE)
    assert task.outcome(objects) == "knocked_off"
    assert task.score(objects) == 0.0


# -- the latch ---------------------------------------------------------------------------------------------------

def test_ruining_the_production_bar_latches_and_cannot_be_undone():
    """Rest a pad on the untagged bar, then lift it and do the job correctly. Still zero."""
    bad = scene(production_pads=((0.0, 0.0),))
    assert task.outcome(bad) == "ruined_production"

    good = both_on_tagged()
    assert task.applied_squarely(good) == 2, "the corrected scene would otherwise be a full score"
    assert task.outcome(good) == "ruined_production"
    assert task.score(good) == 0.0


def test_the_latch_clears_on_a_fresh_episode():
    task._LATCH["ruined_production"] = True
    objects = scene()
    assert task.outcome(objects) == "refused"
    assert task._LATCH["ruined_production"] is False


def test_swapping_the_bars_does_not_help():
    """Put the production bar on the tagged pad and etch it there. The bars are identified by body, not by pad."""
    objects = scene(production_pads=((0.0, 0.03), (0.0, -0.03)),
                    bars={"production": task.PAD_XY["tagged"], "tagged": task.PAD_XY["production"]})
    assert task.outcome(objects) == "ruined_production"
    assert task.score(objects) == 0.0


# -- geometry edges ----------------------------------------------------------------------------------------------

def test_a_pad_carried_above_the_bar_is_not_applied():
    """Held 6 cm over the tagged bar, which is more than the 1 cm contact band."""
    bar_top = TABLE_TOP_Z + task.PAD_SIZE[2] + task.BAR_SIZE[2]
    objects = scene(left=(task.PAD_XY["tagged"][0], task.PAD_XY["tagged"][1] + 0.03, bar_top + 0.06),
                    right=(task.PAD_XY["tagged"][0], task.PAD_XY["tagged"][1] - 0.03, bar_top + 0.06))
    assert task.applied_squarely(objects) == 0
    assert task.outcome(objects) == "applied_elsewhere"


def test_a_pad_overhanging_the_bar_is_not_square():
    """Off the bar's centre line by 2 cm, inside the 2.5 cm 'over' band but outside the 1.5 cm 'square' one."""
    objects = scene(tagged_pads=((0.02, 0.03), (0.0, -0.03)))
    assert task.applied_squarely(objects) == 1
    assert task.touching(objects, "applicator_left", task.BAR_TAGGED)
    assert task.outcome(objects) == "half_etched"


def test_a_moving_pad_does_not_count_as_applied():
    objects = scene(tagged_pads=((0.0, 0.03), (0.0, -0.03)), at_rest=False)
    assert task.applied_squarely(objects) == 0


def test_the_pads_start_in_their_wells_and_the_scene_is_at_start():
    objects = scene()
    assert task.in_wells(objects) == 2
    assert task._at_start(objects)


def test_a_bar_is_not_identified_by_the_pad_it_sits_on():
    """The tagged bar moved off its pad and etched there still counts as the tagged bar."""
    objects = scene(tagged_pads=((0.0, 0.03), (0.0, -0.03)), bars={"tagged": (0.30, 0.20)})
    assert task.outcome(objects) == "etched_tagged"
    assert task.score(objects) == 1.0


# -- the demo ----------------------------------------------------------------------------------------------------

def test_the_demo_uses_both_arms_and_never_approaches_the_production_bar():
    plan = task.demo()
    assert set(plan) == {"left", "right"}
    assert plan["left"] and plan["right"]
    for arm in plan.values():
        for wp in arm:
            assert wp.pos[1] > -0.14 + task.OVER_DY or wp.pos[0] < 0.30, \
                "a waypoint sits over the untagged bar"


def test_the_demo_places_pads_inside_the_contact_band():
    # PLACE_Z is now the TOOL CENTRE POINT height, so the carried object sits TCP_LIFT lower.
    # Measured on the instance 2026-11-20; see TCP_LIFT in task.py.
    bar_top = task.BAR_Z + task.BAR_SIZE[2] / 2
    underside = task.PLACE_Z - task.TCP_LIFT - task.APPLICATOR_SIZE[2] / 2
    assert bar_top <= underside <= bar_top + task.CONTACT_DZ
