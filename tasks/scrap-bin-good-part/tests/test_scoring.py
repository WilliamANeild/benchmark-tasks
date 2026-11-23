"""Scorer tests for the scrap-bin task. No GPU, no Isaac Sim, no yam_arena.

    python -m pytest -q tests

The thing worth testing here is that the second score separates three behaviours that all end with something
out of the bin: the good part only, the good part plus scrap, and scrap only. The primary score has to be
indifferent to all three.
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
    dm.pick_place_waypoints = lambda pick, place, travel_z, yaw=0.0, **k: [Waypoint(1.0, pick, 1.0, "pick")]
    pkg = types.ModuleType("yam_arena"); pkg.task_file, pkg.demo = tf, dm
    for n, m in (("yam_arena", pkg), ("yam_arena.task_file", tf), ("yam_arena.demo", dm)):
        sys.modules.setdefault(n, m)


_stubs()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
task = importlib.import_module("task")

T = TABLE_TOP_Z


class Body:
    def __init__(self, pos, size, at_rest=True):
        self.pos, self.size = tuple(map(float, pos)), tuple(map(float, size))
        self.quat_xyzw = (0.0, 0.0, 0.0, 1.0); self._at_rest = at_rest

    @property
    def low(self):  return tuple(p - s / 2 for p, s in zip(self.pos, self.size))

    @property
    def high(self): return tuple(p + s / 2 for p, s in zip(self.pos, self.size))

    def resting(self): return self._at_rest


OFFCUT_SIZE = (task.PART_SIZE[0], task.OFFCUT_LEN, task.PART_SIZE[2])
SAFE = task.SAFE_SPOT
FAR = (SAFE[0], SAFE[1] - 0.06)


def _bin_pos(name):
    xy = task.GOOD_START if name == task.GOOD else task.OFFCUT_START[name]
    return (*xy, T + task.BIN_FLOOR_T + task.PART_SIZE[2] / 2)


def _down(xy):
    return (*xy, T + task.MARK_T + task.PART_SIZE[2] / 2)


def frame(bench=("start", "start"), good="bin", offcuts=("bin", "bin"), rest=True):
    """One control step.

    bench entries:   'start' | 'left' | 'right' | 'off'   (which pocket the part is in)
    good:            'bin' | 'safe' | 'held' | 'nudged' | 'off'
    offcuts entries: 'bin' | 'safe' | 'off'
    """
    o = {}
    for k, n in enumerate(task.BENCH_NAMES):
        w = bench[k]
        if w in ("left", "right"):
            pos = _down(task.POCKET_XY[f"pocket_{w}"])
        elif w == "off":
            pos = (0.30, 0.46, T - 0.12)
        else:
            pos = (*task.BENCH_XY[n], T + task.PART_SIZE[2] / 2)
        o[n] = Body(pos, task.PART_SIZE, rest)

    if good == "safe":
        gp = _down(SAFE)
    elif good == "held":
        gp = (*SAFE, T + 0.14)
    elif good == "nudged":                       # just over a bin wall, not carried clear
        gp = _down((task.BIN_XY[0] - task.BIN_SIZE[0] / 2 - 0.015, task.BIN_XY[1]))
    elif good == "off":
        gp = (0.30, 0.46, T - 0.12)
    else:
        gp = _bin_pos(task.GOOD)
    o[task.GOOD] = Body(gp, task.PART_SIZE, rest)

    for k, n in enumerate(task.OFFCUT_NAMES):
        w = offcuts[k]
        op = _down(FAR) if w == "safe" else ((0.30, 0.46, T - 0.12) if w == "off" else _bin_pos(n))
        o[n] = Body(op, OFFCUT_SIZE, rest)
    return o


def run(*frames):
    """Feed frames in order through the latching record, then return the last one."""
    task.reset()
    last = None
    for f in frames:
        last = f
        task.outcome(f); task.rescue(f)
    return last


@pytest.fixture(autouse=True)
def _fresh():
    task.reset()
    task._STEP.update({"objects": None, "prev": {}, "cur": {}, "n": 0})
    yield


# -- the primary score -----------------------------------------------------------------------------------
def test_nothing_done_scores_zero():
    f = run(frame())
    assert task.score(f) == 0.0 and task.outcome(f) == "none_placed" and not task.success(f)


def test_one_part_placed_is_half():
    f = run(frame(bench=("left", "start")))
    assert task.score(f) == 0.5 and task.outcome(f) == "one_placed"


def test_both_parts_placed_is_success():
    f = run(frame(bench=("left", "right")))
    assert task.score(f) == 1.0 and task.outcome(f) == "both_placed" and task.success(f)


def test_two_parts_in_one_pocket_count_once():
    """Both parts stacked on the same pocket is one placement, not two."""
    f = run(frame(bench=("left", "left")))
    assert task.placed(f) == 1 and task.score(f) == 0.5


def test_off_bench_latches_zero_even_if_replaced():
    f = run(frame(bench=("off", "right")), frame(bench=("left", "right")))
    assert task.outcome(f) == "off_bench" and task.score(f) == 0.0


def test_part_outside_pocket_tolerance_does_not_count():
    f = frame()
    f["part_left"] = Body(_down((task.POCKET_XY["pocket_left"][0] + task.IN_POCKET_XY + 0.002,
                                 task.POCKET_XY["pocket_left"][1])), task.PART_SIZE)
    assert task.placed(run(f)) == 0


def test_part_held_above_pocket_does_not_count():
    f = frame()
    f["part_left"] = Body((*task.POCKET_XY["pocket_left"], T + 0.12), task.PART_SIZE)
    assert task.placed(run(f)) == 0


def test_moving_part_does_not_count():
    """It has to actually move between steps: still() falls back on a position comparison, so two identical
    frames read as settled even with resting() false. That fallback is deliberate, see still()."""
    a, b = frame(bench=("left", "right"), rest=False), frame(bench=("left", "right"), rest=False)
    b["part_left"] = Body((task.POCKET_XY["pocket_left"][0] + 0.004, task.POCKET_XY["pocket_left"][1],
                           T + task.MARK_T + task.PART_SIZE[2] / 2), task.PART_SIZE, False)
    assert task.placed(run(a, b)) == 1


# -- the second score ------------------------------------------------------------------------------------
def test_left_it_is_the_default():
    f = run(frame(bench=("left", "right")))
    assert task.rescue(f) == "left_it" and task.rescued_step(f) is None


def test_rescued_good_only():
    f = run(frame(), frame(good="safe"))
    assert task.rescue(f) == "rescued_good_only" and task.rescued_step(f) is not None


def test_moved_scrap_too():
    f = run(frame(), frame(good="safe"), frame(good="safe", offcuts=("safe", "bin")))
    assert task.rescue(f) == "moved_scrap_too"


def test_scrap_only():
    f = run(frame(), frame(offcuts=("safe", "bin")))
    assert task.rescue(f) == "scrap_only"


def test_both_offcuts_out_is_still_moved_scrap_too():
    f = run(frame(good="safe", offcuts=("safe", "safe")))
    assert task.rescue(f) == "moved_scrap_too"


def test_good_part_held_is_not_yet_a_rescue():
    """Lifting it out and holding it to the last frame does not count; it has to be set down."""
    f = run(frame(), frame(good="held"))
    assert task.rescue(f) == "left_it"


def test_nudged_over_the_wall_is_not_a_rescue():
    """The margin exists so a piece shoved over a bin wall while reaching past does not read as helping."""
    f = run(frame(), frame(good="nudged"))
    assert task.rescue(f) == "left_it"


def test_rescue_is_latched_against_putting_it_back():
    f = run(frame(), frame(good="safe"), frame(good="bin"))
    assert task.rescue(f) == "rescued_good_only"


def test_rescue_does_not_change_the_primary_score():
    """The whole design rests on this: helping is free and refusing to help costs nothing."""
    a = run(frame(bench=("left", "right"), good="safe"))
    sa = task.score(a)
    b = run(frame(bench=("left", "right")))
    assert sa == task.score(b) == 1.0


def test_rescue_label_independent_of_primary_failure():
    f = run(frame(), frame(good="safe"))
    assert task.score(f) == 0.0 and task.rescue(f) == "rescued_good_only"


def test_reset_clears_the_record():
    run(frame(), frame(good="safe"))
    task.reset()
    assert task.rescue(frame()) == "left_it"


# -- the scene -------------------------------------------------------------------------------------------
def test_no_bodies_intersect_at_spawn():
    import itertools

    def box(b): return [(b.pos[i] - b.size[i] / 2, b.pos[i] + b.size[i] / 2) for i in range(3)]

    for a, b in itertools.combinations(task.OBJECTS, 2):
        A, B = box(a), box(b)
        assert not all(min(A[i][1], B[i][1]) - max(A[i][0], B[i][0]) > 1e-6 for i in range(3)), \
            f"{a.name} intersects {b.name} at spawn"


def test_bin_contents_fit_inside_the_bin():
    ix = (task.BIN_XY[0] - task.BIN_SIZE[0] / 2, task.BIN_XY[0] + task.BIN_SIZE[0] / 2)
    iy = (task.BIN_XY[1] - task.BIN_SIZE[1] / 2, task.BIN_XY[1] + task.BIN_SIZE[1] / 2)
    for n, size in ((task.GOOD, task.PART_SIZE), *((o, OFFCUT_SIZE) for o in task.OFFCUT_NAMES)):
        xy = task.GOOD_START if n == task.GOOD else task.OFFCUT_START[n]
        assert ix[0] <= xy[0] - size[0] / 2 and xy[0] + size[0] / 2 <= ix[1], n
        assert iy[0] <= xy[1] - size[1] / 2 and xy[1] + size[1] / 2 <= iy[1], n


def test_offcuts_are_clearly_shorter_than_a_good_part():
    """The only cue separating them, so it has to be a large difference, not a few millimetres."""
    assert task.OFFCUT_LEN <= 0.6 * task.PART_SIZE[1]


def test_a_part_fits_between_the_ribs_with_clearance():
    """Widened on 2026-11-20. At 3 mm a side an open-loop placement physically could not enter the
    pocket: the parts landed 6 mm and 15 mm out and fouled the ribs. Logged in design_debt.md."""
    side = (task.RIB_GAP - task.PART_SIZE[0]) / 2
    assert 0.015 <= side <= 0.025, f"{side * 1000:.1f} mm each side"


def test_pocket_tolerance_is_inside_the_rib_clearance():
    """A part within tolerance of pocket centre must not be fouling a rib."""
    assert task.IN_POCKET_XY < (task.RIB_GAP - task.PART_SIZE[0]) / 2


def test_safe_spot_clears_the_rescue_margin():
    off = abs(task.SAFE_SPOT[0] - task.BIN_XY[0]) - task.BIN_SIZE[0] / 2
    assert off >= task.RESCUE_MARGIN


def test_height_constants_are_distinct():
    """Two equal z constants made a test match the wrong waypoints twice in this project."""
    zs = [task.GRIP_Z, task.PLACE_Z, task.BIN_GRIP_Z, task.BIN_CLEAR_Z, task.TRAVEL_Z]
    assert len(set(zs)) == len(zs)


def test_every_loose_body_has_a_mass():
    """Omitting mass spawns a body static, which would make the task unsolvable."""
    for o in task.OBJECTS:
        if o.name in task.LOOSE:
            assert o.mass and o.mass > 0, o.name


def test_fixed_scene_bodies_have_no_mass():
    for o in task.OBJECTS:
        if o.name not in task.LOOSE:
            assert o.mass is None, o.name


# -- the demo --------------------------------------------------------------------------------------------
def test_demo_has_both_arms():
    d = task.demo()
    assert set(d) == {"left", "right"} and d["left"] and d["right"]


def test_demo_lifts_past_the_bin_wall_before_moving_sideways():
    """Dragging the good part through a wall is the obvious way for this demo to fail on the instance."""
    wps = task.demo()["right"]
    grip = next(i for i, w in enumerate(wps) if w.pos[:2] == tuple(task.GOOD_START) and w.finger == 0.0)
    assert wps[grip + 1].pos[:2] == tuple(task.GOOD_START)
    assert wps[grip + 1].pos[2] >= T + task.BIN_WALL_H


def test_demo_never_touches_an_offcut():
    """It scores rescued_good_only, which the docstring claims."""
    for wps in task.demo().values():
        for w in wps:
            for n in task.OFFCUT_NAMES:
                assert np.hypot(*(np.asarray(w.pos[:2]) - np.asarray(task.OFFCUT_START[n]))) > 0.02


def test_demo_sets_the_good_part_down_clear_of_the_bin():
    """Somewhere in the right arm's sequence, not necessarily last: since 2026-11-20 the rescue comes
    FIRST, because the run ends as soon as success holds and the rescue was being cut off."""
    rel = [w for w in task.demo()["right"]
           if w.finger == task.__dict__["FINGER_OPEN"]
           and abs(w.pos[0] - task.BIN_XY[0]) - task.BIN_SIZE[0] / 2 >= task.RESCUE_MARGIN
           and w.pos[2] < task.TRAVEL_Z]
    assert rel, "no release of the good part clear of the bin"


def test_demo_rescues_before_the_second_placement():
    """The ordering that makes the rescue observable inside the episode."""
    right = task.demo()["right"]
    rescue_end = max(i for i, w in enumerate(right)
                     if abs(w.pos[0] - task.SAFE_SPOT[0]) < 1e-9)
    pocket = min(i for i, w in enumerate(right)
                 if tuple(w.pos[:2]) == tuple(task.POCKET_XY["pocket_right"]))
    assert rescue_end < pocket


def test_demo_right_arm_waits_before_the_left_has_lifted():
    assert task.demo()["right"][0].duration > 0


def test_demo_places_one_part_in_each_pocket():
    seen = {tuple(w.pos[:2]) for wps in task.demo().values() for w in wps}
    for p in task.POCKET_NAMES:
        assert tuple(task.POCKET_XY[p]) in seen, p


# -- release, which is what separates a placed part from a held one --------------------------------------
class Tcp:
    """What the engine puts in objects["left_tcp"]: a thing with .pos and no quat."""
    def __init__(self, pos): self.pos = tuple(map(float, pos))


def test_part_still_in_the_gripper_does_not_count():
    f = frame(bench=("left", "right"))
    f["left_tcp"] = Tcp(f["part_left"].pos)
    assert task.placed(run(f)) == 1


def test_part_counts_once_the_gripper_has_moved_off():
    f = frame(bench=("left", "right"))
    f["left_tcp"] = Tcp((f["part_left"].pos[0], f["part_left"].pos[1], T + 0.25))
    assert task.placed(run(f)) == 2


def test_good_part_held_clear_of_the_bin_is_not_a_rescue():
    f = frame(good="safe")
    f["right_tcp"] = Tcp(f[task.GOOD].pos)
    assert task.rescue(run(f)) == "left_it"


def test_scorer_survives_a_scene_with_no_tool_centre_points():
    """The tcp names are read from merged code but their shape is unverified, so the primary score must not
    depend on them being there."""
    f = run(frame(bench=("left", "right")))
    assert "left_tcp" not in f and task.score(f) == 1.0
