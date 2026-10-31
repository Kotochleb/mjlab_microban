# SPDX-License-Identifier: GPL-3.0-or-later
"""Routes of the hackathon levels N1-N3 (and an extra N4), from a seed.

    from climbing_sim.routes import generate_wall
    wall = generate_wall(level=2, seed=5)       # a Wall: holds in climbing order + the four start holds
    wall.to_json("my_route.json")               # ClimbingSim(wall="my_route.json")

Seed 0 of levels 1-3 gives the bundled competition walls N1, N2, N3 (:func:`classic_wall`, our original generator,
ported as is). Every route is a stack of *rungs* of four holds (a narrow pair for the feet, 75 mm apart, a wide pair
for the hands, 165 mm apart), plus intermediate holds halfway between two rungs, with the robot's start stance at
the bottom (feet on the bottom rung 50 mm above the wall's bottom edge, hands about 275 mm higher). Other seeds:

    N1   the regular ladder: rungs every 75 mm, aligned (the same for every seed)
    N2   a ladder with missing holds: rungs every 105 mm + intermediate holds, ~30 % of the holds removed
    N3   the robot has to move right and left: the N2 ladder following a random walk (<= 30 mm per rung, <= 150 mm
         from the middle of the wall), ~25 % of the holds removed
    N4   (extra, not in the rulebook) the route forks at 600 mm into two branches drifting apart, built the same
         way and as high as each other: one leads to the top, the other one lacks the hand holds of one rung and
         those next to it in its upper half (a step too long for the hands), hidden among the random holes

Holes are never made around the start (start holds, the first hold above each, the first intermediate row) or the
finish, and never leave a limb a step longer than 105 mm. A hold is only added if its body does not overlap
another one or a corner marker (:func:`overlaps`, the hold's real outline) and, for the intermediate holds, if it does not block the
vertical way out of another hold's socket (:func:`exit_clear`). Our motion planner climbs N2, N3 and N4 to the top
for 16 seeds out of 16 each, and never N4's dead branch (README).

Internally the routes are built on the grid seen from the back of the board (column c is this simulator's 49 - c,
rows are the same); everything returned uses this simulator's convention (``Hold(col, row)`` = left hole seen from
the climbing side).
"""
from __future__ import annotations

import numpy as np

from .wall import EXIT_BLOCK, GRID_COLS, HOLE_PITCH, OVERLAP, WALL_WIDTH, Hold, Wall

PM = HOLE_PITCH * 1000.0                   # grid pitch, mm

# ---- the robot's opening stance and the levels, in millimetres ----------------------------------------
BOTTOM_Z_MM = 50.0                         # bottom (foot) rung above the wall's bottom edge
HAND_LEAD_MM = 275.0                       # hands above feet in the start stance
FOOT_SPACING_MM = 75.0
HAND_SPACING_MM = 175.0
RUNG_OFFSET_MM = 50.0                      # lateral shift budget of level 3
TOP_MARGIN_MM = 50.0                       # material left above the topmost rung

_GEN_Z0 = 0.0175                           # the generator's grid origin height (its rows match ours)
_GEN_ROWS = 92
_GEN_ANCHOR_COLS = GRID_COLS - 2           # first hole of a hold: the second one is 2 columns over


def pitches(mm: float) -> int:
    """Whole grid pitches nearest to ``mm`` millimetres, at least one."""
    return max(1, int(round(mm / PM)))


def _row_at(z_mm: float) -> int:
    """The grid row nearest to ``z_mm`` millimetres above the wall's bottom edge."""
    return max(0, min(_GEN_ROWS - 1, int(round((z_mm / 1000.0 - _GEN_Z0) / HOLE_PITCH))))


def _match_parity(span: int, ref: int, target_mm: float) -> int:
    """Hands and feet share one centre line: both spans must have the same parity."""
    if (span - ref) % 2 == 0:
        return span
    lo, hi = max(1, span - 1), span + 1
    return lo if abs(lo * PM - target_mm) <= abs(hi * PM - target_mm) else hi


def _pair_dc(span: int) -> tuple[int, int]:
    """Column offsets (from the centre column) of the two holds of a pair ``span`` columns apart."""
    lo = (1 - span) // 2
    return (lo, lo + span)


BOTTOM_ROW = _row_at(BOTTOM_Z_MM)
HAND_LEAD_ROWS = pitches(HAND_LEAD_MM)
FOOT_SPAN = pitches(FOOT_SPACING_MM)
HAND_SPAN = _match_parity(pitches(HAND_SPACING_MM), FOOT_SPAN, HAND_SPACING_MM)
FOOT_DC = _pair_dc(FOOT_SPAN)
HAND_DC = _pair_dc(HAND_SPAN)
OFFSET_COLS = pitches(RUNG_OFFSET_MM)
TOP_ROW = _GEN_ROWS - 1 - pitches(TOP_MARGIN_MM)
CENTRE_COL = int(np.ceil((WALL_WIDTH / 2 - 0.0175) / HOLE_PITCH - 1.0))

#: level -> (name, rung spacing in rows, lateral shift in columns)
LEVELS = {1: ("N1", pitches(75), 0), 2: ("N2", pitches(100), 0), 3: ("N3", pitches(100), OFFSET_COLS)}


def overlaps(a: Hold, b: Hold) -> bool:
    """True when the two holds share a hole or their bodies overlap."""
    return (b.col - a.col, b.row - a.row) in OVERLAP


def exit_clear(holds, cand: Hold) -> bool:
    """True when ``cand`` neither blocks nor is blocked by the way out of another hold's socket."""
    for h in holds:
        dr, dc = cand.row - h.row, cand.col - h.col
        if (dr, dc) in EXIT_BLOCK or (-dr, -dc) in EXIT_BLOCK:
            return False
    return True


def _sim(col: int, row: int) -> Hold:
    """A hold given in the generator's columns, in this simulator's convention."""
    return Hold(_GEN_ANCHOR_COLS - 1 - col, row)


def _fits(holds, cand: Hold) -> bool:
    """True when ``cand`` overlaps none of ``holds`` and no corner marker."""
    return not any(overlaps(h, cand) for h in holds) and not cand.on_markers()


def _rung_rows(pitch: int) -> list[int]:
    """Rows of the rungs, from the bottom rung up to the top, every ``pitch`` rows."""
    return list(range(BOTTOM_ROW, TOP_ROW + 1, pitch))


def start_cells(level: int) -> dict[str, tuple[int, int]]:
    """(col, row) of the four start holds of a level, in the generator's columns."""
    _, pitch, _ = LEVELS[level]
    want = BOTTOM_ROW + HAND_LEAD_ROWS + 1
    hr = min(_rung_rows(pitch), key=lambda r: abs(r - want))
    c = CENTRE_COL
    return {"right_foot": (c + FOOT_DC[0], BOTTOM_ROW), "left_foot": (c + FOOT_DC[1], BOTTOM_ROW),
            "right_hand": (c + HAND_DC[0], hr), "left_hand": (c + HAND_DC[1], hr)}


def classic_wall(level: int = 2, seed: int = 0) -> Wall:
    """Our original generator, as is: seed 0 gives the competition walls N1, N2, N3.  Only level 3 depends on the
    seed (each rung shifted sideways at random)."""
    if level not in LEVELS:
        raise ValueError(f"level must be 1, 2 or 3 (N1, N2, N3), not {level}")
    name, pitch, offset = LEVELS[level]
    rng = np.random.default_rng(seed)
    start_c = start_cells(level)
    start = {limb: _sim(c, r) for limb, (c, r) in start_c.items()}
    holds = list(start.values())
    fixed_rows = {r for _, r in start_c.values()}
    # rungs: four holds each; the rungs of the start stance are never shifted
    for row in _rung_rows(pitch):
        full = row in fixed_rows
        shift = 0 if (full or not offset) else int(rng.integers(-offset, offset + 1))
        for dc in (*FOOT_DC, *HAND_DC):
            col = CENTRE_COL + dc + shift
            if 0 <= col < _GEN_ANCHOR_COLS:
                cand = _sim(col, row)
                if _fits(holds, cand):
                    holds.append(cand)
    # levels 2-3: intermediate holds halfway between two rungs, stepped sideways until the way out is clear
    if level in (2, 3):
        rows = _rung_rows(pitch)
        for lo, hi in zip(rows, rows[1:]):
            mrow = (lo + hi) // 2
            shift = 0 if not offset else int(rng.integers(-offset, offset + 1))
            _mid_holds(holds, mrow, shift)
    return Wall(name=f"{name}_seed{seed}", holds=holds, start=start,
                description=f"level {level} ({name}) route, seed {seed}: holds in climbing order (start holds first)")


def _mid_holds(holds, mrow, shift):
    """Add the intermediate holds of row ``mrow`` to ``holds`` (classic generator), each stepped sideways until it
    fits and leaves every way out clear."""
    for dc in (*FOOT_DC, *HAND_DC):
        for extra_dc in (0, 2, -2, 3, -3):
            col = CENTRE_COL + dc + shift + (extra_dc if dc > 0 else -extra_dc)
            if not (0 <= col < _GEN_ANCHOR_COLS):
                continue
            cand = _sim(col, mrow)
            if _fits(holds, cand) and exit_clear(holds, cand):
                holds.append(cand)
                break


# ---- random variants (seed > 0) ------------------------------------------------------------------------------
#: holds removed at random (levels 2-4), never leaving a step longer than MAX_GAP_ROWS in a limb's lane
MISSING_SHARE = {2: 0.30, 3: 0.25, 4: 0.25}
MAX_GAP_ROWS = pitches(100)                # 105 mm: the longest step of the original level 2
#: levels 3-4: the route drifts sideways, one rung to the next by at most DRIFT_STEP columns
DRIFT_STEP = 2                             # 30 mm
DRIFT_MAX = 10                             # 150 mm from the middle of the wall (level 3)
#: level 4: where the route forks, how far apart the two branches end up, the dead branch's impassable gap
FORK_ROW = _row_at(600.0)
BRANCH_SHIFT = (7, 12)                     # 105-180 mm each side of the middle
LEAVE_STEP = 3                             # 45 mm per rung when a branch leaves the middle
#: column offsets of the four limb lanes from the centre column: right foot, left foot, right hand, left hand
LANE_DC = (*FOOT_DC, *HAND_DC)


class _Builder:
    """Holds placed rung by rung, each tagged with its branch and its limb lane (0-1 feet, 2-3 hands), in the
    generator's columns until :meth:`wall`."""

    def __init__(self, level, rng):
        self.level, self.rng = level, rng
        sc = start_cells(level)
        self.start = {limb: _sim(c, r) for limb, (c, r) in sc.items()}
        self.holds = list(self.start.values())
        self.tag = {h: ("main", LANE_DC.index(c - CENTRE_COL)) for h, (c, _) in zip(self.holds, sc.values())}
        self.start_rows = {r for _, r in sc.values()}

    def add(self, col, row, branch, lane, exit_check=False):
        """Add the hold (col, row) if it is on the grid and fits (and, with ``exit_check``, leaves every way out
        clear); True when added."""
        if not (0 <= col < _GEN_ANCHOR_COLS and 0 <= row < _GEN_ROWS):
            return False
        cand = _sim(col, row)
        if _fits(self.holds, cand) and (not exit_check or exit_clear(self.holds, cand)):
            self.holds.append(cand); self.tag[cand] = (branch, lane)
            return True
        return False

    def rung(self, row, shift, branch="main", exit_check=False):
        """The four holds of a rung on ``row``, shifted ``shift`` columns sideways."""
        for k, dc in enumerate(LANE_DC):
            self.add(CENTRE_COL + dc + shift, row, branch, k, exit_check)

    def mid(self, row, shift, branch="main"):
        """The intermediate holds of ``row``, shifted ``shift`` columns sideways."""
        for k, dc in enumerate(LANE_DC):
            for extra in (0, 2, -2, 3, -3):          # stepped sideways until the way out is clear
                if self.add(CENTRE_COL + dc + shift + (extra if dc > 0 else -extra), row, branch, k, exit_check=True):
                    break

    def ladder(self, rows, shifts, branch="main"):
        """Rungs on ``rows`` with their shifts, and an intermediate hold row between each two."""
        for r, sh in zip(rows, shifts):
            self.rung(r, sh, branch)
        for (lo, hi), a, b in zip(zip(rows, rows[1:]), shifts, shifts[1:]):
            self.mid((lo + hi) // 2, int(round((a + b) / 2)), branch)

    def drift(self, n, shift, lo, hi, first_step=None):
        """Shifts of n rungs: a random walk with momentum, within [lo, hi] (turns back at the limits)."""
        out, step = [], first_step if first_step is not None else int(self.rng.choice([-1, 1]))
        for _ in range(n):
            step = int(np.clip(step + self.rng.integers(-1, 2), -DRIFT_STEP, DRIFT_STEP))
            if not lo <= shift + step <= hi:
                step = -step
            shift = int(np.clip(shift + step, lo, hi))
            out.append(shift)
        return out

    def remove_some(self, share, keep_top_of=("main",)):
        """Remove about ``share`` of the holds at random: never a start hold, the first hold above it or the holds
        just above the feet, never the last rung and a half of the branches in ``keep_top_of``, never leaving a step
        longer than MAX_GAP_ROWS in a lane."""
        start = set(self.start.values())
        groups = {}
        for h in self.holds:
            groups.setdefault(self.tag[h], []).append(h)
        first = {min((o for o in groups[self.tag[s]] if o.row > s.row), key=lambda o: o.row, default=None)
                 for s in start}
        tops = {b: max(h.row for h in self.holds if self.tag[h][0] == b) for b in keep_top_of
                if any(self.tag[h][0] == b for h in self.holds)}
        keep = set(self.holds)
        for i in self.rng.permutation(len(self.holds)):
            h = self.holds[i]
            b = self.tag[h][0]
            # (the intermediate holds just above the feet: the robot's first moves step onto them)
            if (h in start or h in first or h.row <= BOTTOM_ROW + MAX_GAP_ROWS // 2
                    or (b in tops and h.row >= tops[b] - MAX_GAP_ROWS - MAX_GAP_ROWS // 2)):
                continue
            if self.rng.random() >= share:
                continue
            rows = sorted(o.row for o in groups[self.tag[h]] if o in keep and o != h)
            below = max((r for r in rows if r < h.row), default=None)
            above = min((r for r in rows if r > h.row), default=None)
            if below is not None and above is not None and above - below <= MAX_GAP_ROWS:
                keep.discard(h)
        self.holds = [h for h in self.holds if h in keep]

    def wall(self, name, description) -> Wall:
        """The holds placed, as a Wall."""
        return Wall(name=name, holds=list(self.holds), start=dict(self.start), description=description)


_SHIFT_LO = -(CENTRE_COL + min(LANE_DC))                         # every hold stays on the wall
_SHIFT_HI = _GEN_ANCHOR_COLS - 1 - (CENTRE_COL + max(LANE_DC))


def _level2(seed, rng) -> Wall:
    """N2: the level-2 ladder with holds missing at random."""
    b = _Builder(2, rng)
    rows = _rung_rows(LEVELS[2][1])
    b.ladder(rows, [0] * len(rows))
    b.remove_some(MISSING_SHARE[2])
    return b.wall(f"N2_seed{seed}", f"level 2 (N2), seed {seed}: a ladder with holds missing at random")


def _level3(seed, rng) -> Wall:
    """N3: the ladder drifting right and left above the start, holds missing at random."""
    b = _Builder(3, rng)
    rows = _rung_rows(LEVELS[3][1])
    n_fixed = sum(r in b.start_rows for r in rows)
    shifts = [0] * n_fixed + b.drift(len(rows) - n_fixed, 0, max(-DRIFT_MAX, _SHIFT_LO), min(DRIFT_MAX, _SHIFT_HI))
    b.ladder(rows, shifts)
    b.remove_some(MISSING_SHARE[3])
    return b.wall(f"N3_seed{seed}", f"level 3 (N3), seed {seed}: wanders right and left, holds missing at random")


def _level4(seed, rng) -> Wall:
    """N4: the forked route (:func:`_build4`)."""
    return _build4(rng).wall(f"N4_seed{seed}", f"level 4 (N4), seed {seed}: the route forks; one branch leads "
                                                      f"to the top, the other one cannot be climbed to the end")


def _build4(rng) -> _Builder:
    """A trunk up to FORK_ROW (600 mm), then two branches drifting apart, right and left, built the same way (rungs,
    intermediate holds, holds missing at random) and as high as each other.  The dead one has a trap in its upper
    half: on one rung, the two HAND holds and the hand holds between it and its neighbours are missing, so the hands
    face a 210 mm step (beyond their reach, MAX_GAP_ROWS) while the feet holds stay -- among the random holes it
    looks like any other, and only trying it (or measuring every gap) tells the branches apart."""
    b = _Builder(3, rng)
    pitch = LEVELS[3][1]
    rows = _rung_rows(pitch)
    trunk = [r for r in rows if r <= FORK_ROW]
    after = [r for r in rows if r > FORK_ROW]
    b.ladder(trunk, [0] * len(trunk))
    side = int(rng.choice([-1, 1]))
    lo, hi = BRANCH_SHIFT

    def branch_shifts(sgn):
        """Shifts of the branch on side ``sgn`` (+1 / -1): leave the middle LEAVE_STEP columns per rung, then
        wander inside the branch's band."""
        out, sh = [], 0
        while abs(sh) < lo and len(out) < len(after):
            sh += sgn * LEAVE_STEP; out.append(sh)
        band = (sgn * lo, sgn * hi) if sgn > 0 else (sgn * hi, sgn * lo)
        band = (max(band[0], _SHIFT_LO), min(band[1], _SHIFT_HI))
        return out + b.drift(len(after) - len(out), sh, *band, first_step=sgn)

    good, dead = branch_shifts(side), branch_shifts(-side)
    # rung by rung, alternating which branch is placed first, so that neither gets all the room where they meet
    for k, r in enumerate(after):
        for br, sh in ((("good", good[k]), ("dead", dead[k])) if k % 2 == 0 else (("dead", dead[k]), ("good", good[k]))):
            b.rung(r, sh, br, exit_check=True)
    for br, shifts in (("good", good), ("dead", dead)):
        allr, alls = [trunk[-1]] + after, [0] + shifts
        for k in range(len(allr) - 1):
            b.mid((allr[k] + allr[k + 1]) // 2, int(round((alls[k] + alls[k + 1]) / 2)), br)
    b.remove_some(MISSING_SHARE[4], keep_top_of=("good", "dead"))
    # the trap: no HAND hold (lanes 2-3) on one rung of the dead branch's upper half, nor on the intermediate rows
    # next to it; the feet holds stay, so the branch looks like the other one
    dead_rungs = after[len(after) // 2:-2] or after[-2:-1]
    kill = int(rng.choice(dead_rungs))
    b.holds = [h for h in b.holds
               if not (b.tag[h] in (("dead", 2), ("dead", 3)) and abs(h.row - kill) <= pitch // 2 + 1)]
    return b


def generate_wall(level: int = 2, seed: int = 0) -> Wall:
    """A route of level 1-4 (N1-N4): holds in climbing order (start holds first), the four start holds set.  The
    same (level, seed) always gives the same route.

    * seed 0, levels 1-3: the competition walls N1, N2, N3 (:func:`classic_wall`);
    * level 1, any seed: the regular ladder (rungs every 75 mm);
    * level 2: the level-2 ladder (rungs every 105 mm and intermediate holds) with holds missing at random;
    * level 3: the same, wandering right and left (up to 150 mm from the middle, 30 mm per rung at most);
    * level 4: the route forks above 600 mm into two branches drifting apart, as high as each other; one of them
      leads to the top, the other one has a step too long for the hands, hidden among the random holes.
    Holes are never put around the start and the finish, and never leave a step longer than 105 mm for a limb.
    """
    if level not in (1, 2, 3, 4):
        raise ValueError(f"level must be 1, 2, 3 or 4 (N1-N4), not {level}")
    if level == 1 or (seed == 0 and level <= 3):
        return classic_wall(level, seed)
    rng = np.random.default_rng([level, seed])
    return {2: _level2, 3: _level3, 4: _level4}[level](seed, rng)
