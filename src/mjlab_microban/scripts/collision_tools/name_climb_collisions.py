"""Name every collision geom (group 3) of robot_climb.xml as <side>_<segment>_<part>[_NN]_coll."""
import re, sys
from collections import Counter

# body -> segment name (side included). Note the ankle blocks' Onshape configuration names are swapped.
SEGMENT = {
    "trunk": "trunk", "head": "head",
    "shoulder": "right_shoulder", "shoulder_2": "left_shoulder",
    "humerus__configuration_right": "right_upper_arm", "humerus__configuration_left": "left_upper_arm",
    "radius_right": "right_forearm", "radius_left": "left_forearm",
    "hip": "right_hip", "hip_2": "left_hip",
    "hip_block__configuration_right": "right_hip_block", "hip_block__configuration_left": "left_hip_block",
    "femur__configuration_right": "right_thigh", "femur__configuration_left": "left_thigh",
    "tibia__configuration_right": "right_shin", "tibia__configuration_left": "left_shin",
    "ankle_block__configuration_left": "right_ankle_block", "ankle_block__configuration_right": "left_ankle_block",
    "foot_climb": "right_foot", "foot_climb_2": "left_foot",
}
# XL330 servo boxes, by body, in file order.
BOXES = {
    "trunk": ["head_servo"],
    "humerus__configuration_right": ["shoulder_roll_servo", "elbow_servo"],
    "humerus__configuration_left": ["shoulder_roll_servo", "elbow_servo"],
    "tibia__configuration_right": ["knee_servo"],
    "tibia__configuration_left": ["knee_servo"],
}

def part_of(body, mesh, attrs):
    """(part name, is a convex-decomposition piece)."""
    if mesh is None:
        return BOXES[body].pop(0), False
    m = re.fullmatch(r"coll_(.+)_(\d\d)", mesh)
    base = m.group(1) if m else mesh
    base = re.sub(r"__configuration_(left|right)$|_mirrored$", "", base)
    if body == "trunk" and base == "arm_bearing_support":
        side = "right" if 'pos="0.00575 -0.036' in attrs else "left"
        return f"{side}_arm_bearing_support", True
    name = {
        "hull_chest": "chest", "hull_pelvis": "pelvis", "tt_main": "top", "tt_handle": "top_handle",
        "head": "shell", "dc15_a01_idle_cap_dummy": "idler_cap", "bts2_m2_6x8": "idler_screw",
        "dc15_a01_horn_idle2_dummy": "idler_horn", "hip_dbl_block": "bracket", "ankle_dbl_block": "bracket",
        "spacer": "spacer",
    }.get(base)
    if name is None:  # the segment's own decomposed shell (shoulder, humerus, radius, hip, femur, tibia, foot)
        name = "hull"
    return name, m is not None

src = open(sys.argv[1]).read().splitlines(keepends=True)
stack, out, names = [], [], []
pieces = Counter()
for line in src:
    if m := re.search(r'<body name="([^"]+)"', line):
        stack.append(m.group(1))
    g = re.search(r"<geom ([^>]*)/>", line)
    if stack and g and ('class="collision"' in line or 'group="3"' in line) and 'name="' not in line:
        body = stack[-1]
        mesh = (re.search(r'mesh="([^"]+)"', line) or [None, None])[1]
        part, piece = part_of(body, mesh, line)
        key = (SEGMENT[body], part)
        if piece:
            name = f"{SEGMENT[body]}_{part}_{pieces[key]:02d}_coll"
            pieces[key] += 1
        else:
            name = f"{SEGMENT[body]}_{part}_coll"
        names.append(name)
        line = line.replace("<geom ", f'<geom name="{name}" ', 1)
    if "</body>" in line:
        stack.pop()
    out.append(line)
# Collapse single-piece groups (e.g. the trunk chest hull) to no index.
dups = [n for n, c in Counter(names).items() if c > 1]
assert not dups, dups
open(sys.argv[2], "w").write("".join(out))
print(len(names), "geoms named")
