"""Copy the climbing model's collision approximation onto the walking model (robot/microban/robot.xml).

Per body, the climbing->walking body frame change is found from the visual meshes the two models share
(same STL geometry, placed in each body frame), and checked to agree across all of them.
"""
import re, sys, shutil, os
import numpy as np
import xml.etree.ElementTree as ET

ROBOT = sys.argv[1]  # src/mjlab_microban/robot
CLIMB = f"{ROBOT}/microban_climb/robot_climb.xml"
WALK = f"{ROBOT}/microban/robot.xml"

SKIP_BODIES = {"radius", "radius_2", "foot", "foot_2"}  # end effectors: spheres set in the env (get_spec)
# Parts whose STL differs between the two robots: use the walking robot's own mesh instead (MuJoCo
# collides with its convex hull), named like the climbing ones.
OWN_MESH = {"trunk": {"trunk_chest_coll": "chest__configuration_right", "trunk_top_coll": "trunk_top__configuration_right"},
            "head": {"head_shell_coll": "head__configuration_right"}}
DROP = {"trunk_top_handle_coll"}  # the climbing trunk top's handle

def quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
def quat_conj(q): return np.array([q[0], -q[1], -q[2], -q[3]])
def rot(q, v): return quat_mul(quat_mul(q, np.r_[0, v]), quat_conj(q))[1:]
def pose(g):
    p = np.array([float(x) for x in g.get("pos", "0 0 0").split()])
    q = np.array([float(x) for x in g.get("quat", "1 0 0 0").split()]); return p, q / np.linalg.norm(q)
def compose(a, b):  # a * b
    return a[0] + rot(a[1], b[0]), quat_mul(a[1], b[1])
def inverse(a):
    qi = quat_conj(a[1]); return -rot(qi, a[0]), qi
def fmt(v): return " ".join(f"{x:.6g}" if abs(x) > 1e-9 else "0" for x in v)

climb, walk = ET.parse(CLIMB).getroot(), ET.parse(WALK).getroot()
climb_mesh_file = {m.get("name"): m.get("file") for m in climb.iter("mesh")}
walk_mesh_names = {m.get("name") or os.path.splitext(m.get("file"))[0] for m in walk.iter("mesh")}
cbodies = {b.get("name"): b for b in climb.iter("body")}

def visuals(body):
    return [(g.get("mesh"), pose(g)) for g in body.findall("geom") if g.get("class") == "visual"]

# Meshes whose geometry is the same in both robots (checked with trimesh: same bounds and volume).
DIFFERENT = {"chest__configuration_right", "trunk_top__configuration_right", "head__configuration_right",
             "radius", "foot__configuration_right"}

new_geoms, used_meshes = {}, set()
for wb in walk.iter("body"):
    name = wb.get("name")
    if name in SKIP_BODIES:
        continue
    cb = cbodies[name]
    cv = visuals(cb)
    T = None
    for mesh, wp in visuals(wb):
        if mesh in DIFFERENT:
            continue
        matches = [cp for m, cp in cv if m == mesh]
        if not matches:
            continue
        # A mesh may appear several times in a body (e.g. xl330_without_horn): pick the match closest to
        # the transform found so far, or the first one.
        cands = [compose(wp, inverse(cp)) for cp in matches]
        if T is None:
            T = cands[0] if len(cands) == 1 else None
            if T is None:
                continue
        else:
            err = [np.linalg.norm(c[0] - T[0]) + min(np.linalg.norm(c[1] - T[1]), np.linalg.norm(c[1] + T[1])) for c in cands]
            if min(err) > 1e-4:
                # Only the head servo moved (4 mm, the walking chest is different): its box follows it below.
                assert (name, mesh) == ("trunk", "xl330_without_horn"), (name, mesh, min(err))
                print(f"  {name}: one {mesh} is {min(err)*1e3:.1f} mm off the body transform")
    if name == "head":  # its only part differs: own mesh below
        T = (np.zeros(3), np.array([1.0, 0, 0, 0]))
    assert T is not None, f"no shared visual mesh for {name}"
    lines = []
    for g in cb.findall("geom"):
        gname = g.get("name")
        if g.get("class") == "visual" or gname in DROP or gname in OWN_MESH.get(name, {}):
            continue
        Tg = T
        if gname == "trunk_head_servo_coll":  # follow the head servo (the xl330 just below the head body)
            head_servo = lambda body: min((p for m, p in visuals(body) if m == "xl330_without_horn"), key=lambda p: -p[0][2])
            Tg = compose(head_servo(wb), inverse(head_servo(cb)))
        p, q = compose(Tg, pose(g))
        attrs = {"name": gname.removesuffix("_coll") + "_collision", "class": "collision",
                 "pos": fmt(p), "quat": fmt(q)}
        for k in ("type", "size", "mesh", "rgba"):
            if g.get(k):
                attrs[k] = g.get(k)
        if g.get("mesh"):
            used_meshes.add(g.get("mesh"))
        lines.append(attrs)
    for gname, mesh in OWN_MESH.get(name, {}).items():
        (wp,) = [p for m, p in visuals(wb) if m == mesh]
        lines.append({"name": gname.removesuffix("_coll") + "_collision", "class": "collision", "pos": fmt(wp[0]),
                      "quat": fmt(wp[1]), "type": "mesh", "mesh": mesh, "rgba": "0.3 0.45 0.95 1"})
    new_geoms[name] = lines

# ---- edit the walking XML as text, to keep its layout
src = open(WALK).read().splitlines(keepends=True)
out, stack = [], []
for line in src:
    if m := re.search(r'<body name="([^"]+)"', line):
        stack.append(m.group(1))
    body = stack[-1] if stack else None
    if body in new_geoms and 'class="collision"' in line:
        continue  # replaced
    if body in new_geoms and re.search(r'<site |<body |</body>', line) and new_geoms[body] is not None \
            and not line.lstrip().startswith(f'<body name="{body}"'):
        indent = re.match(r"\s*", line).group(0)
        if line.lstrip().startswith("</body>"):
            indent += "  "
        out.append(f"{indent}<!-- collision approximation from microban_climb/robot_climb.xml -->\n")
        for a in new_geoms[body]:
            out.append(indent + "<geom " + " ".join(f'{k}="{v}"' for k, v in a.items()) + "/>\n")
        new_geoms[body] = None
    if "</body>" in line:
        stack.pop()
    if line.strip() == "</asset>":
        for mesh in sorted(used_meshes - walk_mesh_names):
            out.append(f'    <mesh name="{mesh}" file="{climb_mesh_file[mesh]}"/>\n')
            shutil.copy(f"{ROBOT}/microban_climb/assets/{climb_mesh_file[mesh]}", f"{ROBOT}/microban/assets/")
    out.append(line)
assert all(v is None for v in new_geoms.values()), [k for k, v in new_geoms.items() if v is not None]
open(WALK, "w").write("".join(out))
print("copied", len(used_meshes - walk_mesh_names), "meshes")
