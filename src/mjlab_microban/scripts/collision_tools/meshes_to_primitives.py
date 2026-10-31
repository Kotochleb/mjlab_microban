"""Replace every mesh collision geom of an MJCF by the smallest enclosing primitive (box, cylinder, capsule
or sphere, the one with the least volume), fitted to the convex hull of the mesh, in the body frame.

    python meshes_to_primitives.py in.xml out.xml [tolerance_m=0.002] [max_depth=2]
"""
import re, sys, os
from collections import defaultdict
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation

src_path, out_path = sys.argv[1:3]
root = ET.parse(src_path).getroot()
meshdir = os.path.join(os.path.dirname(src_path), root.find("compiler").get("meshdir", "."))
mesh_file = {(m.get("name") or os.path.splitext(m.get("file"))[0]): m.get("file") for m in root.iter("mesh")}

def wxyz(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat(); q = np.array([w, x, y, z]); return q if w >= 0 else -q
def fmt(v): return " ".join("0" if abs(x) < 1e-9 else f"{x:.6g}" for x in v)

TOL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.002  # max distance a primitive may stick out of the hull it replaces (m)
MAX_DEPTH = int(sys.argv[4]) if len(sys.argv) > 4 else 2  # a piece is split in two at most this many times


def candidates(hull):
    """[(volume, attrs, surface points)] of the enclosing primitives of a convex hull (body frame)."""
    out = []
    obb = hull.bounding_box_oriented
    T, ext = obb.primitive.transform, obb.primitive.extents
    out.append((np.prod(ext), dict(type="box", size=fmt(ext / 2), pos=fmt(T[:3, 3]), quat=fmt(wxyz(T[:3, :3]))),
                trimesh.creation.box(extents=ext, transform=T).vertices))
    try:
        cyl = hull.bounding_cylinder
        T, r, h = cyl.primitive.transform, cyl.primitive.radius, cyl.primitive.height
        out.append((np.pi * r * r * h, dict(type="cylinder", size=fmt([r, h / 2]), pos=fmt(T[:3, 3]), quat=fmt(wxyz(T[:3, :3]))),
                    trimesh.creation.cylinder(radius=r, height=h, sections=32, transform=T).vertices))
        # Capsule on the cylinder's axis: radius r, shortest half-length enclosing every vertex.
        local = (hull.vertices - T[:3, 3]) @ T[:3, :3]
        radial, axial = np.linalg.norm(local[:, :2], axis=1), local[:, 2]
        hl = max(0.0, np.max(np.abs(axial) - np.sqrt(np.maximum(r * r - radial ** 2, 0.0))))
        out.append((np.pi * r * r * 2 * hl + 4 / 3 * np.pi * r ** 3,
                    dict(type="capsule", size=fmt([r, hl]), pos=fmt(T[:3, 3]), quat=fmt(wxyz(T[:3, :3]))),
                    trimesh.creation.capsule(radius=r, height=2 * hl, count=[16, 16], transform=T).vertices))
    except Exception:
        pass
    c, r = hull.bounding_sphere.primitive.center, hull.bounding_sphere.primitive.radius
    out.append((4 / 3 * np.pi * r ** 3, dict(type="sphere", size=fmt([r]), pos=fmt(c)),
                trimesh.creation.icosphere(subdivisions=2, radius=r).vertices + c))
    return out


def overhang(hull, pts):
    """Largest distance of points outside a convex hull (its face planes)."""
    n, o = hull.face_normals, hull.triangles[:, 0]
    return max(0.0, float(np.max((pts @ n.T) - np.sum(n * o, axis=1))))


def fit(points, depth=0):
    """(hull volume, [(volume, attrs)]): the smallest primitive sticking out of the hull by at most TOL, or
    else the fits of the two halves of the hull, cut across the longest axis of its bounding box."""
    hull = trimesh.convex.convex_hull(points)
    cands = sorted(candidates(hull), key=lambda c: c[0])
    ok = [c for c in cands if overhang(hull, c[2]) <= TOL]
    if ok or depth == MAX_DEPTH:
        vol, attrs, _ = ok[0] if ok else min(cands, key=lambda c: overhang(hull, c[2]))
        return hull.volume, [(vol, attrs)]
    obb = hull.bounding_box_oriented.primitive
    axis = obb.transform[:3, np.argmax(obb.extents)]
    centre = obb.transform[:3, 3]
    # Halves of a convex hull: its vertices on each side, plus the points where its edges cross the plane.
    v = hull.vertices
    side = (v - centre) @ axis
    e = hull.edges_unique
    cross = e[side[e[:, 0]] * side[e[:, 1]] < 0]
    t = side[cross[:, 0]] / (side[cross[:, 0]] - side[cross[:, 1]])
    cut = v[cross[:, 0]] + t[:, None] * (v[cross[:, 1]] - v[cross[:, 0]])
    prims = []
    for keep in (side >= 0, side <= 0):
        prims += fit(np.vstack([v[keep], cut]), depth + 1)[1]
    return hull.volume, prims


def geom_points(g):
    m = trimesh.load(os.path.join(meshdir, mesh_file[g["mesh"]]), force="mesh")
    p = np.array([float(x) for x in g.get("pos", "0 0 0").split()])
    w, x, y, z = (float(v) for v in g.get("quat", "1 0 0 0").split())
    return m.vertices @ Rotation.from_quat([x, y, z, w]).as_matrix().T + p

attr_re = re.compile(r'(\w+)="([^"]*)"')
lines = open(src_path).read().splitlines(keepends=True)
out, used = [], set()
report = defaultdict(lambda: [0.0, 0.0, defaultdict(int)])
for line in lines:
    m = re.match(r'(\s*)<geom (.*?)\s*/>\s*$', line)
    if m and ('class="collision"' in line or 'group="3"' in line) and 'type="mesh"' in line:
        indent, attrs = m.group(1), dict(attr_re.findall(m.group(2)))
        hull_vol, prims = fit(geom_points(attrs))
        # Keep every attribute (name, class, contype/conaffinity, group, density/mass, rgba) but the shape.
        rest = {k: v for k, v in attrs.items() if k not in ("name", "class", "type", "mesh", "pos", "quat", "size")}
        cls = {"class": attrs["class"]} if "class" in attrs else {}
        stem, suffix = re.match(r"(.*?)(_coll(?:ision)?)$", attrs["name"]).groups()
        part = re.sub(r"_\d\d$", "", stem)
        report[part][0] += hull_vol
        for k, (vol, prim) in enumerate(prims):
            name = attrs["name"] if len(prims) == 1 else f"{stem}{chr(ord('a') + k)}{suffix}"
            new = {"name": name, **cls, **prim, **rest}
            out.append(indent + "<geom " + " ".join(f'{a}="{v}"' for a, v in new.items()) + "/>\n")
            report[part][1] += vol; report[part][2][prim["type"]] += 1
        continue
    out.append(line)
text = "".join(out)
# Drop the mesh assets no geom references any more.
referenced = set(re.findall(r'mesh="([^"]+)"', text))
def keep_mesh(l):
    mm = re.search(r'<mesh (.*?)/>', l)
    if not mm: return True
    a = dict(attr_re.findall(mm.group(1))); return (a.get("name") or os.path.splitext(a["file"])[0]) in referenced
text = "".join(l for l in text.splitlines(keepends=True) if keep_mesh(l))
open(out_path, "w").write(text)
print(f"{'part':42s} {'hull cm3':>9s} {'prims cm3':>9s}  ratio  types")
for part, (hv, pv, types) in report.items():
    print(f"{part:42s} {hv*1e6:9.2f} {pv*1e6:9.2f}  {pv/hv:5.2f}  {dict(types)}")
