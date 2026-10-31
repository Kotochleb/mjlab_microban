"""Contacts (and CPU step time) of the climbing env scene in the climb start pose. Usage: python climb_start_contacts.py LABEL"""
import time, sys, mujoco, numpy as np
import mjlab_microban.tasks
from mjlab.scene import Scene
from mjlab_microban.tasks.climbing.microban_climbing_env_cfg import make_microban_climbing_env_cfg
cfg = make_microban_climbing_env_cfg()
m = Scene(cfg.scene, device="cpu").compile(); d = mujoco.MjData(m)
rob = cfg.scene.entities["robot"].init_state
d.qpos[:] = m.qpos0
fj = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.joint(j).name.startswith("robot/")][0]
a = m.jnt_qposadr[fj]; d.qpos[a:a+3] = rob.pos; d.qpos[a+3:a+7] = rob.rot
for n, v in rob.joint_pos.items(): d.qpos[m.jnt_qposadr[m.joint("robot/" + n).id]] = v
mujoco.mj_forward(m, d)
name = lambda g: m.geom(g).name or m.body(m.geom_bodyid[g]).name
pairs = {}
for c in d.contact[:d.ncon]:
    k = tuple(sorted((name(c.geom1), name(c.geom2)))); pairs[k] = max(pairs.get(k, 0), -c.dist)
print(sys.argv[1], "geoms", m.ngeom, "contacts", d.ncon)
for k, v in sorted(pairs.items(), key=lambda x: -x[1]): print(f"   {v*1e3:6.2f} mm  {k}")
t = time.perf_counter()
for _ in range(500): mujoco.mj_step(m, d)
print("   CPU mj_step", round((time.perf_counter() - t) / 500 * 1e6, 1), "us")
