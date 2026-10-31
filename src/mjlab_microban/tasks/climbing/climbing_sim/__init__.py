"""The wall / route part of the organisers' simulator, climbing_sim (hackathon checkout ../simulator, commit
6c180ff), copied here so the climbing task does not need the simulator: wall.py, routes.py and aruco.py
unchanged, with the assets they read (assets/hold/outline.json), the bundled walls (walls/N1-N3.json) and the
hold meshes (assets/hold: hold_visual.stl for tasks/climbing/wall.py; the hulls, skins and cup spheres for
scripts/view_hold_boxes.py). climbing_sim is GPL-3.0 (LICENSE).

Its own __init__ (which imports the robot simulator, microban_sim) is not copied: only numpy is needed.
To update: copy those files again from the simulator checkout.
"""
