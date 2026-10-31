# SPDX-License-Identifier: GPL-3.0-or-later
"""The rulebook's ArUco markers (dictionary DICT_4X4_50): IDs 1-4 at the wall corners.

A wall marker is a 100 x 100 mm black code (6 x 6 cells: a black border cell + 4 x 4 data bits) on a
white plate of 136 x 136 mm (18 mm white border), the plate's outer corner on the wall's corner.

The bit patterns are the genuine DICT_4X4_50 ones (checked with OpenCV's detector when they were
generated for the rulebook figures).
"""
from __future__ import annotations

import numpy as np

CODE = 0.100          # black code side (m)
BORDER = 0.018        # white border (m)
PLATE = CODE + 2 * BORDER

# white cells (col, row) of the 6 x 6 grid, row 0 at the BOTTOM of the marker
_WHITE = {
    1: "1/3,2/3,3/3,4/3,1/2,4/2,1/1,3/1",
    2: "3/4,4/4,3/3,4/3,3/2,1/1,2/1,4/1",
    3: "1/4,4/4,1/3,4/3,2/2,2/1,3/1",
    4: "2/4,4/4,2/3,1/2,4/2,1/1,2/1,3/1",
}

# corner marker positions: id -> (which corner of the wall)
CORNERS = {1: "top-left", 2: "top-right", 3: "bottom-left", 4: "bottom-right"}


def cells(marker_id: int) -> np.ndarray:
    """6 x 6 boolean array, True = white, row 0 at the TOP (image order)."""
    g = np.zeros((6, 6), bool)
    for cr in _WHITE[marker_id].split(","):
        c, r = (int(v) for v in cr.split("/"))
        g[5 - r, c] = True
    return g


def image(marker_id: int, px_per_cell: int = 40, border: bool = True) -> np.ndarray:
    """uint8 RGB image of the marker (with its white plate when border=True), top row first."""
    g = np.kron(cells(marker_id), np.ones((px_per_cell, px_per_cell), bool))
    if border:
        b = int(round(px_per_cell * 6 * BORDER / CODE))
        g = np.pad(g, b, constant_values=True)
    img = np.where(g, 255, 0).astype(np.uint8)
    return np.repeat(img[..., None], 3, axis=2)


def corner_plate_origin(marker_id: int, wall_width: float, wall_height: float) -> np.ndarray:
    """Wall-frame (x right, y up) position of the lower-left corner of the marker's white plate."""
    corner = CORNERS[marker_id]
    x = 0.0 if "left" in corner else wall_width - PLATE
    y = 0.0 if "bottom" in corner else wall_height - PLATE
    return np.array([x, y])
