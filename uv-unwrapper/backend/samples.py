"""Procedural sample meshes (each well under the 200-triangle limit)."""

import numpy as np


def plane_grid(n=8):
    """Subdivided square in the z=0 plane, oriented +z. A disk already."""
    return plane_rect(n, n)


def plane_rect(nx=3, ny=5):
    """Rectangular nx-1 by ny-1 cell grid in z=0 (nx columns, ny rows).

    Used by the atlas sample so a wide island can be shown placed with the
    same common scale as square islands.
    """
    verts, faces = [], []
    for i in range(nx):
        for j in range(ny):
            verts.append((i, j, 0.0))

    def vid(i, j):
        return i * ny + j

    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            faces.append((a, b, d))
            faces.append((b, c, d))
    return np.array(verts, dtype=float), np.array(faces, dtype=np.int64)


CUBE_VERTS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=float,
)

# all faces outward-oriented
CUBE_FACES = np.array(
    [
        [0, 3, 2],
        [0, 2, 1],  # z=0 -z
        [4, 5, 6],
        [4, 6, 7],  # z=1 +z
        [0, 1, 5],
        [0, 5, 4],  # y=0 -y
        [2, 3, 7],
        [2, 7, 6],  # y=1 +y
        [1, 2, 6],
        [1, 6, 5],  # x=1 +x
        [0, 4, 7],
        [0, 7, 3],  # x=0 -x
    ],
    dtype=np.int64,
)


def cube_spanning_tree_seams():
    """Open the cube to a disk by cutting the complement of a dual
    spanning tree (5 uncut hinges -> 7 cut edges).

    Uncut hinges: B-W (0,3), B-S (0,1), S-T (4,5), S-E (1,5), E-N (2,6)
    form a tree on the six faces.
    """
    return [
        (1, 2),
        (2, 3),  # bottom: B-E, B-N
        (5, 6),
        (6, 7),
        (4, 7),  # top:    T-E, T-N, T-W
        (3, 7),
        (0, 4),  # vertical: N-W, S-W
    ]


def cube_closed():
    return CUBE_VERTS.copy(), CUBE_FACES.copy()


def octahedron():
    """Closed sphere with no seams -- witness for 'closed_surface'."""
    verts = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )
    faces = np.array(
        [
            [4, 0, 2],
            [4, 2, 1],
            [4, 1, 3],
            [4, 3, 0],
            [5, 2, 0],
            [5, 1, 2],
            [5, 3, 1],
            [5, 0, 3],
        ],
        dtype=np.int64,
    )
    return verts, faces


def get_sample(name):
    if name == "plane":
        v, f = plane_grid(8)
        return v.tolist(), f.tolist(), []
    if name == "cube_cut":
        v, f = cube_closed()
        return v.tolist(), f.tolist(), cube_spanning_tree_seams()
    if name == "octahedron_closed":
        v, f = octahedron()
        return v.tolist(), f.tolist(), []
    if name == "plane_cut":
        # legal dead-end slit: outer edge (0,1) then interior edge (1,9)
        v, f = plane_grid(8)
        return v.tolist(), f.tolist(), [(0, 1), (1, vid2(8, 1, 1))]
    if name == "cube_closed":
        v, f = cube_closed()
        return v.tolist(), f.tolist(), []
    raise KeyError(name)


def vid2(n, i, j):
    return i * n + j
