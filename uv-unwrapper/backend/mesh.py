"""Mesh loading, validation and UV-vertex splitting along user seams.

Conventions
-----------
* A corner is (face f, local index j in 0..2), i.e. the "face sector" of a 3D
  vertex.  Corners keep their own identity even when they share the same 3D
  position.
* Splitting welds corners across an *un-cut* interior edge (both endpoints are
  merged per-edge).  Across a seam edge, and along boundary edges, the corners
  stay distinct -- this is exactly what makes a "cut vertex" appear as several
  different UV vertices instead of being deduplicated by 3D coordinate.
"""

from dataclasses import dataclass, field

import numpy as np


class MeshError(ValueError):
    """Raised for malformed / unsupported input meshes."""


@dataclass
class SplitMesh:
    # --- source data (unchanged) ---
    vertices: np.ndarray  # (V, 3) float
    faces: np.ndarray  # (F, 3) int, indices into vertices
    seam_edges: tuple  # tuple of (a, b) canonical (min,max)
    # --- split result ---
    corner_uv: np.ndarray  # (F, 3) int, UV vertex id per corner
    orig_of_uv: np.ndarray  # (U,) int, original 3D vertex id
    corners_of_uv: list  # list[ list[(face, local)] ]
    uv_vertices: np.ndarray  # (U, 3) float, 3D position of each UV vertex
    # --- topology on the split mesh ---
    face_adj: list  # list[ F x 3 ] -> neighbour face id or -1
    boundary_edges: list  # list[ (uv_a, uv_b) ] directed half-edges

    @property
    def n_faces(self):
        return self.faces.shape[0]

    @property
    def n_uv(self):
        return self.uv_vertices.shape[0]

    def is_boundary_uv(self, u):
        return any(u in (a, b) for a, b in self.boundary_edges)

    def boundary_uvs(self):
        return sorted({u for e in self.boundary_edges for u in e})


def _validate_input(vertices, faces):
    verts = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise MeshError("vertices must be an (V,3) array")
    if f.ndim != 2 or f.shape[1] != 3:
        raise MeshError("faces must be an (F,3) array")
    n = verts.shape[0]
    if n < 3 or f.shape[0] < 1:
        raise MeshError("mesh needs at least 3 vertices and 1 face")
    if f.shape[0] > 200:
        raise MeshError("this tool handles at most 200 triangles (got %d)" % f.shape[0])
    if np.any(f < 0) or np.any(f >= n):
        raise MeshError("face index out of range")
    for k in range(f.shape[0]):
        if f[k, 0] == f[k, 1] or f[k, 1] == f[k, 2] or f[k, 2] == f[k, 0]:
            raise MeshError("degenerate face %d: repeated vertex index" % k)
    # geometric degeneracy
    p = verts[f]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    if np.any(np.linalg.norm(nrm, axis=1) < 1e-12):
        bad = int(np.argmin(np.linalg.norm(nrm, axis=1)))
        raise MeshError("degenerate (zero-area) face %d" % bad)
    return verts, f


def _check_connected_and_oriented(verts, faces):
    """Adjacency checks on the *un-split* input mesh.

    The task guarantees a connected, consistently oriented mesh; we verify it
    so that bad input is reported honestly instead of producing garbage.
    """
    n_faces = faces.shape[0]
    edge_faces = {}
    for fi, tri in enumerate(faces):
        for j in range(3):
            a, b = int(tri[j]), int(tri[(j + 1) % 3])
            key = (min(a, b), max(a, b))
            edge_faces.setdefault(key, []).append((fi, (a, b)))

    # manifoldness + orientation consistency for interior edges
    adj = [[] for _ in range(n_faces)]
    for key, inc in edge_faces.items():
        if len(inc) > 2:
            raise MeshError("non-manifold edge %s: %d incident faces" % (key, len(inc)))
        if len(inc) == 2:
            (f0, d0), (f1, d1) = inc
            # consistent orientation: the two faces traverse the edge in
            # opposite directions
            if d0 == d1:
                raise MeshError(
                    "faces %d and %d traverse edge %s in the same direction; "
                    "orientation is inconsistent" % (f0, f1, key)
                )
            adj[f0].append(f1)
            adj[f1].append(f0)

    # connectivity by face adjacency
    seen = {0}
    stack = [0]
    while stack:
        cur = stack.pop()
        for nb in adj[cur]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    if len(seen) != n_faces:
        missing = sorted(set(range(n_faces)) - seen)
        raise MeshError(
            "mesh is not connected (e.g. faces %s are in another "
            "component)" % missing[:5]
        )
    return edge_faces


def split_mesh(vertices, faces, seam_edges):
    """Split vertices into UV vertices according to the seam edges."""
    verts, faces = _validate_input(vertices, faces)
    edge_faces = _check_connected_and_oriented(verts, faces)
    n_faces = faces.shape[0]

    seam = set()
    for e in seam_edges:
        a, b = int(e[0]), int(e[1])
        key = (min(a, b), max(a, b))
        if key not in edge_faces:
            raise MeshError("seam edge %s is not a mesh edge" % (key,))
        seam.add(key)

    n_corners = 3 * n_faces
    parent = list(range(n_corners))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    # weld across interior, non-seam edges.  Endpoints match oppositely:
    # face f edge j = (j, j+1); neighbour g edge k = (k+1, k).
    for fi, tri in enumerate(faces):
        for j in range(3):
            a, b = int(tri[j]), int(tri[(j + 1) % 3])
            key = (min(a, b), max(a, b))
            inc = edge_faces[key]
            # boundary edge: nothing to weld; seam interior edge: keep split
            if len(inc) != 2 or key in seam:
                continue
            other, odir = inc[1] if inc[0][0] == fi else inc[0]
            tri2 = faces[other]
            k = next(
                k for k in range(3) if (int(tri2[k]), int(tri2[(k + 1) % 3])) == odir
            )
            union(3 * fi + j, 3 * other + (k + 1) % 3)  # a  <-> end
            union(3 * fi + (j + 1) % 3, 3 * other + k)  # b  <-> start

    # compact component ids
    roots = [find(c) for c in range(n_corners)]
    root_to_uv = {}
    for r in roots:
        if r not in root_to_uv:
            root_to_uv[r] = len(root_to_uv)
    corner_uv = np.array([root_to_uv[r] for r in roots], dtype=np.int64).reshape(
        n_faces, 3
    )
    n_uv = len(root_to_uv)

    orig = faces.reshape(-1)
    orig_of_uv = np.zeros(n_uv, dtype=np.int64)
    corners_of_uv = [[] for _ in range(n_uv)]
    for c in range(n_corners):
        u = root_to_uv[roots[c]]
        orig_of_uv[u] = orig[c]
        corners_of_uv[u].append((c // 3, c % 3))
    uv_vertices = verts[orig_of_uv]

    # adjacency + boundary half-edges on the split mesh
    face_adj = [[-1, -1, -1] for _ in range(n_faces)]
    halfedge_faces = {}
    boundary_edges = []
    for fi, tri in enumerate(corner_uv):
        for j in range(3):
            a, b = int(tri[j]), int(tri[(j + 1) % 3])
            rev = (b, a)
            if rev in halfedge_faces:
                g, k = halfedge_faces.pop(rev)
                face_adj[fi][j] = g
                face_adj[g][k] = fi
            else:
                halfedge_faces[(a, b)] = (fi, j)
    for a, b in halfedge_faces:
        boundary_edges.append((a, b))

    return SplitMesh(
        vertices=verts,
        faces=faces,
        seam_edges=tuple(sorted(seam)),
        corner_uv=corner_uv,
        orig_of_uv=orig_of_uv,
        corners_of_uv=corners_of_uv,
        uv_vertices=uv_vertices,
        face_adj=face_adj,
        boundary_edges=boundary_edges,
    )
