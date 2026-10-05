"""Topology verification on the split mesh and a preview embedding.

The split complex is an ordinary simplicial-ish complex once seam sides are
viewed as boundary half-edges: an un-cut interior edge keeps one twin
half-edge on each side (``face_adj``), while both sides of a seam and every
outer edge have no twin.  Pivoting around a split vertex therefore stays
inside one face sector and naturally exits along the next seam side -- the
boundary walk climbs one side of a slit and comes back down the other with
no special casing.

A triangle mesh is a topological disk iff
  * its faces are connected (through un-cut edges),
  * boundary half-edges trace exactly one closed loop,
  * chi = U - E + F = 1, with E = (3F + B)/2 derived from half-edge
    incidence (B = number of boundary half-edges).
Everything else is rejected with a witness, so LSCM is never run on an
illegal cut.
"""

import numpy as np


def _boundary_successor(sm, face, slot):
    """Rotate about the end vertex of boundary half-edge (face, slot) until
    the next boundary half-edge, crossing ordinary (un-cut) interior edges.
    """
    while True:
        slot = (slot + 1) % 3
        nf = sm.face_adj[face][slot]
        if nf == -1:
            return face, slot
        nslot = sm.face_adj[nf].index(face)
        face, slot = nf, nslot


def boundary_loops(sm):
    """Trace boundary half-edges into oriented loops of UV vertex ids."""
    remaining = {
        (fi, j)
        for fi in range(sm.n_faces)
        for j in range(3)
        if sm.face_adj[fi][j] == -1
    }
    loops = []
    while remaining:
        f0, s0 = next(iter(remaining))
        loop = [int(sm.corner_uv[f0, s0]), int(sm.corner_uv[f0, (s0 + 1) % 3])]
        remaining.discard((f0, s0))
        f, s = _boundary_successor(sm, f0, s0)
        guard = 0
        while (f, s) != (f0, s0):
            if (f, s) not in remaining:
                # open half-edge chain: malformed/non-manifold boundary
                loop.append(None)
                break
            remaining.discard((f, s))
            loop.append(int(sm.corner_uv[f, (s + 1) % 3]))
            f, s = _boundary_successor(sm, f, s)
            guard += 1
            if guard > 6 * sm.n_faces + 6:
                raise RuntimeError("boundary walk failed to close")
        else:
            loop.pop()  # closing copy of the first vertex
        loops.append(loop)
    return loops


def _face_components(sm):
    """Connected components of faces linked through un-cut interior edges."""
    adj = [[] for _ in range(sm.n_faces)]
    for fi in range(sm.n_faces):
        for j in range(3):
            g = sm.face_adj[fi][j]
            if g != -1:
                adj[fi].append(g)
    comp = -np.ones(sm.n_faces, dtype=np.int64)
    example_uvs = []
    cid = 0
    for s in range(sm.n_faces):
        if comp[s] != -1:
            continue
        example_uvs.append(int(sm.corner_uv[s, 0]))
        stack = [s]
        comp[s] = cid
        while stack:
            f = stack.pop()
            for g in adj[f]:
                if comp[g] == -1:
                    comp[g] = cid
                    stack.append(g)
        cid += 1
    sizes = [int(np.sum(comp == c)) for c in range(cid)]
    return cid, sizes, example_uvs


def check_disk(sm):
    """Return (ok, witness). witness is {} when ok."""
    n_comp, sizes, example_uvs = _face_components(sm)
    if n_comp != 1:
        return False, {
            "kind": "disconnected",
            "components": n_comp,
            "component_sizes": sizes,
            "example_vertices": example_uvs[:8],
        }

    loops = boundary_loops(sm)
    if any(None in loop for loop in loops):
        return False, {
            "kind": "open_boundary_chain",
            "message": "boundary half-edges do not close into loops; the "
            "input surface is not a closed 2-manifold",
        }
    if len(loops) == 0:
        return False, {
            "kind": "closed_surface",
            "message": "the cut has no boundary at all; the surface is "
            "still closed and cannot be mapped to a disk",
        }
    if len(loops) > 1:
        return False, {
            "kind": "multiple_boundary_loops",
            "loops": len(loops),
            "boundary_loops": [list(map(int, loop)) for loop in loops],
        }

    b_count = sum(
        1 for fi in range(sm.n_faces) for j in range(3) if sm.face_adj[fi][j] == -1
    )
    e_count = (3 * sm.n_faces + b_count) // 2
    chi = sm.n_uv - e_count + sm.n_faces
    if chi != 1:
        return False, {
            "kind": "non_disk_characteristic",
            "euler_characteristic": int(chi),
            "expected": 1,
            "message": "one boundary loop but chi = U-E+F = %d; the cut "
            "still has handles or uncapped topology" % chi,
        }
    return True, {}


def tutte_preview(sm):
    """Boundary-on-circle / interior-average embedding, used ONLY to draw
    the anchor picker.  It is never used by the actual LSCM solve.
    """
    loop = boundary_loops(sm)[0]
    u = np.zeros((sm.n_uv, 2))
    nb = [set() for _ in range(sm.n_uv)]
    for tri in sm.corner_uv:
        for j in range(3):
            a, b = int(tri[j]), int(tri[(j + 1) % 3])
            nb[a].add(b)
            nb[b].add(a)

    boundary = set(loop)
    k = len(loop)
    for i, vid in enumerate(loop):
        t = 2 * np.pi * i / k
        u[vid] = (np.cos(t), np.sin(t))

    interior = [v for v in range(sm.n_uv) if v not in boundary]
    if interior:
        L = np.zeros((len(interior), len(interior)))
        rhs = np.zeros((len(interior), 2))
        idx = {v: i for i, v in enumerate(interior)}
        for v in interior:
            i = idx[v]
            L[i, i] = 1.0
            w = 1.0 / len(nb[v])
            for z in nb[v]:
                if z in boundary:
                    rhs[i] += w * u[z]
                else:
                    L[i, idx[z]] = -w
        u[interior] = np.linalg.solve(L, rhs)
    return u
