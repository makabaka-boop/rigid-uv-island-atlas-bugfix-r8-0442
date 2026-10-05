"""Area-weighted Least Squares Conformal Maps, assembled from scratch.

For each triangle we build a local orthonormal frame (x, y) in its plane and
the gradient coefficients (a,b,c)/d of a piecewise-linear scalar.  The
discrete conformality residual of the triangle is

    C = (U_x - V_y) + i (U_y + V_x) .

Rows are scaled by sqrt(area), so the least-squares objective is exactly

    sum_f area_f * |C_f|^2 ,

the area-weighted LSCM energy.  Two boundary UV vertices are pinned to
(0,0) and (1,0); the remaining 2U-4 columns are solved with SVD least
squares.  No mesh-generation / UV library is involved.
"""

import numpy as np


class SolveError(ValueError):
    pass


def _local_grad_coeffs(p0, p1, p2):
    """Return (gradients, area) for one triangle.

    gradients[i] is the in-plane 2D gradient of hat function phi_i in a
    local orthonormal frame, via the general formula
        grad phi_i = n x (p_{i+2} - p_{i+1}) / (2A),
    which is robust to any edge direction (a frame built from e1 would
    degenerate when e1 is perpendicular to the chosen in-plane axis).
    """
    e1 = p1 - p0
    e2 = p2 - p0
    cross = np.cross(e1, e2)
    area2 = np.linalg.norm(cross)
    n = cross / area2
    x = e1 / np.linalg.norm(e1)
    y = np.cross(n, x)

    def frame(v):
        return np.array([np.dot(v, x), np.dot(v, y)])

    a2 = area2  # = 2 * area
    grads = np.array(
        [
            frame(np.cross(n, p2 - p1)) / a2,
            frame(np.cross(n, p0 - p2)) / a2,
            frame(np.cross(n, p1 - p0)) / a2,
        ]
    )
    return grads, 0.5 * area2


def solve_lscm(sm, anchor0, anchor1):
    a0, a1 = int(anchor0), int(anchor1)
    if not (0 <= a0 < sm.n_uv and 0 <= a1 < sm.n_uv):
        raise SolveError("anchor id out of range")
    if a0 == a1:
        raise SolveError("the two anchors must be different UV vertices")
    boundary = set(sm.boundary_uvs())
    if a0 not in boundary or a1 not in boundary:
        raise SolveError("both anchors must lie on the cut boundary")

    F, U = sm.n_faces, sm.n_uv
    A = np.zeros((2 * F, 2 * U))
    areas = np.zeros(F)

    for fi, tri in enumerate(sm.corner_uv):
        p = [sm.uv_vertices[int(tri[j])] for j in range(3)]
        g, area = _local_grad_coeffs(p[0], p[1], p[2])
        areas[fi] = area
        w = np.sqrt(area)
        for j in range(3):
            u = int(tri[j])
            gx, gy = g[j]
            # real row:  U_x - V_y
            A[2 * fi, u] += w * gx
            A[2 * fi, U + u] += w * (-gy)
            # imag row:  U_y + V_x
            A[2 * fi + 1, u] += w * gy
            A[2 * fi + 1, U + u] += w * gx

    pins = {a0: (0.0, 0.0), a1: (1.0, 0.0)}
    pinned_cols = sorted([u for u in pins] + [U + u for u in pins])
    free_cols = [c for c in range(2 * U) if c not in set(pinned_cols)]

    B = A[:, pinned_cols]
    z_p = np.array([pins[u][0] for u in pins] + [pins[u][1] for u in pins])
    b = -B @ z_p
    Af = A[:, free_cols]

    z, residuals, rank, sv = np.linalg.lstsq(Af, b, rcond=None)

    tol = max(Af.shape) * np.finfo(float).eps * (sv[0] if sv.size else 0.0)
    rank = int(np.sum(sv > tol))
    expected = min(Af.shape[0], Af.shape[1])
    rank_deficient = rank < expected

    z_full = np.zeros(2 * U)
    z_full[pinned_cols] = z_p
    z_full[free_cols] = z
    uv = np.column_stack([z_full[:U], z_full[U:]])

    # per-face conformality residuals C = r + i s
    face_residual = np.zeros(F)
    proj = A @ z_full
    for fi in range(F):
        face_residual[fi] = np.hypot(proj[2 * fi], proj[2 * fi + 1]) / max(
            np.sqrt(areas[fi]), 1e-30
        )  # |C_f|

    energy = float(np.sum(proj**2))
    rms = float(np.sqrt(np.mean(proj**2)))
    max_c = float(np.max(face_residual))

    # flipped / degenerate faces in UV space
    uvf = uv[sm.corner_uv]
    signed_area2 = (uvf[:, 1, 0] - uvf[:, 0, 0]) * (uvf[:, 2, 1] - uvf[:, 0, 1]) - (
        uvf[:, 1, 1] - uvf[:, 0, 1]
    ) * (uvf[:, 2, 0] - uvf[:, 0, 0])
    uv_areas = 0.5 * signed_area2
    scale = np.median(np.abs(uv_areas)) + 1e-30
    flipped = [int(i) for i in range(F) if uv_areas[i] < -1e-9 * scale]
    degenerate = [int(i) for i in range(F) if abs(uv_areas[i]) < 1e-12 * scale]

    angle_errors = _angle_errors(sm, uv)
    face_angle_error = np.max(angle_errors, axis=1)  # degrees per face

    return {
        "uv": uv,
        "energy": energy,
        "rms_row_residual": rms,
        "max_conformal_residual": max_c,
        "face_conformal_residual": face_residual.tolist(),
        "rank": rank,
        "expected_rank": int(expected),
        "rank_deficient": bool(rank_deficient),
        "smallest_singular_values": sv[-4:].tolist(),
        "flipped_faces": flipped,
        "degenerate_uv_faces": degenerate,
        "angle_errors_deg": angle_errors.tolist(),
        "face_angle_error_deg": face_angle_error.tolist(),
        "anchors": [a0, a1],
    }


def _corner_angles(points):
    """Interior angle at every corner of one triangle, radians."""
    ang = np.zeros(3)
    for i in range(3):
        v1 = points[(i + 1) % 3] - points[i]
        v2 = points[(i - 1) % 3] - points[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        c = np.dot(v1, v2) / (n1 * n2)
        ang[i] = np.arccos(np.clip(c, -1.0, 1.0))
    return ang


def _angle_errors(sm, uv):
    errs = np.zeros((sm.n_faces, 3))
    for fi, tri in enumerate(sm.corner_uv):
        p3 = np.array([sm.uv_vertices[int(tri[j])] for j in range(3)])
        p2 = uv[tri]
        a3 = _corner_angles(p3)
        a2 = _corner_angles(p2)
        errs[fi] = np.degrees(np.abs(a2 - a3))
    return errs
