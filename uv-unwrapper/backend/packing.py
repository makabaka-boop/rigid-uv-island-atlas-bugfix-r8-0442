"""Deterministic multi-island atlas packing.

Invariants enforced here
------------------------
* ONE positive scale common to every island.  Islands of different original
  sizes keep their relative proportions; the scale is never chosen per
  island.
* Each island is only translated and, optionally, rotated by an
  orientation-preserving quarter turn (det = +1).  Reflections are never
  used, so every face keeps its winding/orientation and islands keep their
  shape.
* The gap to every page edge and the gap between island bands are all the
  same normalized ``padding``.
* An island's stable id always travels with its *own* solve result: its own
  faces, corners, original vertices and seam-split UV vertices.  Identities
  are never merged by coordinate.
* Band assignment is computed in id-sorted order, so the layout is
  reproducible regardless of the request order; the returned island list
  follows the request order.
"""

import numpy as np


class PackingError(ValueError):
    """Raised when the atlas request or an island embedding is unusable."""


# Orientation-preserving quarter turns only (determinant +1).
# k=0: identity, k=1: 90 degrees counter-clockwise (u, v) -> (-v, u).
_ROTATIONS = (
    np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
    np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float),
)
ROTATION_DEG = (0, 90)


def _is_real_number(x):
    # bool is a subclass of int but must not be accepted as a padding value
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_request(data):
    """Validate the /api/atlas request before any island is solved.

    Returns (islands, padding_float).  Raises PackingError on anything
    illegal so that the whole atlas is rejected up front (all-or-nothing).
    """
    if not isinstance(data, dict):
        raise PackingError("request body must be an object with 'islands'")
    islands = data.get("islands")
    if not isinstance(islands, list):
        raise PackingError("islands array required")
    n = len(islands)
    if not 1 <= n <= 8:
        raise PackingError("one to eight islands required (got %d)" % n)

    padding = data.get("padding")
    if not _is_real_number(padding):
        raise PackingError("padding must be a finite number")
    padding = float(padding)
    if not np.isfinite(padding) or padding < 0:
        raise PackingError("padding must be a finite, non-negative number")
    upper = 1.0 / (n + 1)
    if padding >= upper:
        raise PackingError(
            "padding %.6g must be smaller than 1/(islands+1) = %.6g"
            % (padding, upper)
        )

    seen = set()
    for k, item in enumerate(islands):
        if not isinstance(item, dict):
            raise PackingError("island %d must be an object" % k)
        iid = item.get("id")
        if not isinstance(iid, str) or not iid:
            raise PackingError("island %d: a non-empty stable string id is required" % k)
        if iid in seen:
            raise PackingError("duplicate island id %r" % iid)
        seen.add(iid)
        if not isinstance(item.get("mesh"), dict):
            raise PackingError("island %r: the original solve request 'mesh' is required" % iid)
    return islands, padding


def _prepare_island(item, band_h, avail_w):
    """Validate one solve result and choose its (proper) rotation.

    Returns placement data; raises PackingError if the embedding is
    unusable (non-finite, degenerate, flipped or mirrored-looking).
    """
    iid = item["id"]
    result = item["result"]
    if not isinstance(result, dict) or "uv" not in result or "corners" not in result:
        raise PackingError("island %r is missing its solve result" % iid)
    # A flipped/degenerate LSCM result is not a usable texture island.
    if result.get("flipped_faces") or result.get("degenerate_uv_faces"):
        raise PackingError(
            "island %r is unusable: its solve reports flipped or "
            "degenerate UV faces" % iid
        )

    uv = np.asarray(result["uv"], dtype=float)
    if uv.ndim != 2 or uv.shape[1] != 2 or not np.all(np.isfinite(uv)):
        raise PackingError("island %r produced non-finite UV coordinates" % iid)
    corners = result["corners"]
    if not isinstance(corners, list) or len(corners) % 3 != 0 or not corners:
        raise PackingError("island %r has malformed corner rows" % iid)

    best = None
    for k, rot in enumerate(_ROTATIONS):
        rotated = uv @ rot.T
        mn = rotated.min(axis=0)
        ext = rotated.max(axis=0) - mn
        if not np.all(np.isfinite(ext)) or np.any(ext <= 0.0):
            raise PackingError("island %r has a zero-width or zero-height embedding" % iid)
        # Largest scale at which this orientation still fits its band.
        fit = min(avail_w / ext[0], band_h / ext[1])
        # Strict '>' keeps k=0 (no rotation) on an exact tie.
        if best is None or fit > best["fit"]:
            best = {"k": k, "fit": float(fit), "ext": ext, "min": mn, "rotated": rotated}

    # The solve itself must not already contain mirrored faces.
    if _any_face_flipped(corners, uv):
        raise PackingError("island %r is unusable: faces are mirrored in UV" % iid)
    return best


def _any_face_flipped(corners, uv):
    """True if any face, read from corner rows, winds clockwise / degenerate."""
    by_face = {}
    for row in corners:
        by_face.setdefault(int(row["face"]), {})[int(row["corner"])] = int(row["uv_vertex"])
    for tri in by_face.values():
        if set(tri) != {0, 1, 2}:
            raise PackingError("corner rows do not carry face/corner identity")
        p = [uv[tri[j]] for j in range(3)]
        signed2 = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (
            p[1][1] - p[0][1]
        ) * (p[2][0] - p[0][0])
        if signed2 <= 0.0:
            return True
    return False


def pack_islands(islands, padding):
    """Place solved islands into the unit square.

    Parameters
    ----------
    islands : list of {"id": str, "result": /api/solve success payload}
        In request order.  Each entry's own id/result pair is kept; the
        returned list also follows this request order.
    padding : float
        Normalized page-edge margin and inter-band gap; 0 <= padding <
        1/(n+1).
    """
    n = len(islands)
    if not 1 <= n <= 8:
        raise PackingError("one to eight islands required (got %d)" % n)
    p = float(padding)
    if not np.isfinite(p) or p < 0 or p >= 1.0 / (n + 1):
        raise PackingError("padding out of range")

    band_h = (1.0 - (n + 1) * p) / n
    avail_w = 1.0 - 2.0 * p

    ordered = sorted(islands, key=lambda it: it["id"])  # reproducible bands
    prepared = {}
    for item in ordered:
        prepared[item["id"]] = _prepare_island(item, band_h, avail_w)

    # The single scale shared by every island.
    scale = min(b["fit"] for b in prepared.values())
    if not np.isfinite(scale) or scale <= 0.0:
        raise PackingError("the common atlas scale must be positive")

    band_of = {item["id"]: i for i, item in enumerate(ordered)}
    outputs = []
    for item in islands:  # output order follows the REQUEST, not sorted order
        iid = item["id"]
        band = band_of[iid]
        b = prepared[iid]
        k, ext, mn, rotated = b["k"], b["ext"], b["min"], b["rotated"]
        # Bottom-left of this band, then center the island inside it.
        x0 = p + (avail_w - scale * ext[0]) / 2.0
        y0 = p + band * (band_h + p) + (band_h - scale * ext[1]) / 2.0
        placed = scale * rotated + np.array(
            [x0 - scale * mn[0], y0 - scale * mn[1]]
        )

        # Corners are rebuilt from THIS island's rows and THIS island's uv.
        corners = []
        for row in item["result"]["corners"]:
            out = dict(row)
            u = int(row["uv_vertex"])
            out["u"] = float(placed[u, 0])
            out["v"] = float(placed[u, 1])
            out["island_id"] = iid
            corners.append(out)

        outputs.append(
            {
                "id": iid,
                "uv": placed.tolist(),
                "corners": corners,
                "rotation_deg": ROTATION_DEG[k],
                "scale": float(scale),
                "translation": [float(x0), float(y0)],
                "band": band,
                "source": item["result"],
            }
        )

    return {
        "scale": float(scale),
        "padding": p,
        "band_height": float(band_h),
        "islands": outputs,
    }
