"""Deterministic multi-island atlas packing.

Packing only rewrites the u,v coordinates of solved islands.  Every island
keeps its own face / corner / original-vertex / split UV-vertex identity;
corners are *never* welded across islands or deduplicated by coordinate.

Invariants enforced here
------------------------
* ONE common positive scale for every island (no per-island rescaling, so
  relative sizes are preserved).
* The only per-island motion beyond translation is an
  orientation-preserving quarter turn ``(u, v) -> (-v, u)`` (det = +1).
  Swapping axes (det = -1) is a reflection and is never used, so the
  signed winding of every face is unchanged.
* ``padding`` is both the outer margin and the guaranteed gap between
  island cells.  It must be finite, non-negative and < 1/(n+1).
* Cells are assigned by the stable string id, so the layout is
  reproducible; the returned islands still follow the *request* order and
  every coordinate batch is labelled with the id it was solved from.

An optimal packing is explicitly not a goal (see README): islands go into
equal-height bands and are centred in their cell.
"""

import math

import numpy as np


class PackingError(ValueError):
    """Raised when the atlas cannot be built from the solved islands."""


def _quarter_turn(uv, k):
    """Rotate (u, v) by k quarter-turns counter-clockwise.  k in {0, 1}.

    Both transforms have determinant +1 (R0 = I, R1 maps (u,v) to (-v,u));
    they preserve shapes and orientation and never reflect.
    """
    if k == 0:
        return uv
    x, y = uv[:, 0], uv[:, 1]
    return np.column_stack((-y, x))


def pack_islands(islands, padding):
    if not isinstance(islands, (list, tuple)) or not islands:
        raise PackingError("at least one solved island is required")
    n = len(islands)

    p = float(padding)
    padding_limit = 1.0 / (n + 1)
    if not math.isfinite(p) or p < 0.0 or p >= padding_limit:
        raise PackingError(
            "padding must be finite, non-negative and smaller than "
            "1/(n+1) = %g (got %r)" % (padding_limit, padding)
        )

    entries = []
    seen_ids = set()
    for slot, item in enumerate(islands):
        if not isinstance(item, dict):
            raise PackingError("island #%d is not an object" % slot)
        iid = item.get("id")
        if not isinstance(iid, str) or not iid:
            raise PackingError(
                "island #%d needs a non-empty stable string id" % slot
            )
        if iid in seen_ids:
            raise PackingError("duplicate island id %r" % iid)
        seen_ids.add(iid)

        uv = np.asarray(item.get("uv"), dtype=float)
        if uv.ndim != 2 or uv.shape[1] != 2 or uv.shape[0] < 1:
            raise PackingError("island %r: uv must be a (U, 2) array" % iid)
        if not np.all(np.isfinite(uv)):
            raise PackingError("island %r: uv contains non-finite values" % iid)

        corners = list(item.get("corners", ()))
        for row in corners:
            if not isinstance(row, dict):
                raise PackingError("island %r: corner row is not an object" % iid)
            u = row.get("uv_vertex")
            if not isinstance(u, int) or isinstance(u, bool) or not 0 <= u < uv.shape[0]:
                raise PackingError(
                    "island %r: corner row %r has no valid uv_vertex" % (iid, row)
                )

        width = float(np.ptp(uv[:, 0]))
        height = float(np.ptp(uv[:, 1]))
        if width <= 0.0 or height <= 0.0:
            raise PackingError(
                "island %r has a zero-width/height UV bounding box" % iid
            )
        entries.append(
            {
                "id": iid,
                "uv": uv,
                "corners": corners,
                "width": width,
                "height": height,
            }
        )

    # Equal-height bands leave p as the top/bottom/left/right margin and p
    # between neighbouring bands.  Every island has to fit one such cell.
    cell_h = (1.0 - (n + 1) * p) / n
    cell_w = 1.0 - 2.0 * p

    # Choose an orientation per island.  A +90 degree turn is allowed only
    # when it strictly improves how that island fills its cell; an exact
    # tie keeps 0 degrees so the result is fully reproducible.
    for e in entries:
        fit0 = min(cell_w / e["width"], cell_h / e["height"])
        fit1 = min(cell_w / e["height"], cell_h / e["width"])
        e["turn"] = 1 if fit1 > fit0 * (1.0 + 1e-12) else 0
        e["fit"] = fit1 if e["turn"] else fit0

    # The single, shared scale: every island is reduced by the same factor.
    scale = min(e["fit"] for e in entries)
    if not math.isfinite(scale) or scale <= 0.0:
        raise PackingError("common scale must be a positive, finite number")

    # Cell assignment is driven solely by the stable ids ...
    layout_order = sorted(range(n), key=lambda i: entries[i]["id"])
    outputs = [None] * n
    for band, i in enumerate(layout_order):
        e = entries[i]
        turned = _quarter_turn(e["uv"], e["turn"])
        lo = turned.min(axis=0)
        span = turned.max(axis=0) - lo

        # Centre inside the (cell_w x cell_h) band cell.
        left = p + (cell_w - span[0] * scale) / 2.0
        bottom = p + band * (cell_h + p) + (cell_h - span[1] * scale) / 2.0
        offset = np.array([left - lo[0] * scale, bottom - lo[1] * scale])
        placed = turned * scale + offset

        # Identity fields (face/corner/orig_vertex/uv_vertex/angle error)
        # are copied verbatim; only u,v move, and each row is tagged with
        # the id of the island it actually belongs to.
        corners = []
        for row in e["corners"]:
            out = dict(row)
            u = int(row["uv_vertex"])
            out["u"] = float(placed[u, 0])
            out["v"] = float(placed[u, 1])
            out["island_id"] = e["id"]
            corners.append(out)

        outputs[i] = {
            "id": e["id"],
            "uv": placed.tolist(),
            "corners": corners,
            "scale": float(scale),
            "rotation_deg": 90 * e["turn"],
            "band": band,
            "bounds": {
                "u_min": float(placed[:, 0].min()),
                "u_max": float(placed[:, 0].max()),
                "v_min": float(placed[:, 1].min()),
                "v_max": float(placed[:, 1].max()),
            },
        }

    # ... while outputs stay in request order, each with its own identity.
    return {"scale": float(scale), "padding": p, "islands": outputs}
