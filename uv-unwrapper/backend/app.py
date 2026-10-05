"""Flask backend: prepare (split + disk check), solve (LSCM) and atlas."""

from flask import Flask, jsonify, request, send_from_directory

from lscm import SolveError, solve_lscm
from mesh import MeshError, split_mesh
from packing import PackingError, pack_islands, validate_request
from samples import (
    cube_closed,
    cube_spanning_tree_seams,
    get_sample,
    plane_rect,
)
from topology import boundary_loops, check_disk, tutte_preview


class _SolveFailed(Exception):
    """An island mesh could not be turned into a usable LSCM solution."""

    def __init__(self, message, status=400, payload=None):
        super().__init__(message)
        self.status = status
        self.payload = payload


def _prepare_payload(data):
    verts = data["vertices"]
    faces = data["faces"]
    seams = data.get("seam_edges", [])
    sm = split_mesh(verts, faces, seams)
    ok, witness = check_disk(sm)
    loops = [list(map(int, l)) for l in boundary_loops(sm)]
    boundary_uvs = sorted({u for l in loops for u in l})
    payload = {
        "valid": ok,
        "witness": witness,
        "n_faces": sm.n_faces,
        "n_orig_vertices": len(sm.vertices),
        "n_uv_vertices": sm.n_uv,
        "corner_uv": sm.corner_uv.tolist(),
        "orig_of_uv": sm.orig_of_uv.tolist(),
        "uv_vertices_3d": sm.uv_vertices.tolist(),
        "seam_edges": [list(e) for e in sm.seam_edges],
        "boundary_loops": loops,
        "boundary_uvs": boundary_uvs,
        "faces": sm.faces.tolist(),
        "vertices": sm.vertices.tolist(),
    }
    if ok:
        payload["preview_uv"] = tutte_preview(sm).tolist()
    return sm, payload


def _solve_payload(data):
    """The exact split -> disk check -> LSCM path, for both /api/solve and
    every island of an atlas.

    Returns the JSON-serializable success payload.  Raises _SolveFailed on
    malformed input, illegal cuts or solve failures so an atlas can reject
    the whole batch instead of exporting a partial atlas.
    """
    try:
        sm, prep = _prepare_payload(data)
    except (MeshError, KeyError, TypeError, ValueError) as exc:
        raise _SolveFailed(str(exc), 400)
    if not prep["valid"]:
        raise _SolveFailed(
            "not a topological disk, no solve was attempted",
            422,
            {"witness": prep["witness"]},
        )
    try:
        res = solve_lscm(sm, data["anchor0"], data["anchor1"])
    except (SolveError, KeyError, TypeError, ValueError) as exc:
        raise _SolveFailed(str(exc), 400)

    uv = res["uv"]
    rows = []
    for fi, tri in enumerate(sm.corner_uv):
        for j in range(3):
            u = int(tri[j])
            rows.append(
                {
                    "face": fi,
                    "corner": j,
                    "orig_vertex": int(sm.faces[fi, j]),
                    "uv_vertex": u,
                    "u": float(uv[u, 0]),
                    "v": float(uv[u, 1]),
                    "angle_error_deg": res["angle_errors_deg"][fi][j],
                }
            )

    return {
        "ok": True,
        "prepare": {
            k: prep[k]
            for k in (
                "boundary_uvs",
                "boundary_loops",
                "corner_uv",
                "orig_of_uv",
                "uv_vertices_3d",
                "n_uv_vertices",
            )
        },
        "uv": uv.tolist(),
        "anchors": res["anchors"],
        "energy": res["energy"],
        "rms_row_residual": res["rms_row_residual"],
        "max_conformal_residual": res["max_conformal_residual"],
        "face_conformal_residual": res["face_conformal_residual"],
        "rank": res["rank"],
        "expected_rank": res["expected_rank"],
        "rank_deficient": res["rank_deficient"],
        "smallest_singular_values": res["smallest_singular_values"],
        "flipped_faces": res["flipped_faces"],
        "degenerate_uv_faces": res["degenerate_uv_faces"],
        "face_angle_error_deg": res["face_angle_error_deg"],
        "angle_errors_deg": res["angle_errors_deg"],
        "corners": rows,
    }


def _atlas_sample_request():
    """A ready-to-run request: differently sized islands whose request order
    deliberately differs from id order (to expose cross-labelling bugs)."""

    def plane(ni, nj, anchor0, anchor1):
        v, f = plane_rect(ni, nj)
        return {
            "vertices": v.tolist(),
            "faces": f.tolist(),
            "seam_edges": [],
            "anchor0": anchor0,
            "anchor1": anchor1,
        }

    v, f = cube_closed()
    cube = {
        "vertices": v.tolist(),
        "faces": f.tolist(),
        "seam_edges": cube_spanning_tree_seams(),
        "anchor0": 0,
        "anchor1": 1,
    }
    return {
        "padding": 0.04,
        "islands": [
            # wide rectangular island (may receive a proper 90 degree turn),
            # requested first though it sorts last
            {"id": "wide-3x5", "mesh": plane(3, 5, 0, 1)},
            {"id": "small-3x3", "mesh": plane(3, 3, 0, 1)},
            {"id": "cube-seamed", "mesh": cube},
        ],
    }


def create_app():
    app = Flask(__name__, static_folder=None)

    @app.get("/api/samples/<name>")
    def sample(name):
        try:
            verts, faces, seams = get_sample(name)
        except KeyError:
            return jsonify({"error": "unknown sample %r" % name}), 404
        return jsonify({"vertices": verts, "faces": faces, "seam_edges": seams})

    @app.post("/api/prepare")
    def prepare():
        try:
            _, payload = _prepare_payload(request.get_json(force=True))
        except (MeshError, KeyError, TypeError, ValueError) as exc:
            return (
                jsonify(
                    {
                        "valid": False,
                        "error": str(exc),
                        "witness": {"kind": "invalid_input", "message": str(exc)},
                    }
                ),
                400,
            )
        return jsonify(payload)

    @app.post("/api/solve")
    def solve():
        try:
            return jsonify(_solve_payload(request.get_json(force=True)))
        except _SolveFailed as exc:
            payload = {"ok": False, "error": str(exc)}
            if exc.payload:
                payload.update(exc.payload)
            return jsonify(payload), exc.status

    @app.get("/api/atlas/sample")
    def atlas_sample():
        return jsonify(_atlas_sample_request())

    @app.post("/api/atlas")
    def atlas():
        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify(ok=False, error="request body is not valid JSON"), 400
        try:
            # Validate the whole request before solving anything: either the
            # complete atlas is produced or no island is exported at all.
            islands, padding = validate_request(data)
            solved = []
            for item in islands:
                try:
                    result = _solve_payload(item["mesh"])
                except _SolveFailed as exc:
                    raise PackingError(
                        "island %r is unusable: %s" % (item["id"], exc)
                    )
                # One bad embedding rejects the entire batch.
                solved.append({"id": item["id"], "result": result})
            return jsonify(ok=True, atlas=pack_islands(solved, padding))
        except PackingError as exc:
            return jsonify(ok=False, error=str(exc)), 400
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify(ok=False, error=str(exc)), 400

    @app.get("/atlas")
    def atlas_page():
        return send_from_directory("static", "atlas.html")

    @app.get("/atlas-view.js")
    def atlas_script():
        return send_from_directory("static", "atlas-view.js")

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=True)
