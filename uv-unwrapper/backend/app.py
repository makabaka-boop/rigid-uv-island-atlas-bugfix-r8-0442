"""Flask backend: prepare (split + disk check) and solve (LSCM)."""

import math

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from lscm import SolveError, solve_lscm
from mesh import MeshError, split_mesh
from samples import get_sample, plane_grid
from topology import boundary_loops, check_disk, tutte_preview


def create_app():
    app = Flask(__name__, static_folder=None)

    @app.get("/api/samples/<name>")
    def sample(name):
        try:
            verts, faces, seams = get_sample(name)
        except KeyError:
            return jsonify({"error": "unknown sample %r" % name}), 404
        return jsonify({"vertices": verts, "faces": faces, "seam_edges": seams})

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

    def _corner_rows(sm, res):
        """One row per face corner: face/corner/original vertex/split UV
        vertex identity plus coordinates and per-corner angle error."""
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
                        "angle_error_deg": float(res["angle_errors_deg"][fi][j]),
                    }
                )
        return rows

    def _solved_mesh(data, quality_gate=False):
        """Run the exact split -> disk check -> LSCM path used by /api/solve.

        Returns (sm, result_dict, corner_rows).  When ``quality_gate`` is
        set (atlas flow), a rank-deficient, flipped or degenerate solve is
        treated as an unusable island so no partial atlas is exported.
        """
        sm, prep = _prepare_payload(data)
        if not prep["valid"]:
            raise MeshError(
                "not a topological disk (%s), no solve was attempted"
                % prep["witness"].get("kind", "invalid")
            )
        res = solve_lscm(sm, data["anchor0"], data["anchor1"])
        uv = np.asarray(res["uv"], dtype=float)
        if not np.all(np.isfinite(uv)):
            raise SolveError("solver returned non-finite UV coordinates")
        if quality_gate:
            if res["rank_deficient"]:
                raise SolveError("LSCM system is rank-deficient; solution is not unique")
            if res["flipped_faces"]:
                raise SolveError(
                    "solver produced %d flipped face(s)" % len(res["flipped_faces"])
                )
            if res["degenerate_uv_faces"]:
                raise SolveError(
                    "solver produced %d degenerate face(s)"
                    % len(res["degenerate_uv_faces"])
                )
        return sm, res, _corner_rows(sm, res)

    @app.post("/api/prepare")
    def prepare():
        try:
            _, payload = _prepare_payload(request.get_json(force=True))
        except MeshError as exc:
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
        data = request.get_json(force=True)
        try:
            sm, res, rows = _solved_mesh(data)
        except MeshError as exc:
            # Re-derive the prepare payload for an honest 422 witness.
            try:
                _, prep = _prepare_payload(data)
            except MeshError as bad_input:
                return jsonify({"ok": False, "error": str(bad_input)}), 400
            if not prep["valid"]:
                return (
                    jsonify(
                        {
                            "ok": False,
                            "message": "not a topological disk, "
                            "no solve was attempted",
                            **prep,
                        }
                    ),
                    422,
                )
            return jsonify({"ok": False, "error": str(exc)}), 400
        except SolveError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        return jsonify(
            {
                "ok": True,
                "prepare": {
                    "boundary_uvs": sm.boundary_uvs(),
                    "boundary_loops": [
                        list(map(int, l)) for l in boundary_loops(sm)
                    ],
                    "corner_uv": sm.corner_uv.tolist(),
                    "orig_of_uv": sm.orig_of_uv.tolist(),
                    "uv_vertices_3d": sm.uv_vertices.tolist(),
                    "n_uv_vertices": sm.n_uv,
                },
                "uv": res["uv"].tolist(),
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
        )

    def _parse_atlas_request(data):
        if not isinstance(data, dict) or not isinstance(data.get("islands"), list):
            raise ValueError("request must be an object with an islands array")
        islands = data["islands"]
        if not 1 <= len(islands) <= 8:
            raise ValueError("one to eight islands required")
        ids = set()
        for i, item in enumerate(islands):
            if not isinstance(item, dict):
                raise ValueError("island #%d is not an object" % i)
            iid = item.get("id")
            if not isinstance(iid, str) or not iid:
                raise ValueError("island #%d needs a non-empty string id" % i)
            if iid in ids:
                raise ValueError("duplicate island id %r" % iid)
            ids.add(iid)
            mesh = item.get("mesh")
            if not isinstance(mesh, dict):
                raise ValueError("island %r: mesh must be an object" % iid)
            for key in ("vertices", "faces", "anchor0", "anchor1"):
                if key not in mesh:
                    raise ValueError("island %r: mesh is missing %r" % (iid, key))
        padding = data.get("padding")
        if isinstance(padding, bool) or not isinstance(padding, (int, float)):
            raise ValueError("padding must be a number")
        p = float(padding)
        if not math.isfinite(p) or not 0 <= p < 1.0 / (len(islands) + 1):
            raise ValueError(
                "padding must be finite and satisfy 0 <= padding < 1/(n+1) = %g"
                % (1.0 / (len(islands) + 1))
            )
        return islands, p

    @app.post("/api/atlas")
    def atlas():
        from packing import PackingError, pack_islands

        data = request.get_json(force=True, silent=False)
        if data is None:
            return jsonify(ok=False, error="request body must be JSON"), 400
        try:
            islands, padding = _parse_atlas_request(data)
            # Solve and validate EVERY island before building any output,
            # so an unusable island rejects the whole atlas atomically.
            solved = []
            for item in islands:
                iid = item["id"]
                try:
                    _, res, corners = _solved_mesh(item["mesh"], quality_gate=True)
                except (MeshError, SolveError) as exc:
                    raise ValueError("island %r is unusable: %s" % (iid, exc))
                solved.append(
                    {"id": iid, "uv": res["uv"], "corners": corners}
                )
            atlas_result = pack_islands(solved, padding)
        except (ValueError, TypeError, KeyError, PackingError) as exc:
            return jsonify(ok=False, error=str(exc)), 400
        return jsonify(ok=True, atlas=atlas_result)

    @app.get("/api/atlas/sample")
    def atlas_sample():
        """A ready-to-use request: three planar islands of different
        sizes, intentionally listed out of id order."""

        def mesh_request(n):
            v, f = plane_grid(n)
            sm = split_mesh(v, f, [])
            boundary = sm.boundary_uvs()
            return {
                "vertices": v.tolist(),
                "faces": f.tolist(),
                "seam_edges": [],
                "anchor0": int(boundary[0]),
                "anchor1": int(boundary[-1]),
            }

        return jsonify(
            {
                "padding": 0.04,
                "islands": [
                    {"id": "island-C", "mesh": mesh_request(4)},
                    {"id": "island-A", "mesh": mesh_request(2)},
                    {"id": "island-B", "mesh": mesh_request(3)},
                ],
            }
        )

    @app.get("/atlas")
    def atlas_page():
        return send_from_directory("static", "atlas.html")

    @app.get("/atlas-view.js")
    def atlas_script():
        return send_from_directory("static", "atlas-view.js")

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=True)
