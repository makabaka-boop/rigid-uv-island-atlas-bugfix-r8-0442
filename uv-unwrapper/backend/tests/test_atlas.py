"""Tests for the batch-atlas flow: one common scale, orientation-preserving
motion only, stable id<->identity mapping, uniform padding and the
all-or-nothing rejection rule."""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from mesh import split_mesh
from packing import PackingError, pack_islands
from samples import cube_closed, cube_spanning_tree_seams, plane_grid
from topology import check_disk


def plane_request(n, a0=None, a1=None, seams=None):
    v, f = plane_grid(n)
    sm = split_mesh(v, f, seams or [])
    boundary = sm.boundary_uvs()
    if a0 is None:
        a0 = boundary[0]
    if a1 is None:
        a1 = boundary[-1]
    return {
        "vertices": v.tolist(),
        "faces": f.tolist(),
        "seam_edges": seams or [],
        "anchor0": a0,
        "anchor1": a1,
    }


def rect_grid(nx, ny):
    """Rectangular nx*ny-vertex planar grid, like samples.plane_grid."""
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
    return np.array(verts, float), np.array(faces, dtype=np.int64)


def plane_rect_request(nx, ny):
    v, f = rect_grid(nx, ny)
    sm = split_mesh(v, f, [])
    boundary = sm.boundary_uvs()
    return {
        "vertices": v.tolist(),
        "faces": f.tolist(),
        "seam_edges": [],
        "anchor0": int(boundary[0]),
        "anchor1": int(boundary[-1]),
    }


def solved_island(app, iid, req):
    """Run one mesh through the real solve route and reduce it to what
    pack_islands consumes (its own uv + corner identity rows)."""
    with app.test_client() as client:
        r = client.post("/api/solve", json=req)
    assert r.status_code == 200 and r.json["ok"], r.json
    res = r.json
    return {"id": iid, "uv": np.array(res["uv"], float), "corners": res["corners"]}


def signed_areas(uv, faces_uv):
    uvf = uv[faces_uv]
    return 0.5 * (
        (uvf[:, 1, 0] - uvf[:, 0, 0]) * (uvf[:, 2, 1] - uvf[:, 0, 1])
        - (uvf[:, 1, 1] - uvf[:, 0, 1]) * (uvf[:, 2, 0] - uvf[:, 0, 0])
    )


def faces_from_corners(corners):
    by_face = {}
    for row in corners:
        by_face.setdefault(row["face"], [None, None, None])[row["corner"]] = row
    faces = []
    for fi in sorted(by_face):
        tri = by_face[fi]
        assert all(tri), "face %d is missing corner rows" % fi
        faces.append([tri[0]["uv_vertex"], tri[1]["uv_vertex"], tri[2]["uv_vertex"]])
    return np.array(faces)


class TestPackingGeometry(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.islands = [
            solved_island(self.app, "C", plane_request(4)),
            solved_island(self.app, "A", plane_request(2)),
            solved_island(self.app, "B", plane_request(3)),
        ]
        self.p = 0.04
        self.atlas = pack_islands([dict(x) for x in self.islands], self.p)

    def test_one_common_scale_reported_everywhere(self):
        scales = [isl["scale"] for isl in self.atlas["islands"]]
        self.assertTrue(all(math.isclose(s, self.atlas["scale"]) for s in scales))
        self.assertGreater(self.atlas["scale"], 0.0)

    def test_relative_sizes_preserved(self):
        # Source sizes and packed sizes must share exactly the one scale.
        for isl, src in zip(self.atlas["islands"], self.islands):
            src_uv = src["uv"]
            packed = np.array(isl["uv"])
            self.assertAlmostEqual(
                np.ptp(packed[:, 0]) / np.ptp(src_uv[:, 0]), self.atlas["scale"]
            )
            self.assertAlmostEqual(
                np.ptp(packed[:, 1]) / np.ptp(src_uv[:, 1]), self.atlas["scale"]
            )
            # packed bbox area ratios between two islands must equal the
            # source bbox area ratios (one common positive factor each axis)
        wide = solved_island(self.app, "wide", plane_rect_request(6, 3))
        rect = solved_island(self.app, "rect", plane_rect_request(5, 2))
        rect_atlas = pack_islands([wide, rect], 0.04)
        by = {i["id"]: i for i in rect_atlas["islands"]}

        def bbox_area(uv):
            uv = np.asarray(uv)
            return float(np.ptp(uv[:, 0]) * np.ptp(uv[:, 1]))

        src_ratio = bbox_area(wide["uv"]) / bbox_area(rect["uv"])
        out_ratio = bbox_area(by["wide"]["uv"]) / bbox_area(by["rect"]["uv"])
        self.assertAlmostEqual(out_ratio, src_ratio, places=9)

    def test_rotation_choice_is_deterministic_and_preserves_orientation(self):
        # A tall UV island only helps itself by a +90 degree turn; the
        # decision must be stable and never reflect.  The solved UV span is
        # built directly (the solver itself cannot orient islands).
        def strip(iid, nu, nv):
            uv = [(u, v) for v in range(nv) for u in range(nu)]
            corners = []
            for v in range(nv - 1):
                for u in range(nu - 1):
                    a = v * nu + u
                    face = v * (nu - 1) + u
                    corners += [
                        {"face": face, "corner": 0, "orig_vertex": a,
                         "uv_vertex": a},
                        {"face": face, "corner": 1, "orig_vertex": a + 1,
                         "uv_vertex": a + 1},
                        {"face": face, "corner": 2, "orig_vertex": a + nu,
                         "uv_vertex": a + nu},
                    ]
            return {"id": iid, "uv": np.array(uv, float), "corners": corners}

        tall = strip("tall", 2, 6)  # domain 1 x 5 -> tall UV box
        square = strip("sq", 3, 3)
        for ids in (["tall", "sq"], ["sq", "tall"]):
            bag = {"tall": tall, "sq": square}
            atlas = pack_islands([bag[i] for i in ids], 0.04)
            by = {i["id"]: i for i in atlas["islands"]}
            self.assertEqual(by["tall"]["rotation_deg"], 90)
            self.assertEqual(by["sq"]["rotation_deg"], 0)
            src_uv, packed = tall["uv"], np.array(by["tall"]["uv"])
            faces = faces_from_corners(by["tall"]["corners"])
            self.assertTrue(np.all(signed_areas(packed, faces) > 0))
            # after a quarter turn the long axis runs along u, and spans
            # merely swap while staying at the one common scale
            self.assertGreater(np.ptp(packed[:, 0]), np.ptp(packed[:, 1]))
            self.assertAlmostEqual(
                np.ptp(packed[:, 0]) / atlas["scale"], np.ptp(src_uv[:, 1])
            )
            self.assertAlmostEqual(
                np.ptp(packed[:, 1]) / atlas["scale"], np.ptp(src_uv[:, 0])
            )

    def test_motion_is_positive_similarity_per_island(self):
        for isl, src in zip(self.atlas["islands"], self.islands):
            src_uv = src["uv"]
            packed = np.array(isl["uv"])
            before = signed_areas(src_uv, faces_from_corners(isl["corners"]))
            after = signed_areas(packed, faces_from_corners(isl["corners"]))
            # same orientation: signed areas stay positive everywhere
            self.assertTrue(np.all(before > 0))
            self.assertTrue(np.all(after > 0), isl["id"])
            # lengths scaled uniformly (one global factor), angles unchanged
            edges = [(0, 1), (1, 2), (0, 2)]
            ratios = [
                np.linalg.norm(packed[a] - packed[b])
                / np.linalg.norm(src_uv[a] - src_uv[b])
                for a, b in edges
            ]
            for r in ratios:
                self.assertAlmostEqual(r, self.atlas["scale"], places=9)

    def test_rotation_is_quarter_turn_orientation_preserving(self):
        for isl in self.atlas["islands"]:
            self.assertIn(isl["rotation_deg"], (0, 90))

    def test_uniform_padding_and_disjoint_bands(self):
        n = len(self.atlas["islands"])
        p = self.p
        all_coords = np.concatenate([np.array(i["uv"]) for i in self.atlas["islands"]])
        self.assertGreaterEqual(all_coords.min(), p - 1e-9)
        self.assertLessEqual(all_coords.max(), 1.0 - p + 1e-9)
        # equal-height bands -> exact p gap between neighbours
        bands = sorted(self.atlas["islands"], key=lambda i: i["band"])
        cell_h = (1 - (n + 1) * p) / n
        for i in range(n - 1):
            gap = bands[i + 1]["bounds"]["v_min"] - bands[i]["bounds"]["v_max"]
            self.assertAlmostEqual(gap, p, places=9)
        # first band bottom and last band top sit at the margin
        self.assertAlmostEqual(bands[0]["bounds"]["v_min"], p, places=9)
        self.assertAlmostEqual(bands[-1]["bounds"]["v_max"], 1 - p, places=9)
        self.assertTrue(0.0 < cell_h <= 1.0)

    def test_output_follows_request_order(self):
        self.assertEqual(
            [i["id"] for i in self.atlas["islands"]], ["C", "A", "B"]
        )

    def test_layout_is_stable_but_assignment_follows_identity(self):
        # Shuffling the request order must not move coordinates onto another
        # island's name: every id keeps its own solved uvs / face rows.
        shuffled = [dict(x) for x in self.islands]
        shuffled.reverse()
        other = pack_islands(shuffled, self.p)
        a = {i["id"]: i for i in self.atlas["islands"]}
        b = {i["id"]: i for i in other["islands"]}
        self.assertEqual(set(a), set(b))
        for iid in a:
            self.assertTrue(
                np.allclose(np.array(a[iid]["uv"]), np.array(b[iid]["uv"])),
                iid,
            )
            self.assertEqual(a[iid]["band"], b[iid]["band"])
            self.assertEqual(a[iid]["rotation_deg"], b[iid]["rotation_deg"])
        # other order -> other response order, same identity
        self.assertEqual([i["id"] for i in other["islands"]], ["B", "A", "C"])

    def test_corner_identity_is_preserved_not_merged_by_coordinate(self):
        by_id = {i["id"]: i for i in self.atlas["islands"]}
        for src in self.islands:
            out = by_id[src["id"]]
            self.assertEqual(len(out["corners"]), len(src["corners"]))
            for srow, orow in zip(src["corners"], out["corners"]):
                for key in ("face", "corner", "orig_vertex", "uv_vertex"):
                    self.assertEqual(srow[key], orow[key], (src["id"], key))
                self.assertAlmostEqual(orow["u"], out["uv"][srow["uv_vertex"]][0])
                self.assertAlmostEqual(orow["v"], out["uv"][srow["uv_vertex"]][1])
                self.assertEqual(orow["island_id"], src["id"])


class TestSeamIdentityInAtlas(unittest.TestCase):
    def test_split_uv_vertices_remain_separate_across_seams(self):
        app = create_app()
        # cube opened by a spanning-tree seam: original vertices recur as
        # multiple UV vertices that must never be welded by coordinate.
        v, f = cube_closed()
        sm = split_mesh(v, f, cube_spanning_tree_seams())
        ok, _ = check_disk(sm)
        self.assertTrue(ok)
        boundary = sm.boundary_uvs()
        req = {
            "vertices": v.tolist(),
            "faces": f.tolist(),
            "seam_edges": cube_spanning_tree_seams(),
            "anchor0": boundary[0],
            "anchor1": boundary[5],
        }
        island = solved_island(app, "cube", req)
        atlas = pack_islands([island], 0.05)
        out = atlas["islands"][0]
        uvs_by_orig = {}
        for row in out["corners"]:
            uvs_by_orig.setdefault(row["orig_vertex"], set()).add(row["uv_vertex"])
        # some original vertex must occur under several split uv ids
        self.assertTrue(any(len(s) > 1 for s in uvs_by_orig.values()))
        # and the copies keep distinct (seam-separated) coordinates
        uv = np.array(out["uv"])
        for s in uvs_by_orig.values():
            if len(s) > 1:
                pts = uv[list(s)]
                self.assertGreater(np.ptp(pts, axis=0).max(), 1e-6)


class TestPackingValidation(unittest.TestCase):
    def _good(self):
        app = create_app()
        return [solved_island(app, "A", plane_request(2))]

    def test_padding_must_be_below_limit(self):
        with self.assertRaises(PackingError):
            pack_islands(self._good(), 0.5)  # 1/(1+1) = 0.5
        with self.assertRaises(PackingError):
            pack_islands(self._good(), -0.01)

    def test_non_finite_and_degenerate_uv_rejected(self):
        bad = self._good()
        bad[0]["uv"][0, 0] = np.nan
        with self.assertRaises(PackingError):
            pack_islands(bad, 0.05)

    def test_duplicate_and_typed_ids_required(self):
        good = self._good()[0]
        dup = [dict(good, id="A"), dict(good, id="A")]
        with self.assertRaises(PackingError):
            pack_islands(dup, 0.05)
        bad_id = [dict(good, id=7)]
        with self.assertRaises(PackingError):
            pack_islands(bad_id, 0.05)


class TestAtlasAPI(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()

    def _payload(self, ids=("C", "A", "B")):
        specs = {"A": plane_request(2), "B": plane_request(3),
                 "C": plane_request(4)}
        return {"padding": 0.04,
                "islands": [{"id": i, "mesh": specs[i]} for i in ids]}

    def test_happy_path_and_response_order(self):
        r = self.client.post("/api/atlas", json=self._payload())
        self.assertEqual(r.status_code, 200, r.json)
        self.assertTrue(r.json["ok"])
        self.assertEqual([i["id"] for i in r.json["atlas"]["islands"]],
                         ["C", "A", "B"])
        scales = {i["scale"] for i in r.json["atlas"]["islands"]}
        self.assertEqual(len(scales), 1)

    def test_reordering_request_keeps_coordinates_with_their_id(self):
        r1 = self.client.post("/api/atlas", json=self._payload(("C", "A", "B")))
        r2 = self.client.post("/api/atlas", json=self._payload(("B", "A", "C")))
        a = {i["id"]: i for i in r1.json["atlas"]["islands"]}
        b = {i["id"]: i for i in r2.json["atlas"]["islands"]}
        for iid in a:
            self.assertEqual(a[iid]["uv"], b[iid]["uv"], iid)
            self.assertEqual(
                [(c["face"], c["corner"], c["u"], c["v"]) for c in a[iid]["corners"]],
                [(c["face"], c["corner"], c["u"], c["v"]) for c in b[iid]["corners"]],
            )

    def test_one_illegal_island_rejects_whole_atlas(self):
        payload = self._payload()
        payload["islands"][1]["mesh"]["vertices"] = []  # invalid mesh
        r = self.client.post("/api/atlas", json=payload)
        self.assertEqual(r.status_code, 400)
        self.assertFalse(r.json["ok"])
        self.assertNotIn("atlas", r.json)

    def test_non_disk_island_rejects_whole_atlas(self):
        # closed octahedron: witness, LSCM must never run
        from samples import octahedron

        v, f = octahedron()
        payload = self._payload()
        payload["islands"][0]["mesh"] = {
            "vertices": v.tolist(), "faces": f.tolist(), "seam_edges": [],
            "anchor0": 0, "anchor1": 1,
        }
        r = self.client.post("/api/atlas", json=payload)
        self.assertEqual(r.status_code, 400)
        self.assertIn("unusable", r.json["error"])

    def test_padding_validation_at_api(self):
        payload = self._payload()
        payload["padding"] = 0.5
        r = self.client.post("/api/atlas", json=payload)
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/atlas", json={**payload, "padding": True})
        self.assertEqual(r.status_code, 400)

    def test_id_rules(self):
        payload = self._payload()
        payload["islands"][1]["id"] = "C"  # duplicate
        self.assertEqual(self.client.post("/api/atlas", json=payload).status_code, 400)
        payload = self._payload()
        payload["islands"][1]["id"] = 12
        self.assertEqual(self.client.post("/api/atlas", json=payload).status_code, 400)

    def test_count_limits(self):
        payload = self._payload()
        payload["islands"] = []
        self.assertEqual(self.client.post("/api/atlas", json=payload).status_code, 400)
        payload = self._payload(("A",) * 9)
        self.assertEqual(self.client.post("/api/atlas", json=payload).status_code, 400)

    def test_bad_json_body_rejected(self):
        r = self.client.post(
            "/api/atlas", data="{not json", content_type="application/json"
        )
        self.assertEqual(r.status_code, 400)

    def test_sample_endpoint(self):
        r = self.client.get("/api/atlas/sample")
        self.assertEqual(r.status_code, 200)
        req = r.json
        rr = self.client.post("/api/atlas", json=req)
        self.assertEqual(rr.status_code, 200, rr.json)
        self.assertTrue(rr.json["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
