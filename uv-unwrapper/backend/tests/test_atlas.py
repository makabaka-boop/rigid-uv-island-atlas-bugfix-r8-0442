"""Tests for the multi-island atlas: one common positive scale, proper
rotations only, stable identity follow-through, uniform padding, request
validation and all-or-nothing rejection.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import _solve_payload, create_app
from packing import PackingError, pack_islands, validate_request
from samples import cube_closed, cube_spanning_tree_seams, plane_grid, plane_rect


def plane_mesh(ni, nj=None, anchors=(0, 1)):
    v, f = (plane_grid(ni) if nj is None else plane_rect(ni, nj))
    return {
        "vertices": v.tolist(),
        "faces": f.tolist(),
        "seam_edges": [],
        "anchor0": int(anchors[0]),
        "anchor1": int(anchors[1]),
    }


def cube_mesh():
    v, f = cube_closed()
    return {
        "vertices": v.tolist(),
        "faces": f.tolist(),
        "seam_edges": cube_spanning_tree_seams(),
        "anchor0": 0,
        "anchor1": 1,
    }


def solved(iid, mesh):
    return {"id": iid, "result": _solve_payload(mesh)}


def octahedron_mesh():
    # closed surface: prepare rejects it with a witness, solve is impossible
    v = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        float,
    )
    faces = np.array(
        [
            [4, 0, 2], [4, 2, 1], [4, 1, 3], [4, 3, 0],
            [5, 2, 0], [5, 1, 2], [5, 3, 1], [5, 0, 3],
        ]
    )
    return {
        "vertices": v.tolist(),
        "faces": faces.tolist(),
        "seam_edges": [],
        "anchor0": 0,
        "anchor1": 1,
    }


def faces_by_id(corners):
    out = {}
    for c in corners:
        out.setdefault(c["face"], {})[c["corner"]] = c
    return out


def area2(c):
    p = [c[0], c[1], c[2]]
    return (p[1]["u"] - p[0]["u"]) * (p[2]["v"] - p[0]["v"]) - (
        p[1]["v"] - p[0]["v"]
    ) * (p[2]["u"] - p[0]["u"])


def fit_similarity(src_uv, placed_uv):
    """Least-squares fit placed = M @ src + t.  Returns (M, t, residual)."""
    q = np.asarray(src_uv, float)
    p = np.asarray(placed_uv, float)
    A = np.column_stack([q, np.ones(len(q))])
    coef, *_ = np.linalg.lstsq(A, p, rcond=None)
    M = coef[:2].T
    t = coef[2]
    residual = float(np.max(np.abs(A @ coef - p)))
    return M, t, residual


class TestCommonScale(unittest.TestCase):
    def setUp(self):
        self.small = solved("small", plane_mesh(3))      # source extent 2
        self.large = solved("large", plane_mesh(5))      # source extent 4
        self.atlas = pack_islands([self.small, self.large], 0.02)

    def test_one_scale_reported_by_every_island(self):
        s = self.atlas["scale"]
        self.assertGreater(s, 0.0)
        self.assertTrue(np.isfinite(s))
        for island in self.atlas["islands"]:
            self.assertEqual(island["scale"], s)

    def test_relative_size_is_preserved_not_normalized_per_island(self):
        ext = {}
        for island in self.atlas["islands"]:
            us = [c["u"] for c in island["corners"]]
            vs = [c["v"] for c in island["corners"]]
            ext[island["id"]] = (max(us) - min(us), max(vs) - min(vs))
        # 5x5 grid is twice as wide/tall as 3x3; that ratio must survive.
        self.assertAlmostEqual(ext["large"][0] / ext["small"][0], 2.0, places=9)
        self.assertAlmostEqual(ext["large"][1] / ext["small"][1], 2.0, places=9)

    def test_unit_3d_edges_map_to_the_same_atlas_length_everywhere(self):
        # Every 3D unit edge of either planar island must cover the same
        # atlas length (the single common scale).
        lengths = {}
        for island in self.atlas["islands"]:
            placed = np.array(island["uv"])
            res_uv = np.array(island["source"]["uv"])
            d_src = np.linalg.norm(res_uv[1] - res_uv[0])  # anchor edge length
            d_placed = np.linalg.norm(placed[1] - placed[0])
            lengths[island["id"]] = d_placed / d_src
        self.assertAlmostEqual(lengths["small"], lengths["large"], places=12)


class TestRotationAndShape(unittest.TestCase):
    def test_placed_islands_are_pure_similarities_with_positive_determinant(self):
        islands = [
            solved("wide", plane_mesh(3, 5)),    # wide, may take 90 degrees
            solved("square", plane_mesh(4)),
            solved("cube", cube_mesh()),
        ]
        atlas = pack_islands(islands, 0.03)
        for island in atlas["islands"]:
            M, t, residual = fit_similarity(island["source"]["uv"], island["uv"])
            self.assertLess(residual, 1e-9, island["id"])
            det = np.linalg.det(M)
            self.assertGreater(det, 0.0, "%s must not be mirrored" % island["id"])
            # M is a rotation-scaled matrix: orthogonal columns, equal norm
            gram = M.T @ M
            self.assertAlmostEqual(gram[0, 0], gram[1, 1], places=9)
            self.assertAlmostEqual(gram[0, 1], 0.0, places=9)
            self.assertAlmostEqual(gram[1, 0], 0.0, places=9)

    def test_rotation_is_zero_or_quarter_turn_and_never_axis_swap_reflection(self):
        wide = solved("wide", plane_mesh(3, 5))  # source: 2 wide, 4 tall
        atlas = pack_islands([wide], 0.05)
        island = atlas["islands"][0]
        self.assertIn(island["rotation_deg"], (0, 90))
        placed = np.array(island["uv"])
        ext_x = np.ptp(placed[:, 0])
        ext_y = np.ptp(placed[:, 1])
        # A reflection (u,v)->(v,u) would still orient bands but wind faces
        # backwards: assert every face keeps positive signed area instead.
        for face, tri in faces_by_id(island["corners"]).items():
            self.assertGreater(area2(tri), 0.0, "face %d flipped" % face)
        # and the shape extents are exactly the source extents (2,4), common
        # scale shared with the other test islands implicitly.
        src = np.array(island["source"]["uv"])
        s = atlas["scale"]
        if island["rotation_deg"] == 90:
            self.assertAlmostEqual(ext_x / s, np.ptp(src[:, 1]), places=9)
            self.assertAlmostEqual(ext_y / s, np.ptp(src[:, 0]), places=9)
        else:
            self.assertAlmostEqual(ext_x / s, np.ptp(src[:, 0]), places=9)
            self.assertAlmostEqual(ext_y / s, np.ptp(src[:, 1]), places=9)

    def test_all_face_areas_stay_positive(self):
        atlas = pack_islands([solved("w", plane_mesh(3, 5)),
                              solved("c", cube_mesh())], 0.03)
        for island in atlas["islands"]:
            for tri in faces_by_id(island["corners"]).values():
                self.assertGreater(area2(tri), 0.0)


class TestIdentityFollowThrough(unittest.TestCase):
    def test_id_travels_with_its_own_faces_regardless_of_sorted_layout(self):
        # request order: "z-small" first, "a-large" second; sorted layout puts
        # a-large in band 0.  Old code attached islands[index]'s id.
        small = solved("z-small", plane_mesh(3))   # 8 faces
        large = solved("a-large", plane_mesh(5))   # 32 faces
        atlas = pack_islands([small, large], 0.02)
        ids = [i["id"] for i in atlas["islands"]]
        self.assertEqual(ids, ["z-small", "a-large"])  # request order
        z, a = atlas["islands"]
        self.assertEqual(len(z["corners"]) // 3, 8)
        self.assertEqual(len(a["corners"]) // 3, 32)
        for island, src in ((z, small), (a, large)):
            self.assertTrue(all(c["island_id"] == island["id"]
                                for c in island["corners"]))
            # angle errors and orig/uv vertex identities match its OWN solve
            src_rows = {(r["face"], r["corner"]): r
                        for r in src["result"]["corners"]}
            for row in island["corners"]:
                ref = src_rows[(row["face"], row["corner"])]
                self.assertEqual(row["orig_vertex"], ref["orig_vertex"])
                self.assertEqual(row["uv_vertex"], ref["uv_vertex"])
                self.assertAlmostEqual(
                    row["angle_error_deg"], ref["angle_error_deg"], places=12
                )

    def test_layout_per_id_is_stable_under_request_permutation(self):
        a = solved("a", plane_mesh(4))
        b = solved("b", plane_mesh(3, 5))
        c = solved("c", cube_mesh())
        placed = {}
        for order in ([a, b, c], [c, a, b], [b, c, a]):
            atlas = pack_islands(list(order), 0.025)
            self.assertEqual([i["id"] for i in atlas["islands"]],
                             [i["id"] for i in order])
            for island in atlas["islands"]:
                placed.setdefault(island["id"], []).append(
                    np.array(island["uv"])
                )
        for iid, variants in placed.items():
            for v in variants[1:]:
                np.testing.assert_allclose(v, variants[0], atol=1e-12)

    def test_seam_split_vertices_are_not_merged_by_position(self):
        cube = solved("cube", cube_mesh())
        atlas = pack_islands([cube], 0.03)
        island = atlas["islands"][0]
        n_uv = island["source"]["prepare"]["n_uv_vertices"]
        self.assertEqual(len(island["uv"]), n_uv)
        # same original vertex can own several distinct UV vertices
        orig_of = {}
        for row in island["corners"]:
            orig_of.setdefault(row["orig_vertex"], set()).add(row["uv_vertex"])
        duplicated = {o: uvs for o, uvs in orig_of.items() if len(uvs) > 1}
        self.assertTrue(duplicated, "a cut cube must contain split vertices")
        for o, uvs in duplicated.items():
            coords = [tuple(island["uv"][u]) for u in uvs]
            # distinct seam-side UV vertices stay distinct after the packing
            self.assertEqual(len(set(coords)), len(coords))

    def test_islands_with_identical_3d_positions_stay_separate_identities(self):
        a = solved("alpha", plane_mesh(3))
        b = solved("beta", plane_mesh(3))
        atlas = pack_islands([a, b], 0.02)
        self.assertEqual(len(atlas["islands"]), 2)
        self.assertNotEqual(atlas["islands"][0]["corners"][0]["island_id"],
                            atlas["islands"][1]["corners"][0]["island_id"])


class TestPadding(unittest.TestCase):
    def test_uniform_page_margin_and_inter_band_gap(self):
        p = 0.04
        atlas = pack_islands(
            [solved("a", plane_mesh(3)), solved("b", plane_mesh(4)),
             solved("c", plane_mesh(3, 5))], p
        )
        boxes = []
        for island in atlas["islands"]:
            us = [c["u"] for c in island["corners"]]
            vs = [c["v"] for c in island["corners"]]
            box = (min(us), min(vs), max(us), max(vs))
            boxes.append((island["band"], box))
            self.assertGreaterEqual(box[0], p - 1e-12)
            self.assertGreaterEqual(box[1], p - 1e-12)
            self.assertLessEqual(box[2], 1 - p + 1e-12)
            self.assertLessEqual(box[3], 1 - p + 1e-12)
        boxes.sort()
        for (_, upper), (_, lower) in zip(boxes, boxes[1:]):
            self.assertGreaterEqual(lower[1] - upper[3], p - 1e-12)
        self.assertAlmostEqual(atlas["padding"], p)

    def test_padding_validated(self):
        valid = [{"id": "a", "mesh": plane_mesh(3)}]
        bad_each = (-0.01, 0.5, 1.0, "0.05", True, False, float("nan"), None)
        for bad in bad_each:
            with self.assertRaises(PackingError):
                validate_request({"islands": valid, "padding": bad})
        # packer-level guard for numeric junk as well
        islands = [{"id": "a", "result": _solve_payload(plane_mesh(3))}]
        for bad in (-0.01, 0.5, 1.0, float("nan")):
            with self.assertRaises(PackingError):
                pack_islands(islands, bad)
        # 1/(n+1) is the exact exclusive bound for n=3
        with self.assertRaises(PackingError):
            validate_request(
                {"islands": [{"id": "a", "mesh": plane_mesh(3)},
                             {"id": "b", "mesh": plane_mesh(3)},
                             {"id": "c", "mesh": plane_mesh(3)}],
                 "padding": 0.25}
            )
        # zero padding is allowed (edges may touch the page)
        validate_request({"islands": [{"id": "a", "mesh": plane_mesh(3)}],
                          "padding": 0})


class TestRequestValidation(unittest.TestCase):
    def test_count_limits(self):
        for n in (0, 9):
            with self.assertRaises(PackingError):
                validate_request(
                    {"islands": [{"id": str(i), "mesh": plane_mesh(3)}
                                 for i in range(n)], "padding": 0.01}
                )

    def test_ids_must_be_unique_nonempty_strings(self):
        good = {"mesh": plane_mesh(3)}
        for items in (
            [{"mesh": good}],                              # missing id
            [{"id": "", "mesh": good}],                   # empty
            [{"id": 7, "mesh": good}],                    # non-string
            [{"id": "a", "mesh": good},
             {"id": "a", "mesh": good}],                  # duplicate
            [{"id": "a"}],                                # missing mesh
            "not-a-list",
        ):
            with self.assertRaises(PackingError):
                validate_request({"islands": items, "padding": 0.02})


class TestAllOrNothingAPI(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()

    def post(self, islands, padding=0.03):
        return self.client.post(
            "/api/atlas", json={"islands": islands, "padding": padding}
        )

    def test_happy_path_200_and_request_order(self):
        r = self.post(
            [{"id": "z", "mesh": plane_mesh(3)},
             {"id": "a", "mesh": plane_mesh(4)}]
        )
        self.assertEqual(r.status_code, 200, r.get_json())
        data = r.get_json()
        self.assertTrue(data["ok"])
        self.assertEqual([i["id"] for i in data["atlas"]["islands"]], ["z", "a"])

    def test_one_unsolvable_island_rejects_the_whole_atlas(self):
        r = self.post(
            [{"id": "good", "mesh": plane_mesh(3)},
             {"id": "closed", "mesh": octahedron_mesh()}]
        )
        self.assertEqual(r.status_code, 400)
        data = r.get_json()
        self.assertFalse(data["ok"])
        self.assertNotIn("atlas", data)  # nothing partial is exported
        self.assertIn("closed", data["error"])

    def test_bad_padding_and_ids_rejected(self):
        for payload in (
            {"islands": [{"id": "a", "mesh": plane_mesh(3)}], "padding": 0.9},
            {"islands": [{"id": "a", "mesh": plane_mesh(3)}], "padding": -1},
            {"islands": [{"id": "a", "mesh": plane_mesh(3)}], "padding": "x"},
            {"islands": [{"id": "a", "mesh": plane_mesh(3)},
                         {"id": "a", "mesh": plane_mesh(3)}], "padding": 0.02},
            {"islands": [], "padding": 0.02},
            {"islands": [{"id": "x",
                          "mesh": {"vertices": [], "faces": []}}],
             "padding": 0.02},
        ):
            r = self.client.post("/api/atlas", json=payload)
            self.assertEqual(r.status_code, 400, payload)
            self.assertFalse(r.get_json()["ok"])

    def test_illegal_cut_island_rejects_atlas(self):
        # two-triangle patch cut along its single shared edge: not a disk
        v = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
        f = [[0, 1, 2], [1, 3, 2]]
        bad = {"vertices": v, "faces": f, "seam_edges": [[1, 2]],
               "anchor0": 0, "anchor1": 1}
        r = self.post([{"id": "good", "mesh": plane_mesh(3)},
                       {"id": "badcut", "mesh": bad}])
        self.assertEqual(r.status_code, 400)
        self.assertIn("badcut", r.get_json()["error"])

    def test_sample_endpoint_request_actually_packs(self):
        req = self.client.get("/api/atlas/sample").get_json()
        r = self.client.post("/api/atlas", json=req)
        self.assertEqual(r.status_code, 200, r.get_json())
        ids = [i["id"] for i in r.get_json()["atlas"]["islands"]]
        self.assertEqual(ids, [i["id"] for i in req["islands"]])

    def test_malformed_json_body_rejected(self):
        r = self.client.post("/api/atlas", data="not json",
                             content_type="application/json")
        self.assertEqual(r.status_code, 400)


class TestPackerRejectsBadEmbeddings(unittest.TestCase):
    def _result_with_uv(self, uv, corners):
        return {"uv": uv, "corners": corners,
                "flipped_faces": [], "degenerate_uv_faces": []}

    def test_flipped_embedding_rejected(self):
        # mirrored unit square triangle: negative signed area
        uv = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        corners = [
            {"face": 0, "corner": 0, "orig_vertex": 0, "uv_vertex": 0},
            {"face": 0, "corner": 1, "orig_vertex": 1, "uv_vertex": 1},
            {"face": 0, "corner": 2, "orig_vertex": 2, "uv_vertex": 2},
        ]
        with self.assertRaises(PackingError):
            pack_islands([{"id": "x",
                           "result": self._result_with_uv(uv, corners)}], 0.02)

    def test_nonfinite_and_zero_extent_rejected(self):
        base_corners = [
            {"face": 0, "corner": j, "orig_vertex": j, "uv_vertex": j}
            for j in range(3)
        ]
        with self.assertRaises(PackingError):
            pack_islands(
                [{"id": "x", "result": self._result_with_uv(
                    [[0, 0], [1, 0], [float("nan"), 1]], base_corners)}],
                0.02,
            )
        # all points collinear in both orientations -> zero extent
        with self.assertRaises(PackingError):
            pack_islands(
                [{"id": "x", "result": self._result_with_uv(
                    [[0, 0], [1, 0], [2, 0]],
                    [{"face": 0, "corner": 0, "orig_vertex": 0, "uv_vertex": 0},
                     {"face": 0, "corner": 1, "orig_vertex": 1, "uv_vertex": 1},
                     {"face": 0, "corner": 2, "orig_vertex": 2, "uv_vertex": 2}])}],
                0.02,
            )

    def test_gate_flags_from_solve_result_are_honored(self):
        good = solved("g", plane_mesh(3))
        good["result"]["flipped_faces"] = [0]
        with self.assertRaises(PackingError):
            pack_islands([good], 0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
