"""Tests: split identity, anchors, planar mesh, illegal cuts, export data."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lscm import SolveError, solve_lscm
from mesh import MeshError, split_mesh
from samples import cube_closed, cube_spanning_tree_seams, plane_grid
from topology import boundary_loops, check_disk


def two_triangles():
    # (0,0)-(1,0)-(0,1) and (1,0)-(1,1)-(0,1), CCW, sharing edge (2,1)
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    f = np.array([[0, 1, 2], [1, 3, 2]])
    return v, f


class TestSplitting(unittest.TestCase):
    def test_uncut_interior_edge_is_welded(self):
        v, f = two_triangles()
        sm = split_mesh(v, f, [])
        self.assertEqual(sm.n_uv, 4)
        self.assertEqual(sm.corner_uv[0].tolist(), [0, 1, 2])
        self.assertEqual(sm.corner_uv[1].tolist(), [1, 3, 2])

    def test_seam_splits_vertices_into_sectors_not_by_position(self):
        v, f = two_triangles()
        sm = split_mesh(v, f, [(1, 2)])  # cut the shared diagonal
        # endpoints of the seam must each exist as TWO uv vertices
        uvs_of_1 = {int(sm.corner_uv[0, 1]), int(sm.corner_uv[1, 0])}
        uvs_of_2 = {int(sm.corner_uv[0, 2]), int(sm.corner_uv[1, 2])}
        self.assertEqual(len(uvs_of_1), 2)
        self.assertEqual(len(uvs_of_2), 2)
        # identical 3D positions, distinct identities
        for a, b in [(uvs_of_1, 1), (uvs_of_2, 2)]:
            ps = np.array([sm.uv_vertices[x] for x in a])
            self.assertTrue(np.allclose(ps, v[b]))
        self.assertEqual(sm.n_uv, 6)  # 4 + two new sectors
        # orig ids are preserved on every copy
        for u in uvs_of_1:
            self.assertEqual(int(sm.orig_of_uv[u]), 1)
        for u in uvs_of_2:
            self.assertEqual(int(sm.orig_of_uv[u]), 2)

    def test_cut_interior_edge_separates_into_two_loops(self):
        # Cutting the single shared edge of a two-triangle patch cuts the
        # patch into two pieces joined only at two points: two boundary
        # loops, hence NOT a disk.  This is an illegal cut and must be
        # reported, not silently solved.
        v, f = two_triangles()
        sm = split_mesh(v, f, [(1, 2)])
        loops = boundary_loops(sm)
        self.assertEqual(len(loops), 2)
        ok, witness = check_disk(sm)
        self.assertFalse(ok)
        self.assertIn(witness["kind"], ("disconnected", "multiple_boundary_loops"))

    def test_dead_end_slit_from_boundary_is_a_disk(self):
        # A slit that starts on the outer boundary and ends at one interior
        # vertex opens one sector fan; the surface stays a single disk.
        n = 4
        v, f = plane_grid(n)
        # cut outer edge (0,1) and two edges to interior vertex (1,1)=5
        sm = split_mesh(v, f, [(0, 1), (1, 5)])
        loops = boundary_loops(sm)
        self.assertEqual(len(loops), 1)
        # slit sides are traversed, so one endpoint appears twice
        self.assertEqual(len(loops[0]), 4 * (n - 1) + 2)
        ok, witness = check_disk(sm)
        self.assertTrue(ok, witness)

    def test_more_than_200_triangles_rejected(self):
        v, f = plane_grid(15)  # 14*14*2 = 392
        with self.assertRaises(MeshError):
            split_mesh(v, f, [])

    def test_inconsistent_orientation_detected(self):
        v, f = two_triangles()
        f[1] = [1, 2, 3]  # shared edge now traversed 1->2 on both faces
        with self.assertRaises(MeshError):
            split_mesh(v, f, [])

    def test_degenerate_face_detected(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
        f = np.array([[0, 1, 2]])
        with self.assertRaises(MeshError):
            split_mesh(v, f, [])

    def test_seam_on_nonexistent_edge(self):
        v, f = two_triangles()
        with self.assertRaises(MeshError):
            split_mesh(v, f, [(0, 3)])  # the other diagonal of the square


class TestTopologyWitnesses(unittest.TestCase):
    def test_closed_surface_witness(self):
        from samples import octahedron

        v, f = octahedron()
        sm = split_mesh(v, f, [])
        ok, w = check_disk(sm)
        self.assertFalse(ok)
        self.assertEqual(w["kind"], "closed_surface")
        self.assertEqual(sm.boundary_edges, [])

    def test_seam_loop_detaches_component(self):
        # cut both diagonals? simpler: cut the whole interior edge in the
        # 2-triangle square is fine (one loop). Detach a triangle instead:
        # cut the three edges of one face of a 4-face patch
        n = 3
        v, f = plane_grid(n)
        # corner triangle 0:1:3 has only one interior edge; cutting it
        # peels the triangle off as its own component
        sm = split_mesh(v, f, [(1, 3)])
        ok, w = check_disk(sm)
        self.assertFalse(ok)
        self.assertIn(w["kind"], ("disconnected", "multiple_boundary_loops"))

    def test_cube_closed_not_disk(self):
        v, f = cube_closed()
        sm = split_mesh(v, f, [])
        ok, w = check_disk(sm)
        self.assertFalse(ok)
        self.assertEqual(w["kind"], "closed_surface")

    def test_cube_spanning_tree_is_disk(self):
        v, f = cube_closed()
        sm = split_mesh(v, f, cube_spanning_tree_seams())
        ok, w = check_disk(sm)
        self.assertTrue(ok, w)
        self.assertEqual(len(boundary_loops(sm)), 1)


class TestLSCM(unittest.TestCase):
    def _plane_case(self, k):
        v, f = plane_grid(k)
        sm = split_mesh(v, f, [])
        ok, _ = check_disk(sm)
        self.assertTrue(ok)
        return sm

    def test_planar_grid_conformal_residual_zero(self):
        sm = self._plane_case(4)
        boundary = sm.boundary_uvs()
        a0, a1 = boundary[0], boundary[-1]
        res = solve_lscm(sm, a0, a1)
        self.assertFalse(res["rank_deficient"])
        self.assertEqual(res["rank"], res["expected_rank"])
        self.assertLess(res["max_conformal_residual"], 1e-10)
        self.assertLess(res["rms_row_residual"], 1e-12)
        self.assertEqual(res["flipped_faces"], [])

    def test_anchors_pinned_exactly(self):
        sm = self._plane_case(4)
        a0, a1 = 0, sm.n_uv - 1
        res = solve_lscm(sm, a0, a1)
        uv = res["uv"]
        np.testing.assert_allclose(uv[a0], [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(uv[a1], [1.0, 0.0], atol=1e-12)

    def test_planar_solution_is_similarity(self):
        sm = self._plane_case(5)
        res = solve_lscm(sm, 0, 1)
        uv = res["uv"]
        # first edge maps to unit length: global scale 1/grid_spacing=1/1
        # and the planar disk must map rigidly (rotation 0)
        for u in range(sm.n_uv):
            orig = int(sm.orig_of_uv[u])
            np.testing.assert_allclose(uv[u], sm.vertices[orig][:2], atol=1e-9)

    def test_distinct_anchors_required(self):
        sm = self._plane_case(3)
        with self.assertRaises(SolveError):
            solve_lscm(sm, 0, 0)

    def test_anchors_must_be_boundary(self):
        sm = self._plane_case(4)  # interior vertex 5 of 4x4 grid
        with self.assertRaises(SolveError):
            solve_lscm(sm, 0, 5)

    def test_cube_unwrap_full_rank_no_flips(self):
        v, f = cube_closed()
        sm = split_mesh(v, f, cube_spanning_tree_seams())
        boundary = sm.boundary_uvs()
        res = solve_lscm(sm, boundary[0], boundary[5])
        self.assertFalse(res["rank_deficient"], res["smallest_singular_values"])
        self.assertEqual(res["flipped_faces"], [])
        self.assertEqual(res["degenerate_uv_faces"], [])
        # cube developable -> conformal residual at numerical noise
        self.assertLess(res["max_conformal_residual"], 1e-9)

    def test_export_rows_carry_face_and_corner_ids(self):
        sm = self._plane_case(3)
        res = solve_lscm(sm, 0, 1)
        # emulate the export the Flask layer builds
        rows = []
        for fi, tri in enumerate(sm.corner_uv):
            for j in range(3):
                u = int(tri[j])
                rows.append(
                    (fi, j, int(sm.faces[fi, j]), u, res["uv"][u, 0], res["uv"][u, 1])
                )
        self.assertEqual(len(rows), 3 * sm.n_faces)
        # one row per corner of face 0, distinct local ids
        face0 = [r for r in rows if r[0] == 0]
        self.assertEqual(sorted(r[1] for r in face0), [0, 1, 2])
        # uv coordinates finite everywhere
        self.assertTrue(all(np.isfinite(r[4:6]).all() for r in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
