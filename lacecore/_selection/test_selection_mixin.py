from lacecore import GroupMap, Mesh
import numpy as np
from polliwog import Plane
from vg.compat import v2 as vg


cube_vertices = np.array(
    [
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [3.0, 0.0, 3.0],
        [0.0, 0.0, 3.0],
        [0.0, 3.0, 0.0],
        [3.0, 3.0, 0.0],
        [3.0, 3.0, 3.0],
        [0.0, 3.0, 3.0],
    ]
)
cube_faces = np.array(
    [
        # Bottom (-y).
        [0, 1, 2],
        [0, 2, 3],
        # Top (+y).
        [7, 6, 5],
        [7, 5, 4],
        # Back side (-z).
        [4, 5, 1],
        [4, 1, 0],
        # Right side (+x).
        [5, 6, 2],
        [5, 2, 1],
        # Front side (+z).
        [6, 7, 3],
        [6, 3, 2],
        # Left side (-x).
        [3, 7, 4],
        [3, 4, 0],
    ]
)


def create_group_map():
    return GroupMap.from_dict(
        {
            "bottom": [0, 1],
            "top": [2, 3],
            "back_side": [4, 5],
            "right_side": [6, 7],
            "front_side": [8, 9],
            "left_side": [10, 11],
            "sides": [4, 5, 6, 7, 8, 9, 10, 11],
            "top_and_bottom": [0, 1, 2, 3],
            "empty": [],
        },
        12,
    )


cube_at_origin = Mesh(v=cube_vertices, f=cube_faces, face_groups=create_group_map())


def assert_subcube(submesh, expected_vertex_indices, expected_face_indices):
    np.testing.assert_array_equal(submesh.v, cube_vertices[expected_vertex_indices])
    np.testing.assert_array_equal(
        submesh.v[submesh.f], cube_vertices[cube_faces[expected_face_indices]]
    )


def test_vertices_at_or_above():
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_at_or_above(
            2, np.array([1.0, 1.0, 1.0])
        ),
        expected_vertex_indices=[2, 3, 6, 7],
        expected_face_indices=[8, 9],
    )
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_at_or_above(
            2, np.array([1.0, 1.0, 3.0])
        ),
        expected_vertex_indices=[2, 3, 6, 7],
        expected_face_indices=[8, 9],
    )
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_at_or_above(
            2, np.array([1.0, 1.0, -1.0])
        ),
        expected_vertex_indices=range(8),
        expected_face_indices=range(12),
    )


def test_vertices_on_or_in_front_of_plane():
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_on_or_in_front_of_plane(
            Plane(np.array([1.0, 1.0, 1.0]), vg.basis.z)
        ),
        expected_vertex_indices=[2, 3, 6, 7],
        expected_face_indices=[8, 9],
    )
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_on_or_in_front_of_plane(
            Plane(np.array([1.0, 1.0, 3.0]), vg.basis.z)
        ),
        expected_vertex_indices=[2, 3, 6, 7],
        expected_face_indices=[8, 9],
    )
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_on_or_in_front_of_plane(
            Plane(np.array([1.0, 1.0, -1.0]), vg.basis.z)
        ),
        expected_vertex_indices=range(8),
        expected_face_indices=range(12),
    )


def test_vertices_above():
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_above(2, np.array([1.0, 1.0, 1.0])),
        expected_vertex_indices=[2, 3, 6, 7],
        expected_face_indices=[8, 9],
    )
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_above(2, np.array([1.0, 1.0, 3.0])),
        expected_vertex_indices=[],
        expected_face_indices=[],
    )
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_above(2, np.array([1.0, 1.0, -1.0])),
        expected_vertex_indices=range(8),
        expected_face_indices=range(12),
    )


def test_vertices_in_front_of_plane():
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_in_front_of_plane(
            Plane(np.array([1.0, 1.0, 1.0]), vg.basis.z)
        ),
        expected_vertex_indices=[2, 3, 6, 7],
        expected_face_indices=[8, 9],
    )
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_in_front_of_plane(
            Plane(np.array([1.0, 1.0, 3.0]), vg.basis.z)
        ),
        expected_vertex_indices=[],
        expected_face_indices=[],
    )
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_in_front_of_plane(
            Plane(np.array([1.0, 1.0, -1.0]), vg.basis.z)
        ),
        expected_vertex_indices=range(8),
        expected_face_indices=range(12),
    )


def test_vertices_at_or_below():
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_at_or_below(
            2, np.array([1.0, 1.0, 1.0])
        ),
        expected_vertex_indices=[0, 1, 4, 5],
        expected_face_indices=[4, 5],
    )
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_at_or_below(
            2, np.array([1.0, 1.0, 0.0])
        ),
        expected_vertex_indices=[0, 1, 4, 5],
        expected_face_indices=[4, 5],
    )
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_at_or_below(
            2, np.array([1.0, 1.0, 3.0])
        ),
        expected_vertex_indices=range(8),
        expected_face_indices=range(12),
    )


def test_vertices_on_or_behind_plane():
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_on_or_behind_plane(
            Plane(np.array([1.0, 1.0, 1.0]), vg.basis.z)
        ),
        expected_vertex_indices=[0, 1, 4, 5],
        expected_face_indices=[4, 5],
    )
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_on_or_behind_plane(
            Plane(np.array([1.0, 1.0, 0.0]), vg.basis.z)
        ),
        expected_vertex_indices=[0, 1, 4, 5],
        expected_face_indices=[4, 5],
    )
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_on_or_behind_plane(
            Plane(np.array([1.0, 1.0, 3.0]), vg.basis.z)
        ),
        expected_vertex_indices=range(8),
        expected_face_indices=range(12),
    )


def test_vertices_below():
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_below(2, np.array([1.0, 1.0, 1.0])),
        expected_vertex_indices=[0, 1, 4, 5],
        expected_face_indices=[4, 5],
    )
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_below(2, np.array([1.0, 1.0, 0.0])),
        expected_vertex_indices=[],
        expected_face_indices=[],
    )
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_below(2, np.array([1.0, 1.0, 4.0])),
        expected_vertex_indices=range(8),
        expected_face_indices=range(12),
    )


def test_vertices_behind_plane():
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_behind_plane(
            Plane(np.array([1.0, 1.0, 1.0]), vg.basis.z)
        ),
        expected_vertex_indices=[0, 1, 4, 5],
        expected_face_indices=[4, 5],
    )
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_behind_plane(
            Plane(np.array([1.0, 1.0, 0.0]), vg.basis.z)
        ),
        expected_vertex_indices=[],
        expected_face_indices=[],
    )
    assert_subcube(
        submesh=cube_at_origin.keeping_vertices_behind_plane(
            Plane(np.array([1.0, 1.0, 4.0]), vg.basis.z)
        ),
        expected_vertex_indices=range(8),
        expected_face_indices=range(12),
    )


def test_pick_vertices_list():
    wanted_vs = [3, 7, 4]
    submesh = cube_at_origin.picking_vertices(wanted_vs)

    np.testing.assert_array_equal(submesh.v, cube_vertices[np.array([3, 4, 7])])
    np.testing.assert_array_equal(
        submesh.v[submesh.f], cube_vertices[cube_faces[10:11]]
    )


def test_pick_vertices_mask():
    wanted_v_mask = np.zeros(8, dtype=np.bool)
    wanted_v_mask[[3, 7, 4]] = True
    submesh = cube_at_origin.picking_vertices(wanted_v_mask)

    np.testing.assert_array_equal(submesh.v, cube_vertices[np.array([3, 4, 7])])
    np.testing.assert_array_equal(
        submesh.v[submesh.f], cube_vertices[cube_faces[10:11]]
    )


def test_pick_faces_list():
    wanted_faces = [10, 11]
    submesh = cube_at_origin.picking_faces(wanted_faces)

    np.testing.assert_array_equal(submesh.v, cube_vertices[np.array([0, 3, 4, 7])])
    np.testing.assert_array_equal(submesh.v[submesh.f], cube_vertices[cube_faces[10:]])


def test_pick_face_groups():
    submesh = cube_at_origin.picking_face_groups("top")

    np.testing.assert_array_equal(submesh.v, cube_vertices[np.array([4, 5, 6, 7])])
    np.testing.assert_array_equal(submesh.v[submesh.f], cube_vertices[cube_faces[2:4]])


def test_sliced_by_plane():
    extent = np.max(cube_at_origin.v, axis=0)

    sliced = cube_at_origin.sliced_by_plane(Plane(extent - 0.05, np.array([1, 1, 1])))

    np.testing.assert_array_almost_equal(np.min(sliced.v, axis=0), extent - 0.15)
    np.testing.assert_array_almost_equal(np.max(sliced.v, axis=0), extent)
    np.testing.assert_array_equal(
        sliced.face_groups["bottom"].nonzero()[0], np.zeros(0)
    )

    # Confidence check.
    np.testing.assert_array_equal(
        sliced.face_normals(),
        np.array([vg.basis.y, vg.basis.x, vg.basis.z, vg.basis.z]),
    )

    for expected_empty_group in "bottom", "back_side", "empty":
        np.testing.assert_array_equal(
            sliced.face_groups[expected_empty_group].nonzero()[0], np.zeros(0)
        )
    np.testing.assert_array_equal(sliced.face_groups["top"].nonzero()[0], np.array([0]))
    np.testing.assert_array_equal(
        sliced.face_groups["right_side"].nonzero()[0], np.array([1])
    )
    np.testing.assert_array_equal(
        sliced.face_groups["front_side"].nonzero()[0], np.array([2, 3])
    )


def test_sliced_by_plane_two_planes():
    extent = np.max(cube_at_origin.v, axis=0)

    sliced = cube_at_origin.sliced_by_plane(
        Plane(extent - 0.05, np.array([1, 1, 1])),
        Plane(extent - 0.1, vg.basis.x),
    )

    np.testing.assert_array_almost_equal(np.max(sliced.v, axis=0), extent)
    np.testing.assert_array_almost_equal(
        np.min(sliced.v, axis=0), extent - np.array([0.1, 0.15, 0.15])
    )


def test_sliced_by_plane_selection():
    extent = np.max(cube_at_origin.v, axis=0)

    sliced = cube_at_origin.sliced_by_plane(
        Plane(extent - 0.05, np.array([1, 1, 1])),
        # This should leave untouched the bottom, which lies along the xy-plane,
        # where `z == 0`.
        only_for_selection=lambda selection: selection.pick_face_groups("top", "sides"),
    )

    # Ensure the bottom is still there.
    bottom_tris = cube_at_origin.v[
        cube_at_origin.f[cube_at_origin.face_groups["bottom"]]
    ]
    assert np.isin(bottom_tris, sliced.v[sliced.f]).all()

    np.testing.assert_array_almost_equal(np.min(sliced.v, axis=0), np.array([0, 0, 0]))
    np.testing.assert_array_almost_equal(np.max(sliced.v, axis=0), extent)
    assert len(sliced.f) == 6
