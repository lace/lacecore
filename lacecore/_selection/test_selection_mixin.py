import numpy as np
from polliwog import Plane
import vg
from .._mesh import Mesh
from ..test_group_map import create_group_map


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
        [0, 1, 2],
        [0, 2, 3],
        [7, 6, 5],
        [7, 5, 4],
        [4, 5, 1],
        [4, 1, 0],
        [5, 6, 2],
        [5, 2, 1],
        [6, 7, 3],
        [6, 3, 2],
        [3, 7, 4],
        [3, 4, 0],
    ]
)
cube_at_origin = Mesh(v=cube_vertices, f=cube_faces)


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
    submesh = Mesh(
        v=cube_vertices, f=cube_faces, face_groups=create_group_map()
    ).picking_face_groups("top")

    np.testing.assert_array_equal(submesh.v, cube_vertices[np.array([0, 3, 4, 7])])
    np.testing.assert_array_equal(submesh.v[submesh.f], cube_vertices[cube_faces[10:]])
