import numpy as np
import pytest
import vg
from polliwog import Plane
from ._mesh import Mesh
from . import shapes

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
        submesh=cube_at_origin.select()
        .vertices_at_or_above(np.array([1.0, 1.0, 1.0]), 2)
        .end(),
        expected_vertex_indices=[2, 3, 6, 7],
        expected_face_indices=[8, 9],
    )
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_at_or_above(np.array([1.0, 1.0, 3.0]), 2)
        .end(),
        expected_vertex_indices=[2, 3, 6, 7],
        expected_face_indices=[8, 9],
    )
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_at_or_above(np.array([1.0, 1.0, -1.0]), 2)
        .end(),
        expected_vertex_indices=range(8),
        expected_face_indices=range(12),
    )


def test_vertices_on_or_in_front_of_plane():
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_on_or_in_front_of_plane(Plane(np.array([1.0, 1.0, 1.0]), vg.basis.z))
        .end(),
        expected_vertex_indices=[2, 3, 6, 7],
        expected_face_indices=[8, 9],
    )
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_on_or_in_front_of_plane(Plane(np.array([1.0, 1.0, 3.0]), vg.basis.z))
        .end(),
        expected_vertex_indices=[2, 3, 6, 7],
        expected_face_indices=[8, 9],
    )
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_on_or_in_front_of_plane(Plane(np.array([1.0, 1.0, -1.0]), vg.basis.z))
        .end(),
        expected_vertex_indices=range(8),
        expected_face_indices=range(12),
    )


def test_vertices_above():
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_above(np.array([1.0, 1.0, 1.0]), 2)
        .end(),
        expected_vertex_indices=[2, 3, 6, 7],
        expected_face_indices=[8, 9],
    )
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_above(np.array([1.0, 1.0, 3.0]), 2)
        .end(),
        expected_vertex_indices=[],
        expected_face_indices=[],
    )
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_above(np.array([1.0, 1.0, -1.0]), 2)
        .end(),
        expected_vertex_indices=range(8),
        expected_face_indices=range(12),
    )


def test_vertices_in_front_of_plane():
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_in_front_of_plane(Plane(np.array([1.0, 1.0, 1.0]), vg.basis.z))
        .end(),
        expected_vertex_indices=[2, 3, 6, 7],
        expected_face_indices=[8, 9],
    )
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_in_front_of_plane(Plane(np.array([1.0, 1.0, 3.0]), vg.basis.z))
        .end(),
        expected_vertex_indices=[],
        expected_face_indices=[],
    )
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_in_front_of_plane(Plane(np.array([1.0, 1.0, -1.0]), vg.basis.z))
        .end(),
        expected_vertex_indices=range(8),
        expected_face_indices=range(12),
    )


def test_vertices_at_or_below():
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_at_or_below(np.array([1.0, 1.0, 1.0]), 2)
        .end(),
        expected_vertex_indices=[0, 1, 4, 5],
        expected_face_indices=[4, 5],
    )
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_at_or_below(np.array([1.0, 1.0, 0.0]), 2)
        .end(),
        expected_vertex_indices=[0, 1, 4, 5],
        expected_face_indices=[4, 5],
    )
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_at_or_below(np.array([1.0, 1.0, 3.0]), 2)
        .end(),
        expected_vertex_indices=range(8),
        expected_face_indices=range(12),
    )


def test_vertices_on_or_behind_plane():
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_on_or_behind_plane(Plane(np.array([1.0, 1.0, 1.0]), vg.basis.z))
        .end(),
        expected_vertex_indices=[0, 1, 4, 5],
        expected_face_indices=[4, 5],
    )
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_on_or_behind_plane(Plane(np.array([1.0, 1.0, 0.0]), vg.basis.z))
        .end(),
        expected_vertex_indices=[0, 1, 4, 5],
        expected_face_indices=[4, 5],
    )
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_on_or_behind_plane(Plane(np.array([1.0, 1.0, 3.0]), vg.basis.z))
        .end(),
        expected_vertex_indices=range(8),
        expected_face_indices=range(12),
    )


def test_vertices_below():
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_below(np.array([1.0, 1.0, 1.0]), 2)
        .end(),
        expected_vertex_indices=[0, 1, 4, 5],
        expected_face_indices=[4, 5],
    )
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_below(np.array([1.0, 1.0, 0.0]), 2)
        .end(),
        expected_vertex_indices=[],
        expected_face_indices=[],
    )
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_below(np.array([1.0, 1.0, 4.0]), 2)
        .end(),
        expected_vertex_indices=range(8),
        expected_face_indices=range(12),
    )


def test_vertices_behind_plane():
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_behind_plane(Plane(np.array([1.0, 1.0, 1.0]), vg.basis.z))
        .end(),
        expected_vertex_indices=[0, 1, 4, 5],
        expected_face_indices=[4, 5],
    )
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_behind_plane(Plane(np.array([1.0, 1.0, 0.0]), vg.basis.z))
        .end(),
        expected_vertex_indices=[],
        expected_face_indices=[],
    )
    assert_subcube(
        submesh=cube_at_origin.select()
        .vertices_behind_plane(Plane(np.array([1.0, 1.0, 4.0]), vg.basis.z))
        .end(),
        expected_vertex_indices=range(8),
        expected_face_indices=range(12),
    )


def test_prune_orphan_vertices_has_no_effect_when_selecting_vertices():
    expected_vertex_indices = [2, 3, 6, 7]
    expected_face_indices = [8, 9]
    for prune_orphan_vertices in [True, False]:
        assert_subcube(
            submesh=cube_at_origin.select()
            .vertices_at_or_above(np.array([1.0, 1.0, 1.0]), 2)
            .end(prune_orphan_vertices=prune_orphan_vertices),
            expected_vertex_indices=expected_vertex_indices,
            expected_face_indices=expected_face_indices,
        )


def test_ret_indices_of_original_faces_and_vertices():
    expected_vertex_indices = [2, 3, 6, 7]
    expected_face_indices = [8, 9]
    (submesh, indices_of_original_faces, indices_of_original_vertices) = (
        cube_at_origin.select()
        .vertices_at_or_above(np.array([1.0, 1.0, 1.0]), 2)
        .end(ret_indices_of_original_faces_and_vertices=True)
    )
    assert_subcube(
        submesh=submesh,
        expected_vertex_indices=expected_vertex_indices,
        expected_face_indices=expected_face_indices,
    )
    np.testing.assert_array_equal(
        indices_of_original_faces,
        np.array([-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1]),
    )
    np.testing.assert_array_equal(
        indices_of_original_vertices, np.array([-1, -1, 0, 1, -1, -1, 2, 3])
    )


def test_selection_validation():
    selection = cube_at_origin.select()

    for method in (
        selection.vertices_at_or_above,
        selection.vertices_above,
        selection.vertices_at_or_below,
        selection.vertices_below,
    ):
        with pytest.raises(ValueError, match="Expected dim to be 0, 1, or 2"):
            method(np.zeros(3), 4)

    for method in (
        selection.vertices_in_front_of_plane,
        selection.vertices_on_or_in_front_of_plane,
        selection.vertices_behind_plane,
        selection.vertices_on_or_behind_plane,
    ):
        with pytest.raises(ValueError, match="Expected an instance of polliwog.Plane"):
            method("not-a-plane")


def test_pick_vertices_list():
    wanted_vs = [3, 7, 4]
    submesh = cube_at_origin.select().pick_vertices(wanted_vs).end()

    np.testing.assert_array_equal(submesh.v, cube_vertices[np.array([3, 4, 7])])
    np.testing.assert_array_equal(
        submesh.v[submesh.f], cube_vertices[cube_faces[10:11]]
    )


def test_pick_vertices_mask():
    wanted_v_mask = np.zeros(8, dtype=np.bool)
    wanted_v_mask[[3, 7, 4]] = True
    submesh = cube_at_origin.select().pick_vertices(wanted_v_mask).end()

    np.testing.assert_array_equal(submesh.v, cube_vertices[np.array([3, 4, 7])])
    np.testing.assert_array_equal(
        submesh.v[submesh.f], cube_vertices[cube_faces[10:11]]
    )


def test_pick_faces_list():
    wanted_faces = [10, 11]
    submesh = cube_at_origin.select().pick_faces(wanted_faces).end()

    np.testing.assert_array_equal(submesh.v, cube_vertices[np.array([0, 3, 4, 7])])
    np.testing.assert_array_equal(submesh.v[submesh.f], cube_vertices[cube_faces[10:]])


def test_union_of_vertices():
    assert_subcube(
        submesh=cube_at_origin.select()
        .pick_vertices([0, 1, 2])
        .union()
        .pick_vertices([3, 4, 7])
        .end(),
        expected_vertex_indices=[0, 1, 2, 3, 4, 7],
        expected_face_indices=[0, 1, 5, 10, 11],
    )


def test_union_of_faces():
    assert_subcube(
        submesh=cube_at_origin.select()
        .pick_faces([0, 11])
        .union()
        .pick_faces([11, 10])
        .end(),
        expected_vertex_indices=[0, 1, 2, 3, 4, 7],
        expected_face_indices=[0, 10, 11],
    )


def test_union_of_vertices_and_faces():
    assert_subcube(
        submesh=cube_at_origin.select()
        .pick_vertices([0, 1, 2])
        .union()
        .pick_faces([11, 10])
        .end(),
        expected_vertex_indices=[0, 1, 2, 3, 4, 7],
        expected_face_indices=[0, 1, 5, 10, 11],
    )
