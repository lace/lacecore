from lacecore import Mesh
import numpy as np
import pytest
from .test_selection_mixin import (
    assert_subcube,
    cube_at_origin,
    cube_faces,
    cube_vertices,
)
from ..test_group_map import create_group_map


def test_prune_orphan_vertices_has_no_effect_when_selecting_vertices():
    expected_vertex_indices = [2, 3, 6, 7]
    expected_face_indices = [8, 9]
    for prune_orphan_vertices in [True, False]:
        assert_subcube(
            submesh=cube_at_origin.select()
            .vertices_at_or_above(2, np.array([1.0, 1.0, 1.0]))
            .end(prune_orphan_vertices=prune_orphan_vertices),
            expected_vertex_indices=expected_vertex_indices,
            expected_face_indices=expected_face_indices,
        )


def test_ret_indices_of_original_faces_and_vertices():
    expected_vertex_indices = [2, 3, 6, 7]
    expected_face_indices = [8, 9]
    (submesh, indices_of_original_faces, indices_of_original_vertices) = (
        cube_at_origin.select()
        .vertices_at_or_above(2, np.array([1.0, 1.0, 1.0]))
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
            method(4, np.zeros(3))

    for method in (
        selection.vertices_in_front_of_plane,
        selection.vertices_on_or_in_front_of_plane,
        selection.vertices_behind_plane,
        selection.vertices_on_or_behind_plane,
    ):
        with pytest.raises(ValueError, match="Expected an instance of polliwog.Plane"):
            method("not-a-plane")


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


def test_pick_vertices_of_face_groups():
    submesh = (
        Mesh(v=cube_vertices, f=cube_faces, face_groups=create_group_map())
        .select()
        .pick_vertices_of_face_groups("top")
        .pick_vertices_of_face_groups("left_side")
        .end(prune_orphan_vertices=False)
    )

    # We're expecting the intersection of top and left side.
    np.testing.assert_array_equal(submesh.v, cube_vertices[np.array([4, 7])])


def test_pick_face_groups_error():
    with pytest.raises(ValueError, match="Mesh has no face groups"):
        cube_at_origin.select().pick_face_groups("anything")


def test_pick_vertices_of_face_groups_error():
    with pytest.raises(ValueError, match="Mesh has no face groups"):
        cube_at_origin.select().pick_vertices_of_face_groups("anything")
