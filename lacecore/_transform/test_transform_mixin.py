import numpy as np
import pytest
import vg
from .._group_map import GroupMap
from .._mesh import Mesh

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


def test_append_transform():
    scale = np.array([[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]])
    transformed = cube_at_origin.transform().append_transform(scale).end()
    assert transformed.f is cube_at_origin.f
    np.testing.assert_array_equal(transformed.v, 3.0 * cube_at_origin.v)


def test_uniformly_scaled():
    transformed = cube_at_origin.uniformly_scaled(3.0)
    assert transformed.f is cube_at_origin.f
    np.testing.assert_array_equal(transformed.v, 3.0 * cube_at_origin.v)


def test_non_uniformly_scaled():
    transformed = cube_at_origin.non_uniformly_scaled(1.0, 2.0, 3.0)
    assert transformed.f is cube_at_origin.f
    np.testing.assert_array_equal(
        transformed.v, np.array([[1.0, 2.0, 3.0]]) * cube_at_origin.v
    )


def test_units_converted():
    transformed = cube_at_origin.units_converted("m", "cm")
    assert transformed.f is cube_at_origin.f
    np.testing.assert_array_equal(transformed.v, 100.0 * cube_at_origin.v)


def test_translated():
    transformed = cube_at_origin.translated(3.0 * vg.basis.z)
    assert transformed.f is cube_at_origin.f
    np.testing.assert_array_equal(transformed.v, 3.0 * vg.basis.z + cube_at_origin.v)


def test_reoriented():
    transformed = cube_at_origin.reoriented(vg.basis.z, vg.basis.y)
    assert transformed.f is cube_at_origin.f
    expected_v = np.copy(cube_at_origin.v)
    expected_v[:, 0] = -expected_v[:, 0]
    expected_v[:, 1], expected_v[:, 2] = expected_v[:, 2], expected_v[:, 1].copy()
    np.testing.assert_array_equal(transformed.v, expected_v)


def test_rotated():
    quarter_turn_around_y = np.array([0, np.pi / 2, 0])
    transformed = cube_at_origin.rotated(quarter_turn_around_y)
    assert transformed.f is cube_at_origin.f
    np.testing.assert_array_almost_equal(
        transformed.v,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -3.0],
                [3.0, 0.0, -3.0],
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 3.0, -3.0],
                [3.0, 3.0, -3.0],
                [3.0, 3.0, 0.0],
            ]
        ),
    )


def test_flip_faces():
    transformed = cube_at_origin.faces_flipped()
    np.testing.assert_array_equal(
        transformed.f,
        np.hstack(
            [
                cube_at_origin.f[:, 2].reshape(-1, 1),
                cube_at_origin.f[:, 1].reshape(-1, 1),
                cube_at_origin.f[:, 0].reshape(-1, 1),
            ]
        ),
    )
    np.testing.assert_array_equal(transformed.v, cube_at_origin.v)


def test_flip():
    transformed = cube_at_origin.flipped(2)
    np.testing.assert_array_equal(
        transformed.f,
        np.hstack(
            [
                cube_at_origin.f[:, 2].reshape(-1, 1),
                cube_at_origin.f[:, 1].reshape(-1, 1),
                cube_at_origin.f[:, 0].reshape(-1, 1),
            ]
        ),
    )
    np.testing.assert_array_equal(
        transformed.v,
        np.hstack([cube_at_origin.v[:, 0:2], -cube_at_origin.v[:, 2].reshape(-1, 1)]),
    )


def test_flip_preserve_vertex_centroid():
    transformed = cube_at_origin.flipped(2, preserve_vertex_centroid=True)
    np.testing.assert_array_equal(
        transformed.f,
        np.hstack(
            [
                cube_at_origin.f[:, 2].reshape(-1, 1),
                cube_at_origin.f[:, 1].reshape(-1, 1),
                cube_at_origin.f[:, 0].reshape(-1, 1),
            ]
        ),
    )
    np.testing.assert_array_equal(
        transformed.v,
        np.hstack(
            [cube_at_origin.v[:, 0:2], 3.0 - cube_at_origin.v[:, 2].reshape(-1, 1)]
        ),
    )
    np.testing.assert_array_equal(np.unique(cube_at_origin.v), np.unique(transformed.v))


def test_flip_error():
    with pytest.raises(ValueError, match="Expected dim to be 0, 1, or 2"):
        cube_at_origin.flipped(-1)


def test_transform_preserves_face_groups():
    face_group_dict = {"bottom": [0, 1]}
    result = Mesh(
        v=cube_vertices,
        f=cube_faces,
        face_groups=GroupMap.from_dict(face_group_dict, len(cube_faces)),
    ).uniformly_scaled(3.0)
    np.testing.assert_array_equal(
        result.face_groups["bottom"].nonzero()[0], face_group_dict["bottom"]
    )
