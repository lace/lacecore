from lacecore import shapes
import numpy as np
from vg.compat import v2 as vg


def test_vertex_centroid():
    cube_at_origin = shapes.cube(np.zeros(3), 3.0)
    np.testing.assert_array_almost_equal(
        cube_at_origin.vertex_centroid, np.repeat(1.5, 3)
    )


def test_bounding_box():
    bounding_box = shapes.cube(np.zeros(3), 3.0).bounding_box
    np.testing.assert_array_equal(bounding_box.origin, np.zeros(3))
    np.testing.assert_array_equal(bounding_box.size, np.repeat(3, 3))


def test_apex():
    cube_at_origin = shapes.cube(np.zeros(3), 3.0)
    np.testing.assert_array_almost_equal(
        cube_at_origin.apex(np.array([1.0, 1.0, -1.0])), np.array([3.0, 3.0, 0.0])
    )


def test_face_normals():
    cube_at_origin = shapes.cube(np.zeros(3), 3.0)
    np.testing.assert_array_equal(
        cube_at_origin.face_normals(),
        np.repeat(
            np.array(
                [
                    vg.basis.neg_y,
                    vg.basis.y,
                    vg.basis.neg_z,
                    vg.basis.x,
                    vg.basis.z,
                    vg.basis.neg_x,
                ]
            ),
            2,
            axis=0,
        ),
    )
