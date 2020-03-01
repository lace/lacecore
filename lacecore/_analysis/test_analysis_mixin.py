import numpy as np
from .. import shapes


def test_vertex_centroid():
    cube_at_origin = shapes.cube(np.zeros(3), 3.0)
    np.testing.assert_array_almost_equal(
        cube_at_origin.vertex_centroid, np.repeat(1.5, 3)
    )
