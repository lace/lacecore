import numpy as np
import pytest
from .reconcile_selection import reconcile_selection
from .. import shapes

def test_reconcile_selection_validation():
    example_mesh = shapes.cube(np.zeros(3), 3.0)

    with pytest.raises(
        ValueError, match="Expected face_mask and vertex_mask to be boolean arrays"
    ):
        reconcile_selection(
            faces=example_mesh.f,
            face_mask=np.zeros(example_mesh.num_f),
            vertex_mask=np.zeros(example_mesh.num_v, dtype=np.bool),
            prune_orphan_vertices=False,
        )
    with pytest.raises(
        ValueError, match="Expected face_mask and vertex_mask to be boolean arrays"
    ):
        reconcile_selection(
            faces=example_mesh.f,
            face_mask=np.zeros(example_mesh.num_f, dtype=np.bool),
            vertex_mask=np.zeros(example_mesh.num_v),
            prune_orphan_vertices=False,
        )
