import numpy as np
from polliwog import Plane
import vg
from .reconcile_selection import reconcile_selection
from .._common.reindexing import indices_of_original_elements_after_applying_mask
from .._common.validation import check_indices


class Selection:
    """
    Encapsulate a set of submesh selection operations.

    Apply the operations by invoking `run()`, which creates a new mesh.
    """

    def __init__(
        self, target, union_with=[],
    ):
        self._target = target
        self._union_with = union_with
        self._vertex_mask = np.ones(target.num_v, dtype=np.bool)
        self._face_mask = np.ones(target.num_f, dtype=np.bool)

    @staticmethod
    def _mask_like(value, num_elements):
        value = np.asarray(value)
        if value.dtype == np.bool:
            vg.shape.check(locals(), "value", (num_elements,))
            return value
        else:
            check_indices(value, num_elements, "mask")
            mask = np.zeros(num_elements, dtype=np.bool)
            mask[value] = True
            return mask

    def _keep_faces(self, mask):
        self._face_mask = np.logical_and(self._face_mask, mask)

    def _keep_vertices(self, mask):
        self._vertex_mask = np.logical_and(self._vertex_mask, mask)

    # TODO: Depends on https://github.com/metabolize/lacecore/pull/1
    # def face_groups(self, *group_names):
    #     self._keep_faces(self._target.face_groups.union(*group_names))
    #     return self

    def vertices_at_or_above(self, point, dim):
        if dim not in [0, 1, 2]:
            raise ValueError("Expected dim to be 0, 1, or 2")
        self._keep_vertices(self._target.v[:, dim] >= point[dim])
        return self

    def vertices_above(self, point, dim):
        if dim not in [0, 1, 2]:
            raise ValueError("Expected dim to be 0, 1, or 2")
        self._keep_vertices(self._target.v[:, dim] > point[dim])
        return self

    def vertices_at_or_below(self, point, dim):
        if dim not in [0, 1, 2]:
            raise ValueError("Expected dim to be 0, 1, or 2")
        self._keep_vertices(self._target.v[:, dim] <= point[dim])
        return self

    def vertices_below(self, point, dim):
        if dim not in [0, 1, 2]:
            raise ValueError("Expected dim to be 0, 1, or 2")
        self._keep_vertices(self._target.v[:, dim] < point[dim])
        return self

    def vertices_on_or_in_front_of_plane(self, plane):
        if not isinstance(plane, Plane):
            raise ValueError("Expected an instance of polliwog.Plane")
        self._keep_vertices(plane.sign(self._target.v) != -1)
        return self

    def vertices_in_front_of_plane(self, plane):
        if not isinstance(plane, Plane):
            raise ValueError("Expected an instance of polliwog.Plane")
        self._keep_vertices(plane.sign(self._target.v) == 1)
        return self

    def vertices_on_or_behind_plane(self, plane):
        if not isinstance(plane, Plane):
            raise ValueError("Expected an instance of polliwog.Plane")
        self._keep_vertices(plane.sign(self._target.v) != 1)
        return self

    def vertices_behind_plane(self, plane):
        if not isinstance(plane, Plane):
            raise ValueError("Expected an instance of polliwog.Plane")
        self._keep_vertices(plane.sign(self._target.v) == -1)
        return self

    def pick_vertices(self, indices_or_boolean_mask):
        self._keep_vertices(
            self._mask_like(indices_or_boolean_mask, len(self._vertex_mask))
        )
        return self

    def pick_faces(self, indices_or_boolean_mask):
        self._keep_faces(self._mask_like(indices_or_boolean_mask, len(self._face_mask)))
        return self

    def union(self):
        return self.__class__(target=self._target, union_with=self._union_with + [self])

    def _reconciled_selection(self, prune_orphan_vertices):
        return reconcile_selection(
            faces=self._target.f,
            face_mask=self._face_mask,
            vertex_mask=self._vertex_mask,
            prune_orphan_vertices=prune_orphan_vertices,
        )

    def end(
        self,
        prune_orphan_vertices=True,
        ret_indices_of_original_faces_and_vertices=False,
    ):
        # Avoid circular import.
        from .._mesh import Mesh

        # The approach here is designed to keep faces which have verts in two
        # halves of a union, and to avoid keeping the entire mesh when faces
        # are selected in one half of a union and verts are selected in the
        # other.

        # First, form the union of reconciled vertices.
        _, initial_vertex_mask_of_union = self._reconciled_selection(
            prune_orphan_vertices=prune_orphan_vertices
        )
        for other in self._union_with:
            _, this_vertex_mask = other._reconciled_selection(
                prune_orphan_vertices=prune_orphan_vertices
            )
            initial_vertex_mask_of_union = np.logical_or(
                initial_vertex_mask_of_union, this_vertex_mask
            )

        # Second, union the faces.
        initial_face_mask_of_union = self._face_mask
        for other in self._union_with:
            initial_face_mask_of_union = np.logical_or(
                initial_face_mask_of_union, other._face_mask
            )

        # Finally, reconcile the union of reconciled vertices with the union of
        # faces.
        face_mask_of_union, vertex_mask_of_union = reconcile_selection(
            faces=self._target.f,
            face_mask=initial_face_mask_of_union,
            vertex_mask=initial_vertex_mask_of_union,
            prune_orphan_vertices=prune_orphan_vertices,
        )

        new_v = self._target.v[vertex_mask_of_union]
        indices_of_original_vertices = indices_of_original_elements_after_applying_mask(
            vertex_mask_of_union
        )
        new_f = indices_of_original_vertices[self._target.f[face_mask_of_union]]
        submesh = Mesh(v=new_v, f=new_f)

        if ret_indices_of_original_faces_and_vertices:
            indices_of_original_faces = indices_of_original_elements_after_applying_mask(
                face_mask_of_union
            )
            return submesh, indices_of_original_faces, indices_of_original_vertices
        else:
            return submesh
