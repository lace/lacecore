import vg
import numpy as np
from polliwog import Plane
from ._common.reindexing import indices_of_original_elements_after_applying_mask
from ._common.validation import check_indices


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
        if not dim in [0, 1, 2]:
            raise ValueError("Expected dim to be 0, 1, or 2")
        self._keep_vertices(self._target.v[:, dim] >= point[dim])
        return self

    def vertices_above(self, point, dim):
        if not dim in [0, 1, 2]:
            raise ValueError("Expected dim to be 0, 1, or 2")
        self._keep_vertices(self._target.v[:, dim] > point[dim])
        return self

    def vertices_at_or_below(self, point, dim):
        if not dim in [0, 1, 2]:
            raise ValueError("Expected dim to be 0, 1, or 2")
        self._keep_vertices(self._target.v[:, dim] <= point[dim])
        return self

    def vertices_below(self, point, dim):
        if not dim in [0, 1, 2]:
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
        return cls(target=self._target, union_with=self._union_with + [self])

    @staticmethod
    def reconcile_masks(faces, face_mask, vertex_mask, prune_orphan_vertices):
        """
        Reconcile the vertex and face masks. When vertices are removed, their
        faces must also be removed. When faces are removed, the vertices can
        be removed or kept, depending on the `prune_orphan_vertices` option.

        Args:
            faces (np.ndarray): A `kx3` or `kx4` array of the vertices of each
                face.
            face_mask (np.ndarray): A boolean face mask.
            vertex_mask (np.ndarray): A boolean vertex mask.
            prune_orphan_vertices (bool): When `True`, remove vertices which
                their last referencing face is removed.

        Returns:
            tuple: The reconciled vertex and face masks.
        """
        num_faces, arity = vg.shape.check(locals(), "faces", (-1, -1))
        if arity not in [3, 4]:
            raise ValueError("Expected 3 or 4 vertices per face")
        vg.shape.check(locals(), "face_mask", (num_faces,))
        num_vertices = vg.shape.check(locals(), "vertex_mask", (-1,))
        if face_mask.dtype != np.bool or vertex_mask.dtype != np.bool:
            raise ValueError("Expected face_mask and vertex_mask to be boolean arrays")
        check_indices(faces, num_vertices, "faces")

        # Invalidate faces containing any vertex which is being removed.
        reconciled_face_mask = np.zeros_like(face_mask, dtype=np.bool)
        reconciled_face_mask[face_mask] = np.all(vertex_mask[faces[face_mask]], axis=1)

        # Optionally, invalidate vertices for faces which are being removed.
        if prune_orphan_vertices:
            # Orphaned verts are those belonging to faces which are being
            # removed, and not faces which are being kept.
            orphaned_vertices = np.setdiff1d(
                faces[~reconciled_face_mask], faces[reconciled_face_mask]
            )
            reconciled_vertex_mask = np.copy(vertex_mask)
            reconciled_vertex_mask[orphaned_vertices] = False
        else:
            reconciled_vertex_mask = vertex_mask

        return reconciled_face_mask, reconciled_vertex_mask

    def end(
        self,
        prune_orphan_vertices=True,
        ret_indices_of_original_faces_and_vertices=False,
    ):
        # Avoid circular import.
        from ._mesh import Mesh

        face_mask_of_union = self._face_mask
        vertex_mask_of_union = self._vertex_mask
        for other in self._union_with:
            face_mask_of_union = np.logical_or(face_mask_of_union, other._face_mask)
            vertex_mask_of_union = np.logical_or(
                vertex_mask_of_union, other._vertex_mask
            )

        reconciled_face_mask, reconciled_vertex_mask = self.reconcile_masks(
            faces=self._target.f,
            face_mask=face_mask_of_union,
            vertex_mask=vertex_mask_of_union,
            prune_orphan_vertices=prune_orphan_vertices,
        )

        new_v = self._target.v[reconciled_vertex_mask]
        indices_of_original_vertices = indices_of_original_elements_after_applying_mask(
            reconciled_vertex_mask
        )
        new_f = indices_of_original_vertices[self._target.f[reconciled_face_mask]]
        submesh = Mesh(v=new_v, f=new_f)

        if ret_indices_of_original_faces_and_vertices:
            indices_of_original_faces = indices_of_original_elements_after_applying_mask(
                reconciled_face_mask
            )
            return submesh, indices_of_original_faces, indices_of_original_vertices
        else:
            return submesh
