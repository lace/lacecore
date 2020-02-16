class SelectionMixin:
    def select(self):
        """
        Begin a chained selection operation. After invoking `.select()`,
        apply selection criteria, then invoke `.end()` to create a submesh.

        Include `.union()` in the chain to combine multiple sets of
        selection criteria into a single submesh.

        Returns:
            lacecore.Selection: The selection operation.

        Example:
            >>> centroid = np.average(mesh.v, axis=0)
            >>> upper_right_quadrant = (
                mesh.select()
                .vertices_above(centroid, dim=0)
                .vertices_above(centroid, dim=1)
                .end()
            )
            >>> upper_half_plus_right_half = (
                mesh.select()
                .vertices_above(centroid, dim=0)
                .union()
                .vertices_above(centroid, dim=1)
                .end()
            )
        """
        # Avoid circular import in .._mesh.
        from .selection_object import Selection

        return Selection(target=self)

    def keeping_vertices_at_or_above(self, dim, point):
        """
        Select vertices which, when projected to the given axis, are either
        coincident with the projection of the given point, or lie further
        along the axis.

        Args:
            dim (int): The axis of interest: 0 for `x`, 1 for `y`, 2 for `z`.
            point (np.arraylike): The point of interest.

        Returns:
            lacecore.Mesh: A submesh containing the selection.
        """
        return self.select().vertices_at_or_above(dim=dim, point=point).end()

    def keeping_vertices_above(self, dim, point):
        """
        Select vertices which, when projected to the given axis, lie after
        the projection of the given point.

        Args:
            dim (int): The axis of interest: 0 for `x`, 1 for `y`, 2 for `z`.
            point (np.arraylike): The point of interest.

        Returns:
            lacecore.Mesh: A submesh containing the selection.
        """
        return self.select().vertices_above(dim=dim, point=point).end()

    def keeping_vertices_at_or_below(self, dim, point):
        """
        Select vertices which, when projected to the given axis, are either
        coincident with the projection of the given point, or lie before it.

        Args:
            dim (int): The axis of interest: 0 for `x`, 1 for `y`, 2 for `z`.
            point (np.arraylike): The point of interest.

        Returns:
            lacecore.Mesh: A submesh containing the selection.
        """
        return self.select().vertices_at_or_below(dim=dim, point=point).end()

    def keeping_vertices_below(self, dim, point):
        """
        Select vertices which, when projected to the given axis, lie before
        the projection fo the given point.

        Args:
            dim (int): The axis of interest: 0 for `x`, 1 for `y`, 2 for `z`.
            point (np.arraylike): The point of interest.

        Returns:
            lacecore.Mesh: A submesh containing the selection.
        """
        return self.select().vertices_below(dim=dim, point=point).end()

    def keeping_vertices_on_or_in_front_of_plane(self, plane):
        """
        Select the vertices which are either on or in front of the given
        plane.

        Args:
            plane (polliwog.Plane): The plane of interest.

        Returns:
            lacecore.Mesh: A submesh containing the selection.
        """
        return self.select().vertices_on_or_in_front_of_plane(plane=plane).end()

    def keeping_vertices_in_front_of_plane(self, plane):
        """
        Select the vertices which are in front of the given plane.

        Args:
            plane (polliwog.Plane): The plane of interest.

        Returns:
            lacecore.Mesh: A submesh containing the selection.
        """
        return self.select().vertices_in_front_of_plane(plane=plane).end()

    def keeping_vertices_on_or_behind_plane(self, plane):
        """
        Select the vertices which are either on or behind the given plane.

        Args:
            plane (polliwog.Plane): The plane of interest.

        Returns:
            lacecore.Mesh: A submesh containing the selection.
        """
        return self.select().vertices_on_or_behind_plane(plane=plane).end()

    def keeping_vertices_behind_plane(self, plane):
        """
        Select the vertices which are behind the given plane.

        Args:
            plane (polliwog.Plane): The plane of interest.

        Returns:
            lacecore.Mesh: A submesh containing the selection.
        """
        return self.select().vertices_behind_plane(plane=plane).end()

    def picking_vertices(self, indices_or_boolean_mask):
        """
        Select only the given vertices.

        Args:
            indices_or_boolean_mask (np.arraylike): Either a list of vertex
                indices, or a boolean mask the same length as the vertex array.

        Returns:
            lacecore.Mesh: A submesh containing the selection.
        """
        return (
            self.select()
            .pick_vertices(indices_or_boolean_mask=indices_or_boolean_mask)
            .end()
        )

    def picking_faces(self, indices_or_boolean_mask):
        """
        Select only the given faces.

        Args:
            indices_or_boolean_mask (np.arraylike): Either a list of vertex
                indices, or a boolean mask the same length as the vertex array.

        Returns:
            lacecore.Mesh: A submesh containing the selection.
        """
        return (
            self.select()
            .pick_faces(indices_or_boolean_mask=indices_or_boolean_mask)
            .end()
        )
