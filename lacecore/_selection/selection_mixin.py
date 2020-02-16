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

    def keeping_vertices_at_or_above(self, point, dim):
        return self.select().vertices_at_or_above(point=point, dim=dim).end()

    def keeping_vertices_above(self, point, dim):
        return self.select().vertices_above(point=point, dim=dim).end()

    def keeping_vertices_at_or_below(self, point, dim):
        return self.select().vertices_at_or_below(point=point, dim=dim).end()

    def keeping_vertices_below(self, point, dim):
        return self.select().vertices_below(point=point, dim=dim).end()

    def keeping_vertices_on_or_in_front_of_plane(self, plane):
        return self.select().vertices_on_or_in_front_of_plane(plane=plane).end()

    def keeping_vertices_in_front_of_plane(self, plane):
        return self.select().vertices_in_front_of_plane(plane=plane).end()

    def keeping_vertices_on_or_behind_plane(self, plane):
        return self.select().vertices_on_or_behind_plane(plane=plane).end()

    def keeping_vertices_behind_plane(self, plane):
        return self.select().vertices_behind_plane(plane=plane).end()

    def picking_vertices(self, indices_or_boolean_mask):
        return (
            self.select()
            .pick_vertices(indices_or_boolean_mask=indices_or_boolean_mask)
            .end()
        )

    def picking_faces(self, indices_or_boolean_mask):
        return (
            self.select()
            .pick_faces(indices_or_boolean_mask=indices_or_boolean_mask)
            .end()
        )
