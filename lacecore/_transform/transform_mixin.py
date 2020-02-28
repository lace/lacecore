from .transform_object import Transform


class TransformMixin:
    def transform(self):
        """
        Begin a composite transform operation. After invoking `.transform()`,
        apply transformations, then invoke `.end()` to create a mesh with
        transformed vertices.

        Does not mutate the callee.

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
        return Transform(self)

    def uniformly_scaled(self, factor):
        """
        Scale by the given factor.

        Args:
            factor (float): The scale factor.

        Returns:
            lacecore.Mesh: A mesh with transformed vertices.
        """
        return self.transform().uniform_scale(factor=factor).end()

    def units_converted(self, from_units, to_units):
        """
        Convert the mesh from one set of units to another.

        Support the length units from Ounce:
        https://github.com/lace/ounce/blob/master/ounce/core.py#L26

        Returns:
            lacecore.Mesh: A mesh with transformed vertices.
        """
        return (
            self.transform()
            .convert_units(from_units=from_units, to_units=to_units)
            .end()
        )

    def non_uniformly_scaled(self, x_factor, y_factor, z_factor):
        """
        Scale along each axis by the given factors.

        Args:
            x_factor (flot): The scale factor along the `x` axis.
            y_factor (flot): The scale factor along the `y` axis.
            z_factor (flot): The scale factor along the `z` axis.

        Returns:
            lacecore.Mesh: A mesh with transformed vertices.
        """
        return self.transform().non_uniform_scale(x_factor, y_factor, z_factor).end()

    def faces_flipped(self):
        """
        Flip the orientation of the faces.

        Returns:
            lacecore.Mesh: A mesh with transformed faces.
        """
        return self.transform().flip_faces().end()

    def flipped(self, dim, preserve_vertex_centroid=False):
        """
        Flip about the given axis.

        Args:
            dim (int): The axis to flip around: 0 for `x`, 1 for `y`, 2 for `z`.
            preserve_vertex_centroid (bool): When `True`, translate after
                flipping to preserve the original vertex centroid.

        Returns:
            lacecore.Mesh: A mesh with transformed vertices.
        """
        return (
            self.transform()
            .flip(dim, preserve_vertex_centroid=preserve_vertex_centroid)
            .end()
        )

    def translated(self, translation):
        """
        Translate by the vector provided.

        Args:
            vector (np.arraylike): A 3x1 vector.

        Returns:
            lacecore.Mesh: A mesh with transformed vertices.
        """
        return self.transform().translate(translation=translation).end()

    def reoriented(self, up, look):
        """
        Reorient using up and look.

        Returns:
            lacecore.Mesh: A mesh with transformed vertices.
        """
        return self.transform().reorient(up=up, look=look).end()

    def rotated(self, rotation):
        """
        Rotate by the given 3x3 rotation matrix or a Rodrigues vector.

        Returns:
            lacecore.Mesh: A mesh with transformed vertices.
        """
        return self.transform().rotate(rotation=rotation).end()
