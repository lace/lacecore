import vg


class AnalysisMixin:
    @property
    def vertex_centroid(self):
        """
        The centroid or geometric average of the vertices.
        """
        return vg.average(self.v)
