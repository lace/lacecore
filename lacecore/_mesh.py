import vg
from ._common.validation import check_arity, check_indices
from ._selection import Selection


class Mesh:
    def __init__(self, v, f, copy_v=False, copy_f=False, segm=None):
        num_vertices = vg.shape.check(locals(), "v", (-1, 3))
        vg.shape.check(locals(), "f", (-1, -1))
        check_arity(f)
        check_indices(f, num_vertices, "f")

        # TODO: Needs coverage.
        # if copy_f:
        #     f = np.copy(f)
        # if copy_v:
        #     v = np.copy(v)
        f.setflags(write=False)
        v.setflags(write=False)
        self.f = f
        self.v = v
        self.segm = segm

    # TODO: Needs coverage.
    # @classmethod
    # def from_lace(cls, mesh):
    #     return cls(v=mesh.v, f=mesh.f)

    # TODO: Needs coverage.
    # @classmethod
    # def from_trimesh(cls, mesh):
    #     return cls(v=mesh.vertices, f=mesh.faces)

    def __repr__(self):
        return f"lacecore.Mesh(num_v={self.num_v}, num_f={self.num_f})"

    @property
    def num_v(self):
        return len(self.v)

    @property
    def num_f(self):
        return len(self.f)

    def select(self):
        return Selection(target=self)
