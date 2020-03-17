__doc__ = """ Restart class dumps and loads rod data """
import numpy as np

# import functools
import pickle

# from ._rod import CosseratRod
# from ._linalg import _batch_matmul, _batch_matvec, _batch_cross
# from ._calculus import quadrature_kernel, difference_kernel
# from ._rotations import _inv_rotate
# from .utils import MaxDimension, Tolerance


class Restart:
    def __init__(self):
        super(Restart, self).__init__()

    def dump_data(self, dump_object, filename):
        assert isinstance(filename, str), "filename type is not string"
        file = open(filename, "wb")
        pickle.dump(dump_object, file)
        file.close()

    def load_data(self, filename):
        assert isinstance(filename, str), "filename type is not string"
        file = open(filename, "rb")
        load_object = pickle.load(file)
        file.close()
        return load_object
