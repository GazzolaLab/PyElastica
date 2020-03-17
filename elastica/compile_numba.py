from numba.pycc import CC
import numba
import numpy as np
import sys

sys.path.append("../")
cc = CC("my_module")
# Uncomment the following line to print out the compilation steps
cc.verbose = True
cc._output_dir


@cc.export("multf", "f8(f8, f8)")
@cc.export("multi", "i4(i4, i4)")
def mult(a, b):
    return a * b


@cc.export("square", "f8(f8)")
def square(a):
    return a ** 2


@cc.export("_batch_matvec", "f8[:,:](f8[:,:,:], f8[:,:])")
@numba.njit()
def _batch_matvec_numba_withloops(matrix_collection, vector_collection):
    blocksize = vector_collection.shape[1]
    output_vector = np.zeros((3, blocksize))

    for i in range(3):
        for j in range(3):
            for k in range(blocksize):
                output_vector[i, k] += (
                    matrix_collection[i, j, k] * vector_collection[j, k]
                )

    return output_vector


if __name__ == "__main__":
    cc.compile()
