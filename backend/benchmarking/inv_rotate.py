from elasticapp._rotations import inv_rotate, inv_rotate_scalar
from elasticapp._PyArrays import Tensor
from elastica._rotations import _inv_rotate
import numpy
import time

# warm up jit for fair comparison
random_1 = numpy.random.random((3, 3, 1))
out1 = _inv_rotate(random_1)


def benchmark_batchsize(funcs: list, batches: list[int], num_iterations: int = 1000):
    ret: dict = {}
    for batch_size in batches:
        random_a = numpy.random.random((3, 3, batch_size))

        ret[batch_size] = {}
        for func_name, func, func_wrap in funcs:
            random_a_w = func_wrap(random_a) if func_wrap else random_a

            start = time.perf_counter()
            for _ in range(num_iterations):
                func(random_a_w)

            ret[batch_size][func_name] = (time.perf_counter() - start) / num_iterations

    return ret


results = benchmark_batchsize(
    [
        ("pyelastica", _inv_rotate, None),
        ("elasticapp_simd", inv_rotate, Tensor),
        ("elasticapp_scalar", inv_rotate_scalar, Tensor),
    ],
    [2**i for i in range(14)],
)
for size, data in results.items():
    pyelastica = data["pyelastica"]
    elasticapp_simd = data["elasticapp_simd"]
    elasticapp_scalar = data["elasticapp_scalar"]
    print(f"{size = }")
    print(f"{pyelastica = }")
    print(f"{elasticapp_simd = }, ratio: {elasticapp_simd / pyelastica}")
    print(f"{elasticapp_scalar = }, ratio: {elasticapp_scalar / pyelastica}")
    print()
