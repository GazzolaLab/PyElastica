from elasticapp._rotations import rotate, rotate_scalar
from elasticapp._PyArrays import Tensor, Matrix
from elastica._rotations import _rotate
import numpy
import time

# warm up jit for fair comparison
random_1 = numpy.random.random((3, 3, 1))
random_2 = numpy.random.random((3, 1))
out1 = _rotate(random_1, 1, random_2)


def benchmark_batchsize(funcs: list, batches: list[int], num_iterations: int = 1000):
    ret: dict = {}
    for batch_size in batches:
        random_a = numpy.random.random((3, 3, batch_size))
        random_b = numpy.random.random((3, batch_size))

        ret[batch_size] = {}
        for func_name, func, func_arg_wrap in funcs:
            tot = 0.0
            for _ in range(num_iterations):
                args = func_arg_wrap(random_a, random_b)
                start = time.perf_counter()
                func(*args)
                tot += time.perf_counter() - start

            ret[batch_size][func_name] = tot / num_iterations

    return ret


def _pyelastica_arg_wrap(x, y):
    return x, 1.0, y


def _elasticapp_arg_wrap(x, y):
    return Tensor(x), Matrix(y)


results = benchmark_batchsize(
    [
        ("pyelastica", _rotate, _pyelastica_arg_wrap),
        ("elasticapp_simd", rotate, _elasticapp_arg_wrap),
        ("elasticapp_scalar", rotate_scalar, _elasticapp_arg_wrap),
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
