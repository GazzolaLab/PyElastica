from elasticapp._linalg_numpy import batch_cross
from elasticapp._linalg import batch_cross as batch_cross_final
from elasticapp._PyArrays import Matrix
from elastica._linalg import _batch_cross
import numpy
import time

# warm up jit for fair comparison
random_1 = numpy.random.random((3, 1))
random_2 = numpy.random.random((3, 1))
out1 = _batch_cross(random_1, random_2)


def benchmark_batchsize(funcs: list, batches: list[int], num_iterations: int = 1000):
    ret: dict = {}
    for batch_size in batches:
        random_a = numpy.random.random((3, batch_size))
        random_b = numpy.random.random((3, batch_size))

        ret[batch_size] = {}
        for func_name, func, func_wrap in funcs:
            random_a_w = func_wrap(random_a) if func_wrap else random_a
            random_b_w = func_wrap(random_b) if func_wrap else random_b

            start = time.perf_counter()
            for _ in range(num_iterations):
                func(random_a_w, random_b_w)

            ret[batch_size][func_name] = (time.perf_counter() - start) / num_iterations

    return ret


results = benchmark_batchsize(
    [
        ("pyelastica", _batch_cross, None),
        ("elasticapp_blaze_copy", batch_cross, None),
        ("elasticapp_blaze_final", batch_cross_final, Matrix),
    ],
    [2**i for i in range(14)],
)
for size, data in results.items():
    pyelastica = data["pyelastica"]
    elasticapp_blaze_copy = data["elasticapp_blaze_copy"]
    elasticapp_blaze_final = data["elasticapp_blaze_final"]
    print(f"{size = }")
    print(f"{pyelastica = }")
    print(f"{elasticapp_blaze_copy = }, ratio: {elasticapp_blaze_copy / pyelastica}")
    print(f"{elasticapp_blaze_final = }, ratio: {elasticapp_blaze_final / pyelastica}")
    print()
