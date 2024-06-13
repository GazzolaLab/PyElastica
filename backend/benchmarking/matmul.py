from elasticapp._linalg import batch_matmul_naive, batch_matmul_blaze
from elastica._linalg import _batch_matmul
import numpy
import time

# warm up jit for fair comparison
random_1 = numpy.random.random((3, 3, 1))
random_2 = numpy.random.random((3, 3, 1))
out1 = _batch_matmul(random_1, random_2)


def benchmark_batchsize(funcs: list, batches: list[int], num_iterations: int = 1000):
    ret: dict = {}
    for batch_size in batches:
        random_a = numpy.random.random((3, 3, batch_size))
        random_b = numpy.random.random((3, 3, batch_size))

        ret[batch_size] = {}
        for func in funcs:
            start = time.perf_counter()
            for _ in range(num_iterations):
                func(random_a, random_b)

            ret[batch_size][func.__name__] = (
                time.perf_counter() - start
            ) / num_iterations

    return ret


results = benchmark_batchsize(
    [batch_matmul_naive, batch_matmul_blaze, _batch_matmul], [2**i for i in range(14)]
)
for size, data in results.items():
    pyelastica = data["_batch_matmul"]
    elasticapp = data["batch_matmul_naive"]
    elasticapp_blaze = data["batch_matmul_blaze"]
    print(f"{size = }")
    print(f"{pyelastica = }")
    print(f"{elasticapp = }, ratio: {elasticapp / pyelastica}")
    print(f"{elasticapp_blaze = }, ratio: {elasticapp_blaze / pyelastica}")
    print()
