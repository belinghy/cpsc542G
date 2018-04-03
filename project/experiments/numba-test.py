import numpy as np
import time

from numba import cuda, jit


@jit(parallel=True)
def VectorAdd(a, b):
    for i in range(100):
        c = a + b
    return a + b


def NumpyAdd(a, b):
    for i in range(100):
        c = a + b
    return a + b


def main():
    N = 32000000
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    start = time.time()
    D = NumpyAdd(A, B)
    numpy_add_time = time.time() - start

    print("D[:5] = {}".format(D[:5]))
    print("D[-5:] = {}".format(D[-5:]))

    print("NumpyAdd took {} seconds".format(numpy_add_time))

    start = time.time()
    C = VectorAdd(A, B)
    vector_add_time = time.time() - start

    print("C[:5] = {}".format(C[:5]))
    print("C[-5:] = {}".format(C[-5:]))

    print("VectorAdd took {} seconds".format(vector_add_time))

    print(cuda.gpus)


if __name__ == '__main__':
    main()
