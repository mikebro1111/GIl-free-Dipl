import time
import numpy as np
from numba import vectorize, float64

@vectorize([float64(float64, float64)], target='parallel')
def simd_add(a, b):
    return a + b

def simd_benchmark(X, y):
    start = time.time()
    result = simd_add(X, y)
    end = time.time()
    return end - start


def benchmark_simd_operations(size):
    X = np.random.rand(size)
    y = np.random.rand(size)

    start = time.time()
    result = simd_add(X, y)
    end = time.time()
    return end - start


if __name__ == "__main__":
    results = benchmark_simd_operations()

    import matplotlib.pyplot as plt
    import os

    os.makedirs("results", exist_ok=True)

    sizes = list(results.keys())
    times = [results[s] for s in sizes]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, marker='o', label="SIMD Add")
    plt.title("SIMD Add Performance")
    plt.xlabel("Data Size")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/simd_add_benchmark.png")
    plt.show()