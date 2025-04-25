import time
import numpy as np
import matplotlib.pyplot as plt
from core.linear_regression import (
    LinearRegressionNP,
    LinearRegressionMP,
    LinearRegressionThread,
    LinearRegressionNumba,
    LinearRegressionSklearn
)
from simd.simd_benchmarks import benchmark_simd_operations
def benchmark(model_class, X, y):
    model = model_class()
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    return end_time - start_time


def benchmark_operations():
    data_sizes = [1_000, 10_000, 100_000, 1_000_000]
    models = {
        "NumPy": LinearRegressionNP,
        "Multiprocessing (GIL-free)": LinearRegressionMP,  # GIL-free
        "Threading (GIL)": LinearRegressionThread,  # З GIL
        "Numba": LinearRegressionNumba,
        "Sklearn": LinearRegressionSklearn,
    }

    results = {}

    for size in data_sizes:
        print(f"\nData size: {size}")
        X = np.random.rand(size, 10)
        y = np.random.rand(size)

        model_results = {}
        for name, model_class in models.items():
            try:
                elapsed = benchmark(model_class, X, y)
                model_results[name] = elapsed
                print(f"{name:<15}: {elapsed:.4f} sec")
            except Exception as e:
                print(f"{name:<15}: Failed with {e}")
                model_results[name] = None
        simd_time = benchmark_simd_operations(size)
        model_results["SIMD"] = simd_time
        print(f"SIMD            : {simd_time:.4f} sec")

        results[size] = model_results

    return results

def plot_benchmark_results(results):
    sizes = list(results.keys())
    model_times = {model: [results[size].get(model, None) for size in sizes] for model in results[sizes[0]].keys()}
    plt.figure(figsize=(10, 6))
    for model, times in model_times.items():
        plt.plot(sizes, times, marker='o', label=model)
    plt.title("Performance Comparison")
    plt.xlabel("Data Size")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/benchmark_comparison.png")
    plt.show()

if __name__ == "__main__":
    results = benchmark_operations()
    for size, times in results.items():
        print(f"\nData size: {size}")
        for k, v in times.items():
            if v is not None:
                print(f"  {k.capitalize():<12}: {v:.6f} s")
            else:
                print(f"  {k.capitalize():<12}: Failed")
    plot_benchmark_results(results)
