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
        "Multiprocessing": LinearRegressionMP,
        "Threading": LinearRegressionThread,
        "Numba": LinearRegressionNumba,
        "Sklearn": LinearRegressionSklearn,
    }

    results = {}

    for size in data_sizes:
        print(f"\n Data size: {size}")
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

        results[size] = model_results

    return results

def plot_results(results):
    data_sizes = list(results.keys())
    models = list(next(iter(results.values())).keys())

    for model in models:
        times = [results[size].get(model, None) for size in data_sizes]
        plt.plot(data_sizes, times, label=model)

    plt.xlabel("Data Size")
    plt.ylabel("Time (seconds)")
    plt.title("Comparison of Linear Regression Time Across Different Implementations")
    plt.legend()
    plt.savefig("results/linear_regression_comparison.png")
    plt.show()
    
if __name__ == "__main__":
    results = benchmark_operations()
    for size, times in results.items():
        print(f"\nData size: {size}")
        for k, v in times.items():
            print(f"  {k:<12}: {v:.6f} s")