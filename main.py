import matplotlib.pyplot as plt
from benchmarks.benchmark_runner import benchmark_operations

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
            if v is not None:
                print(f"  {k.capitalize():<12}: {v:.6f} s")
            else:
                print(f"  {k.capitalize():<12}: Failed")

    plot_results(results)
