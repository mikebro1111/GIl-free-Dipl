import numpy as np
import time
from benchmarks.video import benchmark_video_processing
from core.linear_regression import (
    LinearRegressionNP,
    LinearRegressionMP,
    LinearRegressionThread,
    LinearRegressionNumba,
    LinearRegressionSklearn
)

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
    video_file = 'path_to_your_video.mp4'
    def benchmark(model_class, X, y):
        model = model_class()
        start_time = time.time()
        model.fit(X, y)
        end_time = time.time()
        return end_time - start_time
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
        video_time = benchmark_video_processing(video_file)
        model_results["Video Processing"] = video_time
        print(f"Video Processing : {video_time:.4f} sec")

        results[size] = model_results

    return results
