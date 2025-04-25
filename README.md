# Simple Operations Performance Benchmarking

This project benchmarks the performance of simple mathematical operations (+, -, *, /, %, etc) on large datasets using different approaches:
- Python loops
- NumPy
- Multiprocessing
- Multithreading (CPython 3.13+)
- Numba (JIT Compilation)

## Setup

1. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```

2. Run the benchmark tests:
    ```
    python main.py
    ```

## Test Results

- The benchmark tests will compare the performance of different approaches for various data sizes (up to filling your RAM).
- The results will include the execution time for each method.

## Future Work

This project will be extended with implementations of Linear Regression, RANSAC, video processing, and more.
