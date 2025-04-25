import numpy as np
from multiprocessing import Pool

def add_one(x: float) -> float:
    return x + 1

def parallel_addition(arr: np.ndarray) -> np.ndarray:
    with Pool() as pool:
        result = pool.map(add_one, arr.tolist())
    return np.array(result)