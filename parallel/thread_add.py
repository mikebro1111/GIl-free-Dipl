import numpy as np
from concurrent.futures import ThreadPoolExecutor

def thread_addition(arr: np.ndarray) -> np.ndarray:
    def chunked_add(chunk):
        return chunk + 1

    chunks = np.array_split(arr, 8)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(chunked_add, chunks))
    return np.concatenate(results)