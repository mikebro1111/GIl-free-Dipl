from concurrent.futures import ThreadPoolExecutor
from core.clustering import kmeans
import numpy as np

def chunk_data(X, num_chunks):
    chunk_size = len(X) // num_chunks
    return [X[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

def parallel_kmeans(X, k=3, max_iters=100, n_threads=4):
    X = np.array(X)
    chunks = chunk_data(X, n_threads)
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(max_iters):
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            results = list(executor.map(lambda chunk: assign_clusters(chunk, centroids), chunks))
        
        labels = np.concatenate([r[0] for r in results])
        data_chunks = np.concatenate([r[1] for r in results])
        
        new_centroids = []
        for i in range(k):
            points = data_chunks[labels == i]
            if len(points) > 0:
                new_centroids.append(np.mean(points, axis=0))
            else:
                new_centroids.append(centroids[i])  # Якщо нема точок – залишаємо старий
        new_centroids = np.array(new_centroids)

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, labels

def assign_clusters(chunk, centroids):
    distances = np.linalg.norm(chunk[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels, chunk
