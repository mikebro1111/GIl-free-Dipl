import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LinearRegression
from numba import jit

def _partial_fit(args):
    X_chunk, y_chunk = args
    X_b = np.c_[np.ones((X_chunk.shape[0], 1)), X_chunk]
    return X_b.T @ X_b, X_b.T @ y_chunk

class LinearRegressionNP:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta_best = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        self.intercept_ = theta_best[0]
        self.coef_ = theta_best[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_

class LinearRegressionMP:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_cores = cpu_count()
        chunk_size = int(np.ceil(X.shape[0] / n_cores))
        chunks = [(X[i:i+chunk_size], y[i:i+chunk_size]) for i in range(0, X.shape[0], chunk_size)]

        with Pool(n_cores) as pool:
            results = pool.map(_partial_fit, chunks)

        A = sum(res[0] for res in results)
        b = sum(res[1] for res in results)
        theta_best = np.linalg.pinv(A) @ b

        self.intercept_ = theta_best[0]
        self.coef_ = theta_best[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_

class LinearRegressionThread:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_threads = min(32, (len(X) // 1000) + 1)
        chunk_size = int(np.ceil(X.shape[0] / n_threads))
        chunks = [(X[i:i+chunk_size], y[i:i+chunk_size]) for i in range(0, X.shape[0], chunk_size)]

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            results = list(executor.map(_partial_fit, chunks))

        A = sum(res[0] for res in results)
        b = sum(res[1] for res in results)
        theta_best = np.linalg.pinv(A) @ b

        self.intercept_ = theta_best[0]
        self.coef_ = theta_best[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_
    
@jit(nopython=True)
def _numba_partial_fit(X_chunk, y_chunk):
    X_b = np.ones((X_chunk.shape[0], X_chunk.shape[1] + 1))
    X_b[:, 1:] = X_chunk
    return X_b.T @ X_b, X_b.T @ y_chunk

class LinearRegressionNumba:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_cores = cpu_count()
        chunk_size = int(np.ceil(X.shape[0] / n_cores))
        chunks = [(X[i:i+chunk_size], y[i:i+chunk_size]) for i in range(0, X.shape[0], chunk_size)]

        results = []
        for chunk in chunks:
            results.append(_numba_partial_fit(*chunk))

        A = sum(res[0] for res in results)
        b = sum(res[1] for res in results)
        theta_best = np.linalg.pinv(A) @ b

        self.intercept_ = theta_best[0]
        self.coef_ = theta_best[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_

from sklearn.linear_model import LinearRegression as SklearnLR

class LinearRegressionSklearn:
    def __init__(self):
        self.model = SklearnLR()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
class LinearRegressionRANSAC:
    def __init__(self, max_iter=100, threshold=1.0):
        self.max_iter = max_iter
        self.threshold = threshold

        self.coef_ = None
        self.intercept_ = None
        self.inliers_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        best_inliers_count = 0
        best_model = None
        best_inliers = None

        for _ in range(self.max_iter):
            sample_indices = np.random.choice(X.shape[0], 2, replace=False)
            X_sample, y_sample = X[sample_indices], y[sample_indices]
            model = LinearRegression()
            model.fit(X_sample, y_sample)
            y_pred = model.predict(X)
            residuals = np.abs(y - y_pred)
            inliers = residuals < self.threshold

            inliers_count = np.sum(inliers)

            if inliers_count > best_inliers_count:
                best_inliers_count = inliers_count
                best_model = model
                best_inliers = inliers
        self.coef_ = best_model.coef_
        self.intercept_ = best_model.intercept_
        self.inliers_ = best_inliers

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_.T + self.intercept_

    def get_inliers(self):
        return self.inliers_