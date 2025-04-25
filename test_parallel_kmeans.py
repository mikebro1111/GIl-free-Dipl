import unittest
import numpy as np
from sklearn.datasets import make_blobs
from core.clustering import kmeans, initialize_centroids, assign_clusters, update_centroids, has_converged

class TestKMeans(unittest.TestCase):
    
    def setUp(self):
        self.X, self.y = make_blobs(n_samples=100, centers=3, random_state=42)
        self.k = 3
        print("Test Data Created with 3 clusters.")
    
    def test_kmeans(self):
        print("Running KMeans...")
        centroids, labels = kmeans(self.X, self.k)
        self.assertEqual(len(np.unique(labels)), self.k, f"Expected {self.k} clusters but got {len(np.unique(labels))}")
        self.assertEqual(centroids.shape[0], self.k, f"Centroids count mismatch. Expected {self.k} but got {centroids.shape[0]}")
        self.assertEqual(len(labels), len(self.X), "Mismatch between data points and labels count")
        print("KMeans Test Passed.")
    
    def test_centroid_initialization(self):
        print("Testing Centroid Initialization...")
        centroids = initialize_centroids(self.X, self.k)
        self.assertEqual(centroids.shape[0], self.k, f"Centroids count mismatch. Expected {self.k} but got {centroids.shape[0]}")
        print("Centroid Initialization Test Passed.")
        
    def test_assign_clusters(self):
        print("Testing Cluster Assignment...")
        centroids = np.array([[0, 0], [1, 1], [2, 2]])
        labels = assign_clusters(self.X, centroids)
        
        self.assertEqual(len(labels), len(self.X), "Mismatch between data points and labels count")
        self.assertTrue(np.all(labels >= 0) and np.all(labels < self.k), "Cluster labels are out of bounds")
        print("Cluster Assignment Test Passed.")

    def test_update_centroids(self):
        print("Testing Centroid Update...")
        labels = np.random.randint(0, self.k, len(self.X))
        new_centroids = update_centroids(self.X, labels, self.k)
        
        self.assertEqual(new_centroids.shape[0], self.k, f"Centroids count mismatch. Expected {self.k} but got {new_centroids.shape[0]}")
        print("Centroid Update Test Passed.")
        
    def test_convergence(self):
        print("Testing Convergence...")
        old_centroids = np.random.random((self.k, 2))
        new_centroids = old_centroids + 1e-5
        self.assertTrue(has_converged(old_centroids, new_centroids), "Centroids should have converged")
        print("Convergence Test Passed.")

if __name__ == "__main__":
    unittest.main()