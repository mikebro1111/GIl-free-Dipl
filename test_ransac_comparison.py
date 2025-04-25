import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from core.linear_regression import LinearRegressionRANSAC

def test_ransac_comparison():
    X, y = make_regression(n_samples=100, n_features=1, noise=10)
    y[::10] += 100

    model_lr = LinearRegression()
    model_lr.fit(X, y)
    y_pred_lr = model_lr.predict(X)

    model_ransac = LinearRegressionRANSAC(max_iter=100, threshold=10.0)
    model_ransac.fit(X, y)
    y_pred_ransac = model_ransac.predict(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='black', label='Outliers data')
    plt.plot(X, y_pred_lr, color='blue', label='Linear Regression')
    plt.plot(X, y_pred_ransac, color='red', label='RANSAC')
    plt.legend()
    plt.title("Linear Regression vs RANSAC")
    plt.tight_layout()

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, 'ransac_linear.png')
    plt.savefig(file_path)
    plt.close()

if __name__ == "__main__":
    test_ransac_comparison()
