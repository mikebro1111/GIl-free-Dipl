import os
from sklearn.datasets import make_regression
from core.linear_regression import LinearRegressionRANSAC
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def test_ransac():
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
    y[::10] += 50
    model_lr = LinearRegression()
    model_lr.fit(X, y)
    y_pred_lr = model_lr.predict(X)
    model_ransac = LinearRegressionRANSAC(max_iter=100, threshold=1.0)
    model_ransac.fit(X, y)
    y_pred_ransac = model_ransac.predict(X)
    plt.scatter(X, y, color='black', label='Data with outliers')
    plt.plot(X, y_pred_lr, color='blue', label='Linear Regression')
    plt.plot(X, y_pred_ransac, color='red', label='RANSAC')
    plt.legend()
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    file_path = os.path.join(results_dir, 'ransac_comparison.png')
    plt.savefig(file_path)
    plt.show()

test_ransac()