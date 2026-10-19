import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	X_T = np.transpose(X)
	X_T_X = X_T @ X
	X_T_X_inv = np.linalg.inv(X_T_X)
	theta = X_T_X_inv @ X_T @ y
	return theta