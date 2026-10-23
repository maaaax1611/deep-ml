import numpy as np

def lu_decomposition(A: list) -> tuple:
	"""
	Perform LU decomposition on a square matrix using Doolittle's method.
	
	Args:
		A: Square matrix as a list of lists
	
	Returns:
		tuple: (L, U) where L is lower triangular with 1s on diagonal,
		       U is upper triangular, and A = L @ U
	"""
	# Your code here
	U = np.array(A, dtype=float)
    n = len(U)
    L = np.eye(n)
    for k in range(n - 1):
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]       # Eliminationsfaktor merken
            U[i, k:] -= L[i, k] * U[k, k:]    # Zeile eliminieren
    return L, U