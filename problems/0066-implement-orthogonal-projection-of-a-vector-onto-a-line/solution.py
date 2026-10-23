
def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	import numpy as np
	v = np.array(v, dtype=float)
	L = np.array(L, dtype=float)
	vl = np.dot(v, L)
	ll = np.dot(L, L)
	return vl / ll * L