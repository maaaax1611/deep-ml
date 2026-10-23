import torch

def qr_decomposition(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform QR decomposition using Gram-Schmidt process.
    
    Args:
        A: An m x n matrix as torch.Tensor
    
    Returns:
        Tuple of (Q, R) where Q is orthogonal and R is upper triangular
    """
    A = torch.as_tensor(A).float()
    m, n = A.shape
    Q = torch.zeros(m, n)
    R = torch.zeros(n, n)
    for j in range(n):
        v = A[:, j].clone()
        for i in range(j):
            R[i, j] = Q[:, i] @ v          # Projektionslänge auf q_i
            v = v - R[i, j] * Q[:, i]      # Projektion abziehen
        R[j, j] = torch.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R