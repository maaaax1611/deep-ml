import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float).copy()
    m = len(y)

    if method == 'batch':
        bs = m
    elif method == 'stochastic':
        bs = 1
    elif method == 'mini_batch':
        bs = batch_size
    else:
        raise ValueError(f"Unknown method: {method}")

    for _ in range(n_epochs):
        for start in range(0, m, bs):
            Xb = X[start:start+bs]
            yb = y[start:start+bs]
            grad = 2 * Xb.T @ (Xb @ w - yb) / bs
            w -= learning_rate * grad
    return w      
