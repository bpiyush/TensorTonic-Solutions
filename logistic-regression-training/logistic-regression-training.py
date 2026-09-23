import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    
    # Define the trainable parameters
    n, d = X.shape
    # w = np.random.normal(loc=0, scale=1, size=(d))
    w = np.zeros(d)
    b = 0

    def forward(w, b):
        """Runs the forward pass (compute loss) given current set of weights."""
        p = _sigmoid(X @ w + b)

        # Compute the BCE loss
        loss = 0.
        for i in range(d):
            loss += y[i] * np.log(p[i]) + (1 - y[i]) * np.log(1 - p[i])
        loss *= (-1. / n)

        return p, loss

    def gradients(X, y, p):
        # dl_db = 0.
        # dl_dw = np.zeros(d)
        # for i in range(d):
        #     term = (1 - p[i]) * y[i] - p[i] * (1 - y[i])
        #     dl_db += term
        #     dl_dw += term * X[i]
        # dl_dw *= (-1. / n)
        # dl_db *= (-1. / n)
        # return dl_dw, dl_db
        dl_dw = X.T @ (p - y) / len(y)
        dl_db = np.mean(p - y)
        return dl_dw, dl_db

    # Train loop
    for s in range(steps):

        # Forward
        p, loss = forward(w, b)

        # Backward step: compute gradients dL/dw, dL/db
        dl_dw, dl_db = gradients(X, y, p)

        # Update
        w = w - lr * dl_dw
        b = b - lr * dl_db

    return w, b