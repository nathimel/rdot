import numpy as np
from .information import DKL

def expected_distortion(
    px: np.ndarray, qxhat_x: np.ndarray, dist_mat: np.ndarray
) -> float:
    """Compute the expected distortion $E[D[X, \\hat{X}]]$ of a joint distribution defind by $P(X)$ and $P(\\hat{X}|X)$, where
    
    $D[X, \hat{X}] = \sum_x p(x) \sum_{\\hat{x}} p(\\hat{x}|x) \\cdot d(x, \\hat{x})$
    
    Args:
        px: array of shape `|X|` the prior probability of an input symbol (i.e., the source)    

        qxhat_x: array of shape `(|X|, |X_hat|)` the probability of an output symbol given the input       

        dist_mat: array of shape `(|X|, |X_hat|)` representing the distoriton matrix between the input alphabet and the reconstruction alphabet.    
    """
    return np.sum(np.diag(px) @ (qxhat_x * dist_mat))

# Pairwise distortion measures

def hamming(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 1 - np.eye(len(x),len(y))

def quadratic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([[(xi - yi)**2 for xi in x] for yi in y]) # TODO: vectorize

def ib_kl(py_x: np.ndarray, qy_xhat: np.ndarray) -> np.ndarray:
    # D[p(y|x) || q(y|xhat)]
    return np.array([[DKL(x, xhat) for x in py_x] for xhat in qy_xhat])