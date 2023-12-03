import numpy as np

def expected_distortion(
    px: np.ndarray, pxhat_x: np.ndarray, dist_mat: np.ndarray
) -> float:
    """Compute the expected distortion $E[D[X, \\hat{X}]]$ of a joint distribution defind by $P(X)$ and $P(\\hat{X}|X)$, where
    
    $D[X, \hat{X}] = \sum_x p(x) \sum_{\\hat{x}} p(\\hat{x}|x) \\cdot d(x, \\hat{x})$
    
    Args:
        px: array of shape `|X|` the prior probability of an input symbol (i.e., the source)    

        pxhat_x: array of shape `(|X|, |X_hat|)` the probability of an output symbol given the input       

        dist_mat: array of shape `(|X|, |X_hat|)` representing the distoriton matrix between the input alphabet and the reconstruction alphabet.    
    """
    return np.sum(np.diag(px) @ (pxhat_x * dist_mat))


# Pairwise distortion measures

def hamming(x: np.ndarray, y: np.ndarray) -> float:
    return (x != y).astype(float)

def quadratic(x: np.ndarray, y: np.ndarray) -> float:
    return (x - y) ** 2