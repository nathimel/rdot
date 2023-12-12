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
    return np.subtract.outer(x,y)**2

def ib_kl(py_x: np.ndarray, qy_xhat: np.ndarray) -> np.ndarray:
    # D[p(y|x) || q(y|xhat)]
    return DKL(py_x[:,None,:], qy_xhat[None,:,:], axis=2)

def feature_loss(fx: np.ndarray, fxhat: np.ndarray, weights: np.ndarray = None) -> float:
    # 1/|f| sum_{i}^{|f|} w_i [ f(x)_i - f(\hat{x})_i ]^2
    if fx.shape != fxhat.shape:
        raise ValueError

    num_features = fx.shape[1]
    if weights is None:
        weights = np.ones(num_features)

    result = np.sum(
        # `(x,xhat,f)`
        weights * (fx[:,None,:] - fxhat[None,:,:])**2,
        axis=2,
    ) / num_features
    
    return result

def ib_mse(py_x: np.ndarray, qy_xhat: np.ndarray, fx: np.ndarray, fxhat: np.ndarray, alpha: float, weights: np.ndarray):
    return (
        alpha * ib_kl(py_x, qy_xhat) 
        + (1 - alpha) * 1/fx.shape[1] * feature_loss(fx, fxhat, weights)
        )