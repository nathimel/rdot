import numpy as np
from .probability import PRECISION, joint

##############################################################################
# Helper functions for measuring information-theoretic quantities. Code credit belongs to N. Zaslavsky: https://github.com/nogazs/ib-color-naming/blob/master/src/tools.py
##############################################################################

def xlogx(p):
    """Compute $x \\log p(x)$"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > PRECISION, p * np.log2(p), 0)


def H(p, axis=None):
    """Compute the entropy of p, $H(X) = - \sum_x x \\log p(x)$"""
    return -xlogx(p).sum(axis=axis)


def MI(pXY):
    """Compute mutual information, $I[X:Y]$"""
    return H(pXY.sum(axis=0)) + H(pXY.sum(axis=1)) - H(pXY)


def DKL(p, q, axis=None):
    """Compute KL divergences, $D_{KL}[p~||~q]$"""
    return (xlogx(p) - np.where(p > PRECISION, p * np.log2(q + PRECISION), 0)).sum(
        axis=axis
    )

# Helper function

def information_rate(px: np.ndarray, qxhat_x: np.ndarray) -> float:
    """Compute the information rate $I(X;\hat{X})$ of a joint distribution defind by $P(X)$ and $P(\hat{X}|X)$
    
    Args: 
        px: array of shape `|X|` the prior probability of an input symbol (i.e., the source)    

        qxhat_x: array of shape `(|X|, |X_hat|)` the probability of an output symbol given the input        
    """
    pXY = joint(pY_X=qxhat_x, pX=px)
    # return MI(pXY=pXY)
    mi = MI(pXY=pXY)
    if mi < 0 and not np.isclose(mi, 0.):
        breakpoint()
    return mi