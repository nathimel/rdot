import numpy as np
from .probability import PRECISION, joint

##############################################################################
# Helper functions for measuring information-theoretic quantities. Code credit belongs to N. Zaslavsky: https://github.com/nogazs/ib-color-naming/blob/master/src/tools.py
##############################################################################

def xlogx(p):
    """Compute $x \log p(x)$"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > PRECISION, p * np.log2(p), 0)


def H(p, axis=None):
    """Compute the entropy of p, $H(X) = - \sum_x x \log p(x)$"""
    return -xlogx(p).sum(axis=axis)


def MI(pXY):
    """Compute mutual information, $I[X:Y]$"""
    return H(pXY.sum(axis=0)) + H(pXY.sum(axis=1)) - H(pXY)


def DKL(p, q, axis=None):
    """Compute KL divergences, $D_{KL}[p~||~q]$"""
    return (xlogx(p) - np.where(p > PRECISION, p * np.log2(q + PRECISION), 0)).sum(
        axis=axis
    )

# Common pattern for rate-distortion optimizations
def information_cond(pA: np.ndarray, pB_A: np.ndarray) -> float:
    """Compute the mutual information $I(A;B)$ from a joint distribution defind by $P(A)$ and $P(B|A)$
    
    Args: 
        pA: array of shape `|A|` the prior probability of an input symbol (i.e., the source)    

        pB_A: array of shape `(|A|, |B|)` the probability of an output symbol given the input        
    """
    pXY = joint(pY_X=pB_A, pX=pA)
    mi = MI(pXY=pXY)
    if mi < 0. and not np.isclose(mi, 0., atol=1e-5):
        raise Exception
    return mi

def gNID(pW_X: np.ndarray, pV_X: np.ndarray, pX: np.ndarray):
    """Compute Generalized Normalized Informational Distance (gNID, in Zaslavsky et al. 2018, SI, Section 3.2) between two encoders.

    Args:
        pW_X: first encoder of shape `(|meanings|, |words|)`

        pV_X: second encoder of shape `(|meanings|, |words|)`

        pX: prior over source variables of shape `(|meanings|,)`
    """
    if len(pX.shape) == 1:
        pX = pX[:, None]
    elif pX.shape[0] == 1 and pX.shape[1] > 1:
        pX = pX.T
    pXW = pW_X * pX
    pWV = pXW.T @ (pV_X)
    pWW = pXW.T @ (pW_X)
    pVV = (pV_X * pX).T @ (pV_X)
    score = 1 - MI(pWV) / (np.max([MI(pWW), MI(pVV)]))
    return score