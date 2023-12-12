"""Information Bottleneck plus auxilliary vector distortion optimization tools."""

import numpy as np
from scipy.special import logsumexp

from .probability import PRECISION, random_stochastic_matrix
from .information import information_rate
from .distortions import expected_distortion, ib_mse
from .ba_ib import random_stochastic_matrix

from tqdm import tqdm


def ba_iterate_ib_mse_rda(
    pxy: np.ndarray,
    fx: np.ndarray,
    betas: np.ndarray,
    alphas: np.ndarray,
    num_restarts: int = 1,
    **kwargs,
) -> list[tuple[float]]:
    """Iterate the BA algorithm for an array of values of beta and alpha. 
    
    By default, implement reverse deterministic annealing, and implement multiprocessing otherwise.
    
    Args:
        pxy: 2D ndarray, the joint distribution p(x,y)

        fx: 2D array of shape `(|X|, |f(X)|)` representing the unique vector representations of each value of the source variable X.

        betas: 1D array, values of beta to search

        alphas: 1D array, values of beta to search

        num_restarts: number of initial conditions to try, since we only have convergence to local optima guaranteed.
    """
    # Reverse deterministic annealing
    results = []    
    betas = list(reversed(betas)) # assumes beta was passed low to high

    init_q = np.eye(len(pxy))
    for beta in tqdm(betas):
        for alpha in alphas:
            result = blahut_arimoto_ib_mse(
                pxy, fx, beta, alpha, init_q=init_q, **kwargs
            )
            init_q = result[0]
            results.append(result)

    return results


def blahut_arimoto_ib_mse(
    pxy: np.ndarray,
    fx: np.ndarray,
    beta: float,
    alpha: float,
    weights: np.ndarray = None,
    init_q: np.ndarray = None,
    max_it: int = 200,
    eps: float = 1e-5,
    ignore_converge: bool = False,
) -> tuple[float]:
    """Estimate the optimal encoder for given values of `beta` and `alpha` for the following modified IB objective:

        $\min_{q, f} \frac{1}{\beta} I[X:\hat{X}] + \alpha \mathbb{E}[D_{KL}[p(y|x) || p(y|\hat{x})]] + (1 - \alpha) \mathbb{E}[l(x, \hat{x})],

    where $l$ is a weighted quadratic loss between feature vectors for $x, \hat{x}$:

        $l(x, \hat{x}) = \frac{1}{N} \sum_{i=1}^{N} w_i \cdot \left( f(x)_i - f(\hat{x})_i \right)^2$,

    and $f(x)$ is the feature vector of $x$, and the optimal $f(\hat{x})$ satisfies:

        $f(\hat{x}) = \sum_x q(x|\hat{x}) f(x)$

    Args:
        pxy: 2D array of shape `(|X|, |Y|)` representing the joint probability mass function of the source and relevance variables.

        fx: 2D array of shape `(|X|, |f|)` representing the unique vector representations of each value of the source variable X. Here `|f|` denotes the number of features in each vector x.

        beta: (scalar) the slope of the rate-distoriton function at the point where evaluation is required

        alpha: (scalar) a float between 0 and 1, specifying the trade-off between KL divergence and domain specific (MSE) distortion between feature vectors.

        max_it: max number of iterations

        eps: accuracy required by the algorithm: the algorithm stops if there is no change in distortion value of more than 'eps' between consecutive iterations

        ignore_converge: whether to run the optimization until `max_it`, ignoring the stopping criterion specified by `eps`.

    Returns:
        a tuple of `(qxhat_x, rate, distortion, accuracy)` values. This is the optimal encoder `qxhat_x`, such that the  `rate` (in bits) of compressing X into X_hat, is minimized for the level of `distortion` between X, X_hat
    """
    # Do everything in logspace for stability
    ln_pxy = np.log(pxy + PRECISION) # `(x,y)`
    ln_fx = np.log(fx + PRECISION) # `(x,f)`

    ln_px = logsumexp(ln_pxy, axis=1) # `(x)`
    ln_py_x = ln_pxy - logsumexp(ln_pxy, axis=1, keepdims=True)  # `(x, y)`
    
    # initial encoder, shape `(x, xhat)`; we assume x, xhat are same size
    if init_q is not None:
        ln_qxhat_x = np.log(init_q)
    else:
        ln_qxhat_x = np.log(random_stochastic_matrix((len(ln_px), len(ln_px))))

    # initial q(xhat), shape `(xhat)`
    ln_qxhat = logsumexp(ln_px[:, None] + ln_qxhat_x)

    def update_eqs(
        ln_qxhat: np.ndarray,
        ln_qxhat_x: np.ndarray,
    ) -> tuple[np.ndarray]:
        """Update the required self-consistent equations."""
        # q(xhat) = sum_x p(x) q(xhat | x), 
        # shape `(xhat)`
        ln_qxhat = logsumexp(ln_px[:, None] + ln_qxhat_x, axis=0)

        # q(x,xhat) = p(x) q(xhat|x), 
        # shape `(x, xhat)`
        ln_qxxhat = ln_px[:, None] + ln_qxhat_x

        # p(x|xhat) = q(x, xhat) / q(xhat),
        # shape `(xhat, x)`
        ln_qx_xhat = ln_qxxhat.T - ln_qxhat[:, None]

        # p(y|xhat) = sum_x p(y|x) p(x|xhat),
        # shape `(xhat, y)`
        ln_qy_xhat = logsumexp(
            ln_py_x[None, :, :] + ln_qx_xhat[:, :, None], # `(xhat, x, y)`
            axis=1,
        )

        # f(xhat) = sum_x q(x|xhat) f(x)
        # shape `(xhat, 1)`
        ln_fxhat = logsumexp(
            ln_qx_xhat[:, :,None] + ln_fx[None,:,:], # `(xhat, x, f)`
            axis=1,
        )

        # d(x, xhat) = alpha * E[D[p(y|x)|q(y|xhat)]] + (1-alpha)*E[l(x,xhat)]
        # shape `(x, xhat)`
        # dist_mat = ib_kl(np.exp(ln_py_x), np.exp(ln_qy_xhat))
        dist_mat = ib_mse(
            py_x=np.exp(ln_py_x),
            qy_xhat=np.exp(ln_qy_xhat),
            fx=fx,
            fxhat=np.exp(ln_fxhat),
            alpha=alpha,
            weights=weights,
        )

        # p(xhat | x) = p(xhat) exp(- beta * d(xhat, x)) / Z(x),
        # shape `(x, xhat)`
        ln_qxhat_x = ln_qxhat[None,: ] - beta*dist_mat
        ln_qxhat_x = ln_qxhat_x - logsumexp(ln_qxhat_x, axis=1, keepdims=True,) 

        return (ln_qxhat, ln_qxhat_x, ln_qy_xhat, ln_fxhat, dist_mat)

    it = 0
    distortion = 2 * eps
    converged = False
    while not converged:
        it += 1
        distortion_prev = distortion

        # Main BA update
        ln_qxhat, ln_qxhat_x, ln_qy_xhat, ln_fxhat, dist_mat = update_eqs(ln_qxhat, ln_qxhat_x)

        # for convergence check
        distortion = expected_distortion(
            np.exp(ln_px),
            np.exp(ln_qxhat_x),
            dist_mat,
        )

        # convergence check
        if ignore_converge:
            converged = it == max_it
        else:
            converged = it == max_it or np.abs(distortion - distortion_prev) < eps

    qxhat_x = np.exp(ln_qxhat_x)
    rate = information_rate(np.exp(ln_px), qxhat_x)
    accuracy = information_rate(np.exp(ln_qxhat), np.exp(ln_qy_xhat)) # maximizing this is no longer equivalent to minimizing distortion
    return (qxhat_x, np.exp(ln_fxhat), rate, distortion, accuracy)