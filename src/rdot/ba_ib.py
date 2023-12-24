"""Vanilla Information Bottleneck optimization tools."""

import numpy as np
from scipy.special import logsumexp

from .probability import PRECISION, random_stochastic_matrix
from .information import information_rate
from .distortions import expected_distortion, ib_kl
from .postprocessing import compute_lower_bound
from collections import namedtuple

from tqdm import tqdm


IBResult = namedtuple(
    'IBResult',
    [
        'qxhat_x',
        'rate',
        'distortion',
        'accuracy',
        'beta',
    ]
)

def ba_iterate_ib_rda(
    pxy: np.ndarray,
    betas: np.ndarray,
    num_restarts: int = 1,
    ensure_monotonicity: bool = True,
    **kwargs,
) -> list[IBResult]:
    """Iterate the BA algorithm for an array of values of beta. 
    
    By default, implement reverse deterministic annealing, and implement multiprocessing otherwise.
    
    Args:
        pxy: 2D ndarray, the joint distribution p(x,y)

        betas: 1D array, values of beta to search

        num_restarts: number of initial conditions to try, since we only have convergence to local optima guaranteed.

    Returns: 
        a list of `IBResult` namedtuples, each containing `(qxhat_x, rate, distortion, accuracy, beta)`
    """
    # Reverse deterministic annealing
    results = []    
    betas = np.sort(betas)[::-1] # sort betas in decreasing order

    init_q = np.eye(len(pxy))
    for beta in tqdm(betas):
        candidates = []
        for _ in range(num_restarts):
            cand = blahut_arimoto_ib(pxy, beta, init_q=init_q, **kwargs)
            init_q = cand[0]
            candidates.append(cand)
        best = min(candidates, key=lambda x: x[1] + beta * x[2])
        results.append(best)

    # Postprocessing
    results = results[::-1]
    if ensure_monotonicity:
        indices = compute_lower_bound([(item.rate, item.distortion) for item in results])
        results = [x if i in indices else None for i, x in enumerate(results)]
    return results


def blahut_arimoto_ib(
    pxy: np.ndarray,
    beta: float,
    init_q: np.ndarray = None,
    max_it: int = 200,
    eps: float = 1e-5,
    ignore_converge: bool = False,
) -> IBResult:
    """Estimate the optimal encoder for a given value of `beta` for the Information Bottleneck objective [Tishby et al., 1999].

    Args:
        pxy: 2D array of shape `(|X|, |Y|)` representing the joint probability mass function of the source and relevance variables.

        beta: (scalar) the slope of the rate-distoriton function at the point where evaluation is required

        init_q: the initial encoder `qxhat_x` to begin the optimization; if using RDA, this is the output of the previous optimization. `None` by default, and the initial encoder will be created by random energy-based initialization.

        max_it: max number of iterations

        eps: accuracy required by the algorithm: the algorithm stops if there is no change in distortion value of more than 'eps' between consecutive iterations

        ignore_converge: whether to run the optimization until `max_it`, ignoring the stopping criterion specified by `eps`.

    Returns:
        a IBResult namedtuple of `(qxhat_x, rate, distortion, accuracy, beta)` values. This is:
            `qxhat_x`, the optimal encoder, such that the

            `rate` (in bits) of compressing X into X_hat, is minimized for the level of 
            
            `distortion` between X, X_hat with respect to Y, i.e. the 

            `accuracy` I[X_hat:Y] is maximized, for the specified

            `beta` trade-off parameter
    """
    # Do everything in logspace for stability
    ln_pxy = np.log(pxy + PRECISION)

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

        # d(x, xhat) = E[D[ p(y|x) | q(y|xhat) ]],
        # shape `(x, xhat)`
        dist_mat = ib_kl(np.exp(ln_py_x), np.exp(ln_qy_xhat))

        # p(xhat | x) = p(xhat) exp(- beta * d(xhat, x)) / Z(x),
        # shape `(x, xhat)`
        ln_qxhat_x = ln_qxhat[None,: ] - beta*dist_mat
        ln_qxhat_x = ln_qxhat_x - logsumexp(ln_qxhat_x, axis=1, keepdims=True,) 

        return ln_qxhat, ln_qxhat_x, ln_qy_xhat

    it = 0
    distortion = 2 * eps
    converged = False
    while not converged:
        it += 1
        distortion_prev = distortion

        # Main BA update
        ln_qxhat, ln_qxhat_x, ln_qy_xhat = update_eqs(ln_qxhat, ln_qxhat_x)

        # for convergence check
        distortion = expected_distortion(
            np.exp(ln_px),
            np.exp(ln_qxhat_x),
            ib_kl(np.exp(ln_py_x), np.exp(ln_qy_xhat)),
        )

        # convergence check
        if ignore_converge:
            converged = it == max_it
        else:
            converged = it == max_it or np.abs(distortion - distortion_prev) < eps

    qxhat_x = np.exp(ln_qxhat_x)
    rate = information_rate(np.exp(ln_px), qxhat_x)
    accuracy = information_rate(np.exp(ln_qxhat), np.exp(ln_qy_xhat))
    return IBResult(qxhat_x, rate, distortion, accuracy, beta)