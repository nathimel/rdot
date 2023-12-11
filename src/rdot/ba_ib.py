import numpy as np
from scipy.special import logsumexp, softmax

from .probability import PRECISION
from .information import information_rate
from .distortions import expected_distortion, ib_kl

from tqdm import tqdm

def ib_method(
    pxy: np.ndarray,
    betas: np.ndarray,
    num_restarts: int = 1,
    **kwargs,
) -> list[tuple[float]]:
    """Iterate the BA algorithm for an array of values of beta. 
    
    By default, implement reverse deterministic annealing, and implement multiprocessing otherwise.
    
    Args:
        pxy: 2D ndarray, the joint distribution p(x,y)

        betas: 1D array, values of beta to search

        num_processes: number of CPU cores to pass to multiprocessing.Pool to compute different solutions in parallel for each beta. Each beta solution is still computed serially.

        num_restarts: number of initial conditions to try, since we only have convergence to local optima guaranteed.
    """
    # Reverse deterministic annealing
    results = []    
    betas = list(reversed(betas)) # assumes beta was passed low to high

    init_q = np.eye(len(pxy))
    for beta in tqdm(betas):
        candidates = []
        for _ in range(num_restarts):
            cand = blahut_arimoto_ib(pxy, beta, init_q=init_q, **kwargs)
            init_q = cand[0]
            candidates.append(cand)
        best = min(candidates, key=lambda x: x[1] + beta * x[2])
        results.append(best)

    return results


def random_stochastic_matrix(shape: tuple[int], alpha = 1.) -> np.ndarray:
    """Initialize a stochastic matrix (2D tensor) that sums to 1. along the rows."""
    energies = alpha * np.random.normal(size=shape)
    return softmax(energies, axis=1)


def blahut_arimoto_ib(
    pxy: np.ndarray,
    beta: float,
    init_q: np.ndarray = None,
    max_it: int = 200,
    eps: float = 1e-5,
    ignore_converge: bool = False,
) -> tuple[float]:
    """Compute the rate-distortion function of an i.i.d distribution p(x)

    Args:
        px: (1D array of shape `|X|`) representing the probability mass function of the source.

        dist_mat: array of shape `(|X|, |X_hat|)` representing the distortion matrix between the input alphabet and the reconstruction alphabet.

        beta: (scalar) the slope of the rate-distoriton function at the point where evaluation is required

        max_it: max number of iterations

        eps: accuracy required by the algorithm: the algorithm stops if there is no change in distortion value of more than 'eps' between consecutive iterations

        ignore_converge: whether to run the optimization until `max_it`, ignoring the stopping criterion specified by `eps`.

    Returns:
        a tuple of `(qxhat_x, rate, distortion, accuracy)` values. This is the optimal encoder `qxhat_x`, such that the  `rate` (in bits) of compressing X into X_hat, is minimized for the level of `distortion` between X, X_hat
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
    return (qxhat_x, rate, distortion, accuracy)