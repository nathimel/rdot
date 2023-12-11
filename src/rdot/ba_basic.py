import numpy as np
from scipy.special import logsumexp
from .information import information_rate
from .distortions import expected_distortion

def ba_iterate(
    px: np.ndarray,
    dist_mat: np.ndarray,
    betas: np.ndarray,
    **kwargs,
) -> list[tuple[float]]:
    """Iterate the BA algorithm for an array of values of beta."""
    # Unlike the I.B. objective, there are guaranteed results about the convergence to global minima for the 'vanilla' rate distortion objective, using the BA algorithm. This suggests we should not need to use reverse deterministic annealing, although it is unlikely that that hurts.
    return [blahut_arimoto(px, dist_mat, beta, **kwargs) for beta in betas]


def blahut_arimoto(
    px: np.ndarray,
    dist_mat: np.ndarray,
    beta: float,
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
        a tuple of (qxhat_x, rate, distortion) values. This is the optimal encoder `qxhat_x`, such that the  `rate` (in bits) of compressing X into X_hat, is minimized for the level of `distortion` between X, X_hat
    """
    # Do everything in logspace for stability
    ln_px = np.log(px)
    
    # initial encoder, shape `(x, xhat)`
    ln_qxhat_x = np.log(np.full(dist_mat.shape,  1/dist_mat.shape[1]))

    # initial q(xhat), shape `(xhat)`
    ln_qxhat = logsumexp(ln_px[:, None] + ln_qxhat_x)

    def update_eqs(
        ln_qxhat: np.ndarray,
        ln_qxhat_x: np.ndarray,
    ) -> tuple[np.ndarray]:
        """Update the required self-consistent equations."""
        # q(x_hat) = sum p(x) q(x_hat | x)
        ln_qxhat = logsumexp(ln_px[:, None] + ln_qxhat_x, axis=1)
        # q(x_hat | x) = q(x_hat) exp(- beta * d(x_hat, x)) / Z(x)
        ln_qxhat_x = ln_qxhat[None,: ] - beta*dist_mat
        ln_qxhat_x = ln_qxhat_x - logsumexp(ln_qxhat_x, axis=1, keepdims=True,)

        return ln_qxhat, ln_qxhat_x

    it = 0
    distortion = 2 * eps
    converged = False
    while not converged:
        it += 1
        distortion_prev = distortion

        # Main BA update
        ln_qxhat, ln_qxhat_x = update_eqs(ln_qxhat, ln_qxhat_x)

        # for convergence check
        distortion = expected_distortion(px, np.exp(ln_qxhat_x), dist_mat)

        # convergence check
        if ignore_converge:
            converged = it == max_it
        else:
            converged = it == max_it or np.abs(distortion - distortion_prev) < eps

    rate = information_rate(px, np.exp(ln_qxhat_x))
    return (np.exp(ln_qxhat_x), rate, distortion)