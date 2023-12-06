import numpy as np
from .probability import PRECISION
from .information import information_rate, DKL
from .distortions import ib_kl, expected_distortion
from multiprocessing import Pool
from tqdm import tqdm

def ib_method(
    pxy: np.ndarray,
    betas: np.ndarray,
    num_processes: int = 1,
    **kwargs,
) -> list[tuple[float]]:
    """Iterate the BA algorithm for an array of values of beta. By default, implement reverse deterministic annealing, and implement multiprocessing otherwise."""

    ba = lambda beta: blahut_arimoto_ib(pxy, beta, **kwargs)

    if num_processes > 1:
        with Pool(num_processes) as p:
            async_results = [
                p.apply_async(
                    ba,
                    args=[beta],
                )
                for beta in betas
            ]
            p.close()
            p.join()
        results = [async_result.get() for async_result in async_results]

    else:
        results = reversed([ba(beta) for beta in tqdm(reversed(betas))])

    return results


def blahut_arimoto_ib(
    pxy: np.ndarray,
    beta: float,
    max_it: int = 200,
    eps: float = 1e-5,
    ignore_converge: bool = False,
) -> tuple[float]:
    """Solve the IB objective.

    Args:
        pxy: 2D array of shape `(|X|, |Y|)` representing the joint probability mass function of the source variable and relevance variable.

        beta: (scalar) the slope of the rate-distoriton function at the point where evaluation is required

        max_it: max number of iterations

        eps: accuracy required by the algorithm: the algorithm stops if there is no change in distortion value of more than 'eps' between consecutive iterations

        ignore_converge: whether to run the optimization until `max_it`, ignoring the stopping criterion specified by `eps`.

    Returns:
        a tuple of (qxhat_x, rate, distortion) values. This is the optimal encoder `qxhat_x`, such that the  `rate` (in bits) of compressing X into X_hat, is minimized for the level of `distortion` between X, X_hat
    """

    px = pxy.sum(axis=1)
    py_x = pxy / pxy.sum(axis=1, keepdims=True)

    # start with iid conditional distribution
    qxhat_x = np.tile(px, (len(px), 1)).T
    qxhat = px @ qxhat_x

    def update_eqs(
        qxhat: np.ndarray,
        qxhat_x: np.ndarray,
    ) -> tuple[np.ndarray]:
        """Update the required self-consistent equations."""
        # q(xhat) = sum_x p(x) q(xhat | x)
        qxhat = px @ qxhat_x

        # q(x,xhat) = p(x) q(xhat|x)
        qxxhat = px * qxhat_x
        # p(x|x_hat)
        qx_xhat = qxxhat.T / qxhat
        # q(y|xhat) = sum_x p(y|x) q(x | xhat)
        qy_xhat = qx_xhat @ py_x
        dist_mat = ib_kl(py_x, qy_xhat)

        # p(xhat | x) = p(xhat) exp(- beta * d(xhat, x)) / Z(x)
        breakpoint()
        exp_term = np.exp(-beta * dist_mat)
        qxhat_x = np.where(
            exp_term > PRECISION, exp_term * qxhat, 1/len(qxhat) * qxhat
        )
        qxhat_x /= np.expand_dims(np.sum(qxhat_x, 1), 1)

        # qxhat_x = np.exp(-beta * dist_mat) * qxhat # this causes underflow
        # Zx = np.expand_dims(np.sum(qxhat_x, 1), 1)
        # qxhat_x = np.where(Zx > PRECISION, qxhat_x / Zx, 1 / qxhat_x.shape[1])
        # qxhat_x /= np.expand_dims(np.sum(qxhat_x, 1), 1)

        return qxhat, qxhat_x, qy_xhat

    it = 0
    distortion = 2 * eps
    converged = False
    while not converged:
        it += 1
        distortion_prev = distortion

        # Main BA update
        qxhat, qxhat_x, qy_xhat = update_eqs(qxhat, qxhat_x)

        # for convergence check
        distortion = expected_distortion(px, qxhat_x, ib_kl(py_x, qy_xhat))

        # convergence check
        if ignore_converge:
            converged = it == max_it
        else:
            converged = it == max_it or np.abs(distortion - distortion_prev) < eps

    rate = information_rate(px, qxhat_x)
    return qxhat_x, rate, distortion