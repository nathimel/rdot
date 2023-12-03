import numpy as np
from .information import information_rate
from .distortion import expected_distortion
from multiprocessing import Pool

def ba_iterate(
    px: np.ndarray,
    dist_mat: np.ndarray,
    betas: np.ndarray,
    num_processes: int = 1,
    **kwargs,
) -> list[tuple[float]]:
    """Iterate the BA algorithm for an array of values of beta."""

    # Unlike the I.B. objective, there are guaranteed results about the convergence to global minima for the 'vanilla' rate distortion objective, using the BA algorithm. This suggests we should not need to use reverse deterministic annealing, although it is unlikely that that hurts.
    ba = lambda beta: blahut_arimoto(px, dist_mat, beta, **kwargs)    

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
        results = [ba(beta) for beta in betas]

    return results



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

        dist_mat: array of shape `(|X|, |X_hat|)` representing the distortion matrix between the input alphabet and the reconstruction alphabet. dist_mat[i,j] = dist(x[i],x_hat[j]). Note that |Xhat| may be smaller than |X|.

        beta: (scalar) the slope of the rate-distoriton function at the point where evaluation is required

        max_it: max number of iterations

        eps: accuracy required by the algorithm: the algorithm stops if there is no change in distortion value of more than 'eps' between consecutive iterations

        ignore_converge: whether to run the optimization until `max_it`, ignoring the stopping criterion specified by `eps`.

    Returns:
        a tuple of (rate, distortion) values. This is the rate (in bits) of compressing X into X_hat, and distortion between X, X_hat
    """
    # start with iid conditional distribution
    pxhat_x = np.tile(px, (dist_mat.shape[1], 1)).T

    # normalize
    px /= np.sum(px)
    pxhat_x /= np.sum(pxhat_x, 1, keepdims=True)

    it = 0
    traj = []
    distortion = 2 * eps
    converged = False
    while not converged:
        it += 1
        distortion_prev = distortion

        # p(x_hat) = sum p(x) p(x_hat | x)
        pxhat = px @ pxhat_x

        # p(x_hat | x) = p(x_hat) exp(- beta * d(x_hat, x)) / Z(x)
        pxhat_x = np.exp(-beta * dist_mat) * pxhat
        pxhat_x /= np.expand_dims(np.sum(pxhat_x, 1), 1)

        # update for convergence check
        rate = information_rate(px, pxhat_x)
        distortion = expected_distortion(px, pxhat_x, dist_mat)

        # convergence check
        if ignore_converge:
            converged = it == max_it
        else:
            converged = it == max_it or np.abs(distortion - distortion_prev) < eps

        return (rate, distortion)