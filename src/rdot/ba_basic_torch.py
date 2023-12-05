import torch
import numpy as np
from .probability import PRECISION
from .information import information_rate, DKL
from .distortions import ib_kl, expected_distortion
from multiprocessing import Pool
from tqdm import tqdm

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

        dist_mat: array of shape `(|X|, |X_hat|)` representing the distortion matrix between the input alphabet and the reconstruction alphabet.

        beta: (scalar) the slope of the rate-distoriton function at the point where evaluation is required

        max_it: max number of iterations

        eps: accuracy required by the algorithm: the algorithm stops if there is no change in distortion value of more than 'eps' between consecutive iterations

        ignore_converge: whether to run the optimization until `max_it`, ignoring the stopping criterion specified by `eps`.

    Returns:
        a tuple of (qxhat_x, rate, distortion) values. This is the optimal encoder `qxhat_x`, such that the  `rate` (in bits) of compressing X into X_hat, is minimized for the level of `distortion` between X, X_hat
    """
    # start with iid conditional distribution
    # qxhat_x = np.tile(px, (dist_mat.shape[1], 1)).T
    qxhat_x = np.full((len(px), len(px)),  1/len(px))

    # qxhat = px @ qxhat_x

    # convert to logspace
    dist_mat = torch.from_numpy(dist_mat)
    ln_qxhat_x = torch.log(torch.from_numpy(qxhat_x))
    ln_px = torch.log(torch.from_numpy(px))
    ln_qxhat = torch.logsumexp(ln_px + ln_qxhat_x, dim=0)

    def update_eqs(
        ln_qxhat: torch.Tensor,
        ln_qxhat_x: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Update the required self-consistent equations."""
        # q(x_hat) = sum p(x) q(x_hat | x)
        ln_qxhat = torch.logsumexp(ln_px + ln_qxhat_x, dim=0)


        # q(x_hat | x) = q(x_hat) exp(- beta * d(x_hat, x)) / Z(x)
        # qxhat_x = np.exp(-beta * dist_mat) * qxhat
        # qxhat_x /= np.expand_dims(np.sum(qxhat_x, 1), 1)

        ln_qxhat_x = torch.log_softmax(ln_qxhat.exp() - beta*dist_mat, dim=1)

        return (ln_qxhat, ln_qxhat_x)

    it = 0
    distortion = 2 * eps
    converged = False
    while not converged:
        it += 1
        distortion_prev = distortion

        # Main BA update
        ln_qxhat, ln_qxhat_x = update_eqs(ln_qxhat, ln_qxhat_x)

        # for convergence check
        distortion = expected_distortion(ln_px.exp().numpy(), ln_qxhat_x.exp().numpy(), dist_mat.numpy())

        # convergence check
        if ignore_converge:
            converged = it == max_it
        else:
            converged = it == max_it or np.abs(distortion - distortion_prev) < eps

    rate = information_rate(ln_px.exp().numpy(), ln_qxhat_x.exp().numpy())
    return (ln_qxhat_x.exp().numpy(), rate, distortion)