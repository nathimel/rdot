import torch
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
    """Iterate the BA algorithm for an array of values of beta."""

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

        beta: (scalar) the slope of the rate-distortion function at the point where evaluation is required

        max_it: max number of iterations

        eps: accuracy required by the algorithm: the algorithm stops if there is no change in distortion value of more than 'eps' between consecutive iterations

        ignore_converge: whether to run the optimization until `max_it`, ignoring the stopping criterion specified by `eps`.

    Returns:
        a tuple of (qxhat_x, rate, distortion) values. This is the optimal encoder `qxhat_x`, such that the  `rate` (in bits) of compressing X into X_hat, is minimized for the level of `distortion` between X, X_hat
    """
    # Do everything in log space to prevent numerical underflow 
    ln_pxy = torch.log(torch.from_numpy(pxy))

    ln_px = torch.logsumexp(ln_pxy, dim=1)
    ln_py_x = ln_pxy - torch.logsumexp(ln_pxy, dim=1, keepdim=True)

    # initial encoder
    ln_qxhat_x = (torch.randn(len(ln_px), len(ln_px))).log_softmax(dim=1)

    # initial q(xhat), see `update_eqs`
    ln_qxhat = torch.logsumexp(ln_px + ln_qxhat_x, dim=0)

    def update_eqs(
        ln_qxhat: torch.Tensor,
        ln_qxhat_x: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Update the required self-consistent equations."""
        # q(xhat) = sum_x p(x) q(xhat | x)
        ln_qxhat = torch.logsumexp(ln_px + ln_qxhat_x, dim=0)

        # q(x,xhat) = p(x) q(xhat|x)
        ln_qxxhat = ln_px + ln_qxhat_x

        # p(x|x_hat)
        ln_qx_xhat = ln_qxxhat - ln_qxhat

        # q(y|xhat) = sum_x p(y|x) q(x | xhat)
        ln_qy_xhat = torch.logsumexp(ln_qx_xhat + ln_py_x, dim=0)

        # d(x, xhat) = E[D[ p(y|x) | q(y|xhat) ]]
        dist_mat = torch.from_numpy(ib_kl(ln_py_x.exp(), ln_qy_xhat.exp()))

        breakpoint()
        # problem: converting to log space makes differences in distortion too small, so that we never get the deterministic mapping! why?
        ln_qxhat_x = torch.log_softmax(ln_qxhat - beta * dist_mat, dim=1)

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
            ln_px.exp().numpy(), 
            ln_qxhat_x.exp().numpy(), 
            ib_kl(ln_py_x.exp(), ln_qy_xhat.exp()),
        )

        # convergence check
        if ignore_converge:
            converged = it == max_it
        else:
            converged = it == max_it or np.abs(distortion - distortion_prev) < eps

    rate = information_rate(ln_px.exp().numpy(), ln_qxhat_x.exp().numpy())
    return ln_qxhat_x.exp(), rate, distortion