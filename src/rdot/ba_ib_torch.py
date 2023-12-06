import torch
import numpy as np
from .probability import PRECISION
from .information import information_rate, DKL
from .distortions import ib_kl, expected_distortion
from multiprocessing import Pool
from tqdm import tqdm

def random_stochastic_matrix(shape: tuple[int], alpha = 1.) -> torch.Tensor:
    """Initialize a stochastic matrix (2D tensor) that sums to 1. along the rows."""
    energies = alpha * torch.randn(*shape)
    return torch.softmax(energies, dim=1)

def ib_method(
    pxy: np.ndarray,
    betas: np.ndarray,
    num_processes: int = 1,
    **kwargs,
) -> list[tuple[float]]:
    """Iterate the BA algorithm for an array of values of beta. By default, implement reverse deterministic annealing, and implement multiprocessing otherwise."""

    if num_processes > 1:
        with Pool(num_processes) as p:
            async_results = [
                p.apply_async(
                    args=[pxy, beta],
                    kwds=kwargs,
                )
                for beta in betas
            ]
            p.close()
            p.join()
        results = [async_result.get() for async_result in async_results]

    else:
        # Reverse deterministic annealing
        results = []
        q = random_stochastic_matrix((len(pxy), len(pxy))).numpy() # or np.eye
        for beta in tqdm(reversed(betas)):
            # initial encoder to BA at each step is result of previous opt
            q, rate, dist = blahut_arimoto_ib(pxy, beta, qxhat_x=q, **kwargs)
            results.append((q, rate, dist))

    return results


def blahut_arimoto_ib(
    pxy: np.ndarray,
    beta: float,
    qxhat_x: np.ndarray = None,
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

    ln_px = torch.logsumexp(ln_pxy, dim=1) # `[x]`
    ln_py_x = ln_pxy - torch.logsumexp(ln_pxy, dim=1, keepdim=True)  # `[x, y]`

    # initial encoder, shape `[x, x] = [x, xhat]`
    if qxhat_x is not None:
        ln_qxhat_x = torch.log(torch.from_numpy(qxhat_x))
    ln_qxhat_x = torch.log(random_stochastic_matrix((len(ln_px), len(ln_px))))

    # initial q(xhat), see `update_eqs`
    ln_qxhat = torch.logsumexp(ln_px + ln_qxhat_x, dim=0)

    def update_eqs(
        ln_qxhat: torch.Tensor,
        ln_qxhat_x: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Update the required self-consistent equations."""
        # q(xhat) = sum_x p(x) q(xhat | x), 
        # shape `[xhat]`
        ln_qxhat = torch.logsumexp(ln_px + ln_qxhat_x, dim=0)

        # q(x,xhat) = p(x) q(xhat|x), 
        # shape `[x, xhat]`
        ln_qxxhat = ln_px + ln_qxhat_x

        # p(x|x_hat) = q(x, xhat) / q(xhat),
        # shape `[xhat, x]`
        ln_qx_xhat = ln_qxxhat - ln_qxhat

        # q(y|xhat) = sum_x p(y|x) q(x | xhat),
        # shape `[xhat, y]`
        # breakpoint()
        ln_qy_xhat = torch.logsumexp(
            # shape `[xhat, x, y]`
            ln_py_x[None, :, :] + ln_qx_xhat[:, :, None],
            dim=1, # sum over x
        )
        # breakpoint()

        # Alternative
        # = 1/q(xhat) sum_x p(x,y) q(xhat|x)
        ln_qxyxhat = ln_qxhat_x[:, None, :] + ln_pxy[:, :, None] #`[x, y, xhat]`
        ln_qxhaty = torch.logsumexp(ln_qxyxhat, dim=0).T # `[xhat, y]`
        ln_qxhat = torch.logsumexp(ln_qxhaty, dim=1, keepdim=True) # `[xhat, 1]`
        ln_qy_xhat = ln_qxhaty - ln_qxhat # `[xhat, y]`

        # d(x, xhat) = E[D[ p(y|x) | q(y|xhat) ]],
        # shape `[x, xhat]`
        dist_mat = torch.from_numpy(ib_kl(ln_py_x.exp(), ln_qy_xhat.exp()))

        # p(xhat | x) = p(xhat) exp(- beta * d(xhat, x)) / Z(x),
        # shape `[x, xhat]`
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
    return ln_qxhat_x.exp().numpy(), rate, distortion