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


