"""Unified Rate Distortion optimization classes and methods using Blahut-Arimoto algorithms.
"""
import numpy as np

from collections import namedtuple
from scipy.special import logsumexp
from typing import Any
from tqdm import tqdm
from .distortions import expected_distortion, ib_kl, ib_mse
from .information import information_cond
from .probability import PRECISION, random_stochastic_matrix
from .postprocessing import compute_lower_bound

##############################################################################
# Base Rate Distortion class
##############################################################################

class BaseRDOptimizer:

    def __init__(
        self,
        betas: np.ndarray,
        *args,
        max_it: int = 200,
        eps: float = 1e-5,
        ignore_converge: bool = False,
        **kwargs,
        ) -> None:
        """Base initializer for a Blahut-Arimoto-based optimizer of the Rate Distortion function.
        
        Args:
            betas: 1D array, values of beta to search

            max_it: max number of iterations

            eps: accuracy required by the algorithm: the algorithm stops if there is no change in distortion value of more than `eps` between consecutive iterations

            ignore_converge: whether to run the optimization until `max_it`, ignoring the stopping criterion specified by `eps`.
        """
        self.betas = betas
        self.max_it = max_it
        self.eps = eps
        self.ignore_converge = ignore_converge

        self.init_args = args
        self.init_kwargs = kwargs

        self.ln_px = None # shape `(x)`
        self.ln_qxhat_x = None # shape `(x, xhat)`
        self.dist_mat = None # shape `(x, xhat)`
        self.result = None # namedtuple
        self.results: list[Any] = [] # list of namedtuples

    def get_results(self) -> list[Any]:
        # Re-initialize results
        self.result = None
        self.results = []

        self.beta_iterate(*self.init_args, **self.init_kwargs)
        return self.results

    def update_eqs(self, *args, **kwargs, ) -> None:
        """Main Blahut-Arimoto update steps."""
        raise NotImplementedError
    
    def next_result(self, beta, *args, **kwargs) -> None:
        """Get the result of the converged BA iteration."""
        raise NotImplementedError
    
    def beta_iterate(
        self, 
        *args, 
        num_restarts: int = 1, 
        ensure_monotonicity: bool = True,        
        **kwargs, 
        ) -> None:
        """Iterate the BA algorithm for an array of values of beta, using reverse deterministic annealing.
        
        Args:
            num_restarts: number of times to restart each beta-optimization 

            ensure_monotonicity: whether to remove points that would make a rate distortion curve non-monotonic
        """
        # Reverse deterministic annealing
        results = []
        betas = np.sort(self.betas)[::-1] # sort betas in decreasing order

        init_q = np.eye(len(self.ln_px))
        for beta in tqdm(betas):
            candidates = []
            for _ in range(num_restarts):
                self.blahut_arimoto(beta, *args, init_q=init_q, **kwargs)
                cand = self.results[-1]
                init_q = cand.qxhat_x
                candidates.append(cand)
            best = min(candidates, key=lambda x: x.rate + beta * x.distortion)
            results.append(best)

        # Postprocessing
        results = results[::-1]
        if ensure_monotonicity:
            indices = compute_lower_bound([(item.rate, item.distortion) for item in results])
            results = [x if i in indices else None for i, x in enumerate(results)]
        self.results = results
    
    ############################################################################
    # Blahut Arimoto iteration
    ############################################################################

    def blahut_arimoto(
        self,
        beta,
        *args,
        **kwargs,
        ) -> None:
        """Update the self-consistent equations for a Rate Distortion objective.
        
        Args:
            beta: (scalar) the slope of the rate-distoriton function at the point where evaluation is required
        """
        len_x = len(self.ln_px)
        if "init_q" in kwargs:
            self.ln_qxhat_x = np.log(kwargs["init_q"])
        else:
            self.ln_qxhat_x = np.log(random_stochastic_matrix((len_x,len_x)))

        it = 0
        distortion = 2 * self.eps
        converged = False
        while not converged:
            # print(it)
            it += 1
            distortion_prev = distortion

            # Main BA update
            self.update_eqs(beta, *args, **kwargs)

            # for convergence check
            distortion = self.compute_distortion()

            # convergence check
            if self.ignore_converge:
                converged = it == self.max_it
            else:
                converged = it == self.max_it or np.abs(distortion - distortion_prev) < self.eps

        self.results.append(self.next_result(beta, *args, **kwargs))

    def compute_distortion(self, *args, **kwargs) -> float:
        """Compute the expected distortion for the current p(x), q(xhat|x) and dist_mat."""
        return expected_distortion(
            np.exp(self.ln_px), np.exp(self.ln_qxhat_x), self.dist_mat,
        )
    
    def compute_rate(self, *args, **kwargs,) -> float:
        """Compute the information rate for the current p(x), q(xhat|x)."""
        return information_cond(np.exp(self.ln_px), np.exp(self.ln_qxhat_x))


##############################################################################
# Vanilla Rate Distortion
##############################################################################

RateDistortionResult = namedtuple(
    'RateDistortionResult',
    [
        'qxhat_x',
        'rate',
        'distortion',
        'beta',
    ]
)

def next_ln_qxhat(ln_px: np.ndarray, ln_qxhat_x: np.ndarray) -> np.ndarray:
    # q(xhat) = sum_x p(x) q(xhat | x), 
    # shape `(xhat)`
    return logsumexp(ln_px[:, None] + ln_qxhat_x, axis=0)

def next_ln_qxhat_x(ln_qxhat: np.ndarray, beta: float, dist_mat: np.ndarray):
    # q(x_hat | x) = q(x_hat) exp(- beta * d(x_hat, x)) / Z(x)
    ln_qxhat_x = ln_qxhat[None,: ] - beta*dist_mat
    ln_qxhat_x = ln_qxhat_x - logsumexp(ln_qxhat_x, axis=1, keepdims=True,)
    return ln_qxhat_x

class RateDistortionOptimizer(BaseRDOptimizer):

    def __init__(
        self,
        px: np.ndarray,
        dist_mat: np.ndarray,
        betas: np.ndarray,
        *args,
        **kwargs,
        ) -> None:
        """Compute the rate-distortion function of an i.i.d distribution p(x)

        Args:
            px: (1D array of shape `|X|`) representing the probability mass function of the source.

            dist_mat: array of shape `(|X|, |X_hat|)` representing the distortion matrix between the input alphabet and the reconstruction alphabet.

            beta: (scalar) the slope of the rate-distoriton function at the point where evaluation is required
        """
        super().__init__(betas, *args, **kwargs)
        self.px = px
        self.ln_px = np.log(px)
        self.dist_mat = dist_mat
        self.results: list[RateDistortionResult] = []        

    def get_results(self) -> list[RateDistortionResult]:
        return super().get_results()
    
    def next_result(self, beta, *args, **kwargs) -> None:
        """Get the result of the converged BA iteration.
        
        Returns:
            a RateDistortionResult namedtuple of `(qxhat_x, rate, distortion, accuracy)` values. This is:

                `qxhat_x`, the optimal encoder, such that the

                `rate` (in bits) of compressing X into X_hat, is minimized for the level of 
                
                `distortion` between X, X_hat with respect to Y, i.e. the 

                `accuracy` I[X_hat:Y] is maximized, for the specified

                `beta` trade-off parameter
        """
        return RateDistortionResult(
            np.exp(self.ln_qxhat_x), 
            self.compute_rate(),
            self.compute_distortion(),
            beta,
        )

    def update_eqs(self, beta, *args, **kwargs,) -> None:
        """Iterate the vanilla RD update equations."""
        self.ln_qxhat = next_ln_qxhat(self.ln_px, self.ln_qxhat_x)
        self.ln_qxhat_x = next_ln_qxhat_x(self.ln_qxhat, beta, self.dist_mat)

##############################################################################
# IB
##############################################################################

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

def next_ln_qy_xhat(ln_pxy: np.ndarray, ln_qxhat_x: np.ndarray) -> np.ndarray:
    # p(x),
    # shape `(x)`
    ln_px = logsumexp(ln_pxy, axis=1)

    # p(y|x),
    # shape `(x,y)`
    ln_py_x = ln_pxy - logsumexp(ln_pxy, axis=1, keepdims=True)  # `(x, y)`

    ln_qx_xhat = next_ln_qx_xhat(ln_px, ln_qxhat_x) # `(xhat, x)`

    # p(y|xhat) = sum_x p(y|x) p(x|xhat),
    # shape `(xhat, y)`
    ln_qy_xhat = logsumexp(
        ln_py_x[None, :, :] + ln_qx_xhat[:, :, None], # `(xhat, x, y)`
        axis=1,
    )

    return ln_qy_xhat

def next_ln_qx_xhat(ln_px: np.ndarray, ln_qxhat_x: np.ndarray) -> np.ndarray:
    # q(xhat), 
    # shape `(xhat)`
    ln_qxhat = next_ln_qxhat(ln_px, ln_qxhat_x)

    # q(x,xhat) = p(x) q(xhat|x), 
    # shape `(x, xhat)`
    ln_qxxhat = ln_px[:, None] + ln_qxhat_x

    # p(x|xhat) = q(x, xhat) / q(xhat),
    # shape `(xhat, x)`
    ln_qx_xhat = ln_qxxhat.T - ln_qxhat[:, None]

    return ln_qx_xhat

def next_ln_fxhat(ln_fx: np.ndarray, ln_px: np.ndarray, ln_qxhat_x: np.ndarray) -> np.ndarray:
    ln_qx_xhat = next_ln_qx_xhat(ln_px, ln_qxhat_x) # `(xhat, x)`

    # f(xhat) = sum_x f(x) p(x|xhat),
    # shape `(xhat, f)`
    ln_fxhat = logsumexp(
        ln_fx[None, :, :] + ln_qx_xhat[:, :, None], # `(xhat, x, y)`
        axis=1,
    )
    return ln_fxhat

class IBOptimizer(BaseRDOptimizer):

    def __init__(
        self,
        pxy: np.ndarray,
        betas: np.ndarray,
        *args,
        **kwargs,
        ) -> None:
        """Estimate the optimal encoder for a given value of `beta` for the Information Bottleneck objective [Tishby et al., 1999].

        Args:
            pxy: 2D array of shape `(|X|, |Y|)` representing the joint probability mass function of the source and relevance variables.

            dist_mat: array of shape `(|X|, |X_hat|)` representing the distortion matrix between the input alphabet and the reconstruction alphabet.

            beta: (scalar) the slope of the rate-distoriton function at the point where evaluation is required
        """
        super().__init__(betas, *args, **kwargs)
        self.ln_pxy = np.log(pxy + PRECISION)
        self.ln_px = logsumexp(self.ln_pxy, axis=1) # `(x)`
        self.ln_py_x = self.ln_pxy - logsumexp(self.ln_pxy, axis=1, keepdims=True)  # `(x, y)`
        self.results: list[IBResult] = None

    def get_results(self) -> list[IBResult]:
        return super().get_results()        
    
    def next_dist_mat(self, *args, **kwargs,) -> None:
        """Vanilla IB distortion matrix."""
        self.dist_mat = ib_kl(np.exp(self.ln_py_x), np.exp(self.ln_qy_xhat))
    
    def update_eqs(self, beta, *args, **kwargs,) -> None:
        """Iterate the vanilla IB update equations."""
        self.ln_qxhat = next_ln_qxhat(self.ln_px, self.ln_qxhat_x)
        self.ln_qy_xhat = next_ln_qy_xhat(self.ln_pxy, self.ln_qxhat_x)
        self.next_dist_mat(*args, **kwargs)
        self.ln_qxhat_x = next_ln_qxhat_x(self.ln_qxhat, beta, self.dist_mat)

    def next_result(self, beta, *args, **kwargs) -> IBResult:
        """Get the result of the converged BA iteration for the IB objective.
        
        Returns:
            an IBResult namedtuple of `(qxhat_x, rate, distortion, accuracy, beta)` values. This is:

                `qxhat_x`, the optimal encoder, such that the

                `rate` (in bits) of compressing X into X_hat, is minimized for the level of 
                
                `distortion` between X, X_hat with respect to Y, i.e. the 

                `accuracy` I[X_hat:Y] is maximized, for the specified

                `beta` trade-off parameter
        """
        return IBResult(
            np.exp(self.ln_qxhat_x), 
            self.compute_rate(),
            self.compute_distortion(),
            information_cond(
                np.exp(self.ln_qxhat), np.exp(self.ln_qy_xhat),
            ),
            beta,
        )

##############################################################################
# IB+MSE
##############################################################################

IBMSEResult = namedtuple(
    'IBMSEResult',
    [
        'qxhat_x',
        'fxhat',
        'rate',
        'distortion',
        'accuracy',
        'beta',
        'alpha',
    ]
)

class IBMSEOptimizer(IBOptimizer):

    def __init__(
        self, 
        pxy: np.ndarray, 
        fx: np.ndarray,
        betas: np.ndarray,
        alphas: np.ndarray,
        *args, 
        **kwargs,
        ) -> None:
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

            weights: 1D array of shape `(|f|)` representing weights for feature values
        """
        super().__init__(pxy, betas, *args, **kwargs)
        self.alphas = alphas
        self.ln_fx = np.log(fx + PRECISION) # `(x,f)`
        self.ln_fxhat = None  # `(xhat,f)`

    def get_results(self) -> list[IBMSEResult]:
        return super().get_results()

    def beta_iterate(
        self, 
        *args, 
        num_restarts: int = 1, 
        ensure_monotonicity: bool = True, 
        **kwargs,
        ) -> None:
        """Run the BA iteration for many values of beta and alpha."""
        # Reverse deterministic annealing
        alpha_results: list[IBMSEResult] = []

        for alpha in self.alphas:
            # Run beta_iterate for a single alpha
            kwargs["alpha"] = alpha
            super().beta_iterate(
                *args, 
                num_restarts=num_restarts, ensure_monotonicity=ensure_monotonicity, 
                **kwargs,
            )
            # Collect and reset internal results
            beta_results = self.results
            self.results = []
            alpha_results.extend(beta_results)

        self.results = alpha_results
        return self.results
    
    def update_eqs(self, beta, *args, **kwargs,) -> None:
        """Iterate the IB+MSE objective update equations."""
        self.ln_qxhat = next_ln_qxhat(self.ln_px, self.ln_qxhat_x)
        self.ln_qy_xhat = next_ln_qy_xhat(self.ln_pxy, self.ln_qxhat_x)
        self.ln_fxhat = next_ln_fxhat(self.ln_fx, self.ln_px, self.ln_qxhat_x)
        self.next_dist_mat(*args, **kwargs)
        self.ln_qxhat_x = next_ln_qxhat_x(self.ln_qxhat, beta, self.dist_mat)

    def next_dist_mat(self, *args, **kwargs) -> None:
        """IB+MSE distortion matrix."""
        alpha = kwargs["alpha"]

        weights = None
        if "weights" in kwargs:
            weights = kwargs["weights"]

        self.dist_mat = ib_mse(
            np.exp(self.ln_py_x),
            np.exp(self.ln_qy_xhat),
            np.exp(self.ln_fx),
            np.exp(self.ln_fxhat),
            alpha,
            weights=weights,
        )

    def next_result(self, beta, alpha, *args, **kwargs) -> IBResult:
        """Get the result of the converged BA iteration for the IB+MSE objective.
        
        Returns: 
            an IBMSEResult namedtuple of `(qxhat_x, fxhat, rate, distortion, accuracy, beta, alpha)` values. This is:

                `qxhat_x`, the optimal encoder, 

                `fxhat`, corresponding feature vectors such that the 

                `rate` (in bits) of compressing X into X_hat, is minimized for the level of 

                `distortion` between X, X_hat with respect to Y and f(x), and

                `accuracy` I[X_hat:Y] is maximized, for the specified

                `beta` trade-off parameter, and the specified

                `alpha` combination distortion trade-off parameter.

        """
        return IBMSEResult(
            np.exp(self.ln_qxhat_x),
            np.exp(self.ln_fxhat),
            self.compute_rate(),
            self.compute_distortion(),
            information_cond(
                np.exp(self.ln_qxhat), np.exp(self.ln_qy_xhat),
            ),
            beta,
            alpha,
        )