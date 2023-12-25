import numpy as np
from rdot import (
    ba,
    information,
    distortions,
    postprocessing,
)

# The following test cases were taken from the following file in Alon Kipnis' repo: https://github.com/alonkipnis/BlahutArimoto/blob/master/example.py

class TestRDBinaryHamming:

    """Binary random variable with hamming distortion"""

    def test_compute_rate(self):

        # Mutual info between X, Y is bounded from above by the entropy of the variable with the smaller alphabet, i.e.
        # I(X;Y) <= log(min(|X|, |Y|))
        px = np.array([.5, .5,])
        pxhat_x = np.eye(2)
        rate = information.information_cond(
            px,
            pxhat_x
        )
        upper_bound = np.log2(2)
        assert rate == upper_bound

    def test_compute_distortion(self):
        # Simple uniform prior and alphabet = 4
        x = np.arange(4)
        dist_mat = distortions.hamming(x, x)
        px = np.ones(4)/4
        pxhat_x = np.full((4,4), 1/4)
        expected = 0.75
        actual = distortions.expected_distortion(px, pxhat_x, dist_mat)
        assert expected == actual

    def test_ba_beta_0(self):
        
        x = np.array([0,1]) # Binary input
        xhat = np.array([0,1]) # Binary reconstruction
        p = 0.5 # P(X=1) = p
        px = np.array([1-p, p])
        beta = 0.
        
        # distortion matrix
        dist_mat = distortions.hamming(x, xhat)
        optimizer = ba.RateDistortionOptimizer(
            px, dist_mat, np.array([beta]),
        )
        result, = optimizer.get_results()
        rate = result.rate
        dist = result.distortion
        encoder = result.qxhat_x

        # The true R(D) function is bounded by H(p) - H(D); see Cover and Thomas, Rate Distortion Theory, Eq. (10.23).
        true_rate = information.H(p) - information.H(dist)
        assert rate >= true_rate

        # degenerate
        assert dist == 0.5

        expected = np.full_like(encoder, 1/len(xhat))
        assert np.allclose(expected, encoder)


    def test_ba_beta_1e10(self):

        x = np.array([0,1]) # Binary input
        xhat = np.array([0,1]) # Binary reconstruction
        p = 0.5 # P(X=1) = p
        px = np.array([1-p, p])
        beta = 1e10
        
        # distortion matrix
        dist_mat = distortions.hamming(x, xhat)
        optimizer = ba.RateDistortionOptimizer(
            px, dist_mat, np.array([beta]),
        )
        result, = optimizer.get_results()
        rate = result.rate
        dist = result.distortion
        encoder = result.qxhat_x

        # The true R(D) function is bounded by H(p) - H(D); see Cover and Thomas, Rate Distortion Theory, Eq. (10.23).
        true_rate = information.H(p) - information.H(dist)

        # Should it be equal?
        assert rate >= true_rate

        # deterministic
        assert np.isclose(dist, 0.)
        assert len(np.argwhere(encoder)) == len(xhat)

    def test_curve(self):

        x = np.array([0,1]) # Binary input
        xhat = np.array([0,1]) # Binary reconstruction
        p = 0.5 # P(X=1) = p
        px = np.array([1-p, p])
        
        # distortion matrix
        dist_mat = distortions.hamming(x, xhat)

        # Test many values of beta to sweep out a curve. 
        betas = np.logspace(-5, 5, num=100)

        optimizer = ba.RateDistortionOptimizer(px, dist_mat, betas)
        results = optimizer.get_results()

        # rd_values = ba_basic.ba_iterate(px, dist_mat, betas)

        # Check for convexity
        ind1 = 20
        ind2 = 30
        ind3 = 40
        
        # R, D points
        x1, y1 = results[ind1].rate, results[ind1].distortion
        x2, y2 = results[ind2].rate, results[ind2].distortion
        x3, y3 = results[ind3].rate, results[ind3].distortion

        assert x1 < x2
        assert x2 < x3

        assert y1 > y2
        assert y2 > y3

        # The more general version of this test would check that all points on the curve satisfy the definition of convexity for a function


class TestRDGaussianQuadratic:

    """Gaussian random variable with quadratic distortion"""

    def test_ba_beta_0(self):
        # (truncated) Gaussian input with quadratic distortion
        x = np.linspace(-5,5,1000) # source alphabet
        xhat = np.linspace(-5,5,1000) # reconstruction alphabet
        px = 1/(2*np.pi) * np.exp(-x ** 2 / 2) # source pdf
        px /= px.sum() # guess we actually need this

        dist_mat = distortions.quadratic(x, xhat)
        beta = 0.
        
        # distortion matrix
        dist_mat = distortions.hamming(x, xhat)
        optimizer = ba.RateDistortionOptimizer(
            px, dist_mat, np.array([beta]),
        )
        result, = optimizer.get_results()
        rate = result.rate
        dist = result.distortion
        encoder = result.qxhat_x

        true = 2 ** (-2 * rate) # D(R) = σ^2 2^{−2R} in theory, but we truncated
        estimated = dist
        assert np.isclose(rate, 0., atol=1e-5)

        # Is this too strong a requirement in 'Gaussian' case?
        # expected = np.full_like(encoder, 1/len(xhat))
        # assert np.allclose(expected, encoder, atol=1e-5)

    def test_ba_beta_1e10(self):
        # (truncated) Gaussian input with quadratic distortion
        x = np.linspace(-5,5,1000) # source alphabet
        xhat = np.linspace(-5,5,1000) # reconstruction alphabet
        px = 1/(2*np.pi) * np.exp(-x ** 2 / 2) # source pdf
        px /= px.sum() # guess we actually need this

        dist_mat = distortions.quadratic(x, xhat)
        beta = 1e10
        
        # distortion matrix
        dist_mat = distortions.hamming(x, xhat)
        optimizer = ba.RateDistortionOptimizer(
            px, dist_mat, np.array([beta]),
        )
        result, = optimizer.get_results()
        rate = result.rate
        dist = result.distortion
        encoder = result.qxhat_x

        # deterministic
        assert np.isclose(dist, 0.)

        assert len(np.argwhere(encoder)) == len(xhat)

    def test_curve(self):

        # (truncated) Gaussian input with quadratic distortion
        x = np.linspace(-5,5,1000) # source alphabet
        xhat = np.linspace(-5,5,1000) # reconstruction alphabet
        px = 1/(2*np.pi) * np.exp(-x ** 2 / 2) # source pdf
        px /= px.sum() # guess we actually need this

        dist_mat = distortions.quadratic(x, xhat)

        # Test many values of beta to sweep out a curve. 
        betas = np.logspace(-5, 5, num=100)

        optimizer = ba.RateDistortionOptimizer(px, dist_mat, betas)
        results = optimizer.get_results()

        # rd_values = ba_basic.ba_iterate(px, dist_mat, betas)

        # Check for convexity
        ind1 = 20
        ind2 = 30
        ind3 = 40
        
        # R, D points
        x1, y1 = results[ind1].rate, results[ind1].distortion
        x2, y2 = results[ind2].rate, results[ind2].distortion
        x3, y3 = results[ind3].rate, results[ind3].distortion

        assert x1 < x2
        assert x2 < x3

        assert y1 > y2
        assert y2 > y3


class TestIB:

    def test_ba_beta_0(self):

        # Gaussian-like p(y|x)
        py_x = np.array([[np.exp(-(i - j)**2) for j in range(10)] for i in range(10)])
        py_x /= py_x.sum(axis=1)[:, None]
        # get joint by multiplying by p(x)
        px = np.full(py_x.shape[0], 1/10)
        pxy = py_x * px

        beta = 0.
        
        # distortion matrix
        optimizer = ba.IBOptimizer(
            pxy, betas=np.array([beta]),
        )
        result, = optimizer.get_results()
        rate = result.rate
        dist = result.distortion
        encoder = result.qxhat_x

        # degenerate
        assert np.isclose(rate, 0.)

        expected = np.full_like(encoder, 1/len(px))
        assert np.allclose(expected, encoder)

    def test_ba_beta_1e10(self):
        # Gaussian-like p(y|x)
        py_x = np.array([[np.exp(-(i - j)**2) for j in range(10)] for i in range(10)])
        py_x /= py_x.sum(axis=1)[:, None]
        # get joint by multiplying by p(x)
        px = np.full(py_x.shape[0], 1/10)
        pxy = py_x * px
        
        beta = 1e10
        
        # distortion matrix
        optimizer = ba.IBOptimizer(
            pxy, betas=np.array([beta]),
        )
        result, = optimizer.get_results()
        rate = result.rate
        dist = result.distortion
        encoder = result.qxhat_x

        # trivial
        assert np.isclose(dist, 0.)

        assert len(np.argwhere(encoder)) == len(px)

    def test_curve_exp(self):
        py_x = np.array([[np.exp(-(i - j)**2) for j in range(10)] for i in range(10)])
        py_x /= py_x.sum(axis=1)[:, None]
        # get joint by multiplying by p(x)
        px = np.full(py_x.shape[0], 1/10)
        pxy = py_x * px

        # Test many values of beta to sweep out a curve. 
        betas = np.logspace(-2, 5, num=50)

        optimizer = ba.IBOptimizer(pxy, betas)
        rd_values = [
            (result.rate, result.distortion) 
            for result in optimizer.get_results() 
            if result is not None
        ]

        # Check for convexity
        ind1 = 0
        ind2 = int(len(rd_values)/2)
        ind3 = len(rd_values) - 1

        # R, D points
        x1, y1 = rd_values[ind1]
        x2, y2 = rd_values[ind2]
        x3, y3 = rd_values[ind3]

        assert x1 < x2
        assert x2 < x3

        assert y1 > y2
        assert y2 > y3        


    def test_ba_beta_1e10_x100_y10(self):
        # Make sure we test when |Y| != |X|, e.g. |X| = 100, |Y| = 10
        # This test is very minimal; we're really only making sure no syntax or runtime errors are thrown when cardinality of X, Y are different.

        # Gaussian-like p(y|x)
        py_x = np.array([[np.exp(-(i - j)**2) for j in range(0, 100, 10)] for i in range(100)])
        py_x /= py_x.sum(axis=1)[:, None]
        # get joint by multiplying by p(x)
        px = np.full(py_x.shape[0], 1/100)
        pxy = py_x * px[:, None]
        
        beta = 1e10
        
        # distortion matrix
        optimizer = ba.IBOptimizer(
            pxy, betas=np.array([beta]),
        )
        result, = optimizer.get_results()
        rate = result.rate
        dist = result.distortion
        encoder = result.qxhat_x 

    def test_ba_binary_dist_beta_0(self):

        # Same kind of checks as above, but using a diff distribution
        py_x = np.array(
            [[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], 
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
        ).T
        py_x /= py_x.sum(axis=1)[:, None]
        # get joint by multiplying by p(x)
        px = np.full(py_x.shape[0], 1/py_x.shape[0])
        pxy = py_x * px[:, None]

        beta = 0.
        
        # distortion matrix
        optimizer = ba.IBOptimizer(
            pxy, betas=np.array([beta]),
        )
        result, = optimizer.get_results()
        rate = result.rate
        dist = result.distortion
        encoder = result.qxhat_x

    def test_ba_binary_dist_beta_1e10(self):

        py_x = np.array(
            [[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], 
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
        ).T
        py_x /= py_x.sum(axis=1)[:, None]
        # get joint by multiplying by p(x)
        px = np.full(py_x.shape[0], 1/py_x.shape[0])
        pxy = py_x * px[:, None]

        beta = 1e10
        
        # distortion matrix
        optimizer = ba.IBOptimizer(
            pxy, betas=np.array([beta]),
        )
        result, = optimizer.get_results()
        rate = result.rate
        dist = result.distortion
        encoder = result.qxhat_x
        

    def test_ba_binary_dist_beta_low(self):

        # Same kind of checks as above, but using a diff distribution
        py_x = np.array(
            [[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], 
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
        ).T
        py_x /= py_x.sum(axis=1)[:, None]
        # get joint by multiplying by p(x)
        px = np.full(py_x.shape[0], 1/py_x.shape[0])
        pxy = py_x * px[:, None]

        # Trivial solutions occur for beta < 1
        betas = np.logspace(-5, 0., num=30) # 0. can be changed to -1 if nec.

        optimizer = ba.IBOptimizer(pxy, betas)
        rates = [
            result.rate for result in optimizer.get_results()
            if result is not None
        ]

        assert np.allclose(rates, 0.)

    def test_ba_binary_dist_deterministic(self):

        # Should be a trivial bound, since I[X:Xhat] = I[Xhat:Y]

        # Medin and Schaffer deterministic category labels
        py_x = np.array(
            [
                [0., 1.],
                [0., 1.],
                [0., 1.],
                [0., 1.],
                [0., 1.],
                [1., 0.],
                [1., 0.],
                [1., 0.],
                [1., 0.],
            ]
        )
        py_x /= py_x.sum(axis=1)[:, None]
        # get joint by multiplying by p(x)
        px = np.full(py_x.shape[0], 1/py_x.shape[0])
        pxy = py_x * px[:, None]

        betas = np.logspace(-2, 5, num=30)

        results = ba.IBOptimizer(pxy, betas).get_results()


class TestIBMSE:


    def test_recover_ib(self):

        # Medin and Schaffer deterministic category labels
        py_x = np.array(
            [
                [0., 1.],
                [0., 1.],
                [0., 1.],
                [0., 1.],
                [0., 1.],
                [1., 0.],
                [1., 0.],
                [1., 0.],
                [1., 0.],
            ]
        )
        py_x /= py_x.sum(axis=1)[:, None]
        # get joint by multiplying by p(x)
        px = np.full(py_x.shape[0], 1/py_x.shape[0])
        pxy = py_x * px[:, None]

        fx = np.array(
            [
                # A
                [0.,0.,0.,1.],
                [0.,1.,0.,1.],
                [0.,1.,0.,0.],
                [0.,0.,1.,0.],
                [1.,0.,0.,0.],
                # B
                [0.,0.,1.,1.],
                [1.,0.,0.,1.],
                [1.,1.,1.,0.],
                [1.,1.,1.,1.],
            ]
        )

        betas = np.logspace(-2, 5, num=30)

        results_ib = ba.IBOptimizer(
            pxy, 
            betas,
            num_restarts=1,
        ).get_results()

        alphas = np.array([1.]) # 0 <= alpha <= 1
        weights = np.ones(fx.shape[1]) # 4, just testing kwargs works

        results_ibmse = ba.IBMSEOptimizer(
            pxy, 
            fx,
            betas,
            alphas,
            num_restarts=1,
            weights=weights,
        ).get_results()

        for i, result_ib in enumerate(results_ib):
            result_ibmse = results_ibmse[i]

            # Make sure the same results were filtered out if at all
            if result_ib is not None and result_ibmse is None:
                raise Exception
            
            elif result_ibmse is not None and result_ib is None:
                raise Exception

            elif result_ib is None and result_ibmse is None:
                continue

            # encoder
            assert np.allclose(result_ib[0], result_ibmse[0])

            # Don't check the feature vectors, since reg ib doesn't have those

            # rate, 
            assert np.isclose(result_ib[-4], result_ibmse[-5])

            # distortion, 
            assert np.isclose(result_ib[-3], result_ibmse[-4])

            # accuracy
            assert np.isclose(result_ib[-2], result_ibmse[-3])

            # beta
            assert np.isclose(result_ib[-1], result_ibmse[-2])

            # Don't check alpha, since reg ib doesn't have

    def test_ba_binary_dist_deterministic(self):

        # Should be a trivial bound, since I[X:Xhat] = I[Xhat:Y]

        # Medin and Schaffer deterministic category labels
        py_x = np.array(
            [
                [0., 1.],
                [0., 1.],
                [0., 1.],
                [0., 1.],
                [0., 1.],
                [1., 0.],
                [1., 0.],
                [1., 0.],
                [1., 0.],
            ]
        )
        py_x /= py_x.sum(axis=1)[:, None]
        # get joint by multiplying by p(x)
        px = np.full(py_x.shape[0], 1/py_x.shape[0])
        pxy = py_x * px[:, None]

        fx = np.array(
            [
                # A
                [0.,0.,0.,1.],
                [0.,1.,0.,1.],
                [0.,1.,0.,0.],
                [0.,0.,1.,0.],
                [1.,0.,0.,0.],
                # B
                [0.,0.,1.,1.],
                [1.,0.,0.,1.],
                [1.,1.,1.,0.],
                [1.,1.,1.,1.],
            ]
        )

        betas = np.logspace(-2, 5, num=30)
        alphas = np.logspace(-2, 0., num=30) # 0 <= alpha <= 1

        weights = np.ones(fx.shape[1]) # 4, just testing kwargs works

        ba.IBMSEOptimizer(
            pxy, 
            fx,
            betas,
            alphas,
            num_restarts=1,
            weights=weights,
            ensure_monotonicity=True,
        ).get_results()

class TestPostProcessing:

    def test_compute_lower_bound(self):

        xs = list(range(10))
        ys = list(range(0, 20, 2))[::-1]

        # Insert nonmon        
        ys[5] = ys[5] + 3

        inputs = list(zip(xs,ys))

        actual = postprocessing.compute_lower_bound(inputs)
        expected = list(range(10))
        expected.pop(5)

        assert expected == actual
