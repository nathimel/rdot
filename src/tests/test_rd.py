import numpy as np
from rdot import distortion, ba_basic, information


# The following test cases were taken from the following file in Alon Kipnis' repo: https://github.com/alonkipnis/BlahutArimoto/blob/master/example.py

class TestRDBinaryHamming:

    """Binary random variable with hamming distortion"""

    def test_compute_rate(self):

        # Mutual info between X, Y is bounded from above by the entropy of the variable with the smaller alphabet, i.e.
        # I(X;Y) <= log(min(|X|, |Y|))
        px = np.array([.5, .5,])
        pxhat_x = np.eye(2)
        rate = information.information_rate(
            px,
            pxhat_x
        )
        upper_bound = np.log2(2)
        assert rate == upper_bound

    def test_compute_distortion(self):
        # Simple uniform prior and alphabet = 4
        x = np.arange(4)
        dist_mat = distortion.hamming(*np.meshgrid(x, x))
        px = np.ones(4)/4
        pxhat_x = np.full((4,4), 1/4)
        expected = 0.75
        actual = distortion.expected_distortion(px, pxhat_x, dist_mat)
        assert expected == actual

    def test_ba_beta_0(self):
        
        x = np.array([0,1]) # Binary input
        xhat = np.array([0,1]) # Binary reconstruction
        p = 0.5 # P(X=1) = p
        px = np.array([1-p, p])
        
        # distortion matrix
        dist_mat = distortion.hamming(*np.meshgrid(x, xhat))
        rate, dist = ba_basic.blahut_arimoto(
            px=px,
            dist_mat=dist_mat,            
            beta=0., # evaluation beta
        )

        # The true R(D) function is bounded by H(p) - H(D); see Cover and Thomas, Rate Distortion Theory, Eq. (10.23).
        true_rate = information.H(p) - information.H(dist)

        # Should it be equal?
        assert rate >= true_rate

        # degenerate
        assert dist == 0.5


    def test_ba_beta_1e10(self):

        x = np.array([0,1]) # Binary input
        xhat = np.array([0,1]) # Binary reconstruction
        p = 0.5 # P(X=1) = p
        px = np.array([1-p, p])
        
        # distortion matrix
        dist_mat = distortion.hamming(*np.meshgrid(x, xhat))
        rate, dist = ba_basic.blahut_arimoto(
            px=px,
            dist_mat=dist_mat,            
            beta=1e10, # evaluation beta
        )

        # The true R(D) function is bounded by H(p) - H(D); see Cover and Thomas, Rate Distortion Theory, Eq. (10.23).
        true_rate = information.H(p) - information.H(dist)

        # Should it be equal?
        assert rate >= true_rate

        # deterministic
        assert dist == 0.

    def test_curve(self):

        x = np.array([0,1]) # Binary input
        xhat = np.array([0,1]) # Binary reconstruction
        p = 0.5 # P(X=1) = p
        px = np.array([1-p, p])
        
        # distortion matrix
        dist_mat = distortion.hamming(*np.meshgrid(x, xhat))        

        # Test many values of beta to sweep out a curve. 
        betas = np.logspace(-5, 5, num=100)        

        rd_values = ba_basic.ba_iterate(px, dist_mat, betas)

        # Check for convexity
        ind1 = 20
        ind2 = 30
        ind3 = 40
        
        # R, D points
        x1, y1 = rd_values[ind1]
        x2, y2 = rd_values[ind2]
        x3, y3 = rd_values[ind3]

        assert x1 < x2
        assert x2 < x3

        assert y1 > y2
        assert y2 > y3

        # The more general version of this test would check that all points on the curve satisfy the definition of convexity for a function


class TestRDGaussianQuadratic:

    """Gaussian random variable with quadratic distortion"""

    def test_compute_rate(self):
        pass

    def test_compute_distortion(self):
        pass

    def test_ba(self):
        pass
