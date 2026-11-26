import numpy as np
from fedbatch_af_base import FedBatchAFBase

#exponential feed
class FedBatch3Reactor(FedBatchAFBase):
    feed_names = ["t_feed [min]", "r [-]"]
    feed_LOWER = [1.0,  0.01]
    feed_UPPER = [1200.0, 1.0]

    def feed_coeffs(self, A, feed_par):
        T, r = feed_par
        if r > 1.0 - 1e-9:                     #limit r -> 1: constant feed, avoids 0/0
            return A / T, 0.0
        k = -np.log(r) / T
        f0 = A * k / (1 - r)
        return f0, k

    def feed_rate(self, t, coeffs):
        f0, k = coeffs
        return f0 * np.exp(-k * t)
