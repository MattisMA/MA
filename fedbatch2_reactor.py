import numpy as np
from fedbatch_af_base import FedBatchAFBase

#linear feed

class FedBatch2Reactor(FedBatchAFBase):
    feed_names = ["t_feed [min]", "r [-]"]
    feed_LOWER = [1, 0.001]
    feed_UPPER = [1200, 1.0]

    def feed_coeffs(self, A, feed_par):
        T, r = feed_par
        f0 = 2 * A / (T * (1 + r))
        m = f0 * (1 - r) / T
        return f0, m

    def feed_rate(self, t, coeffs):
        f0, m = coeffs
        return f0 - m * t
