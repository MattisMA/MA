import numpy as np
from fedbatch_af_base import FedBatchAFBase
#log feed
class FedBatch4Reactor(FedBatchAFBase):
    feed_names = ["t_feed [min]", "r [-]", "tau_ref [min]"]
    feed_LOWER = [1.0,  0.01, 1.0]
    feed_UPPER = [1200.0, 1.0,  1000.0]

    def feed_coeffs(self, A, feed_par):
        T, r, tau = feed_par
        L = np.log1p(T / tau)
        f0 = A / (T - (1 - r) * ((tau + T) - T / L))
        b = f0 * (1 - r) / L
        return f0, b, tau

    def feed_rate(self, t, coeffs):
        f0, b, tau = coeffs
        return f0 - b * np.log1p(t / tau)