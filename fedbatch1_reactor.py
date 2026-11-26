import numpy as np
from fedbatch_af_base import FedBatchAFBase

#constnt feed

class FedBatch1Reactor(FedBatchAFBase):
    feed_names = ["t_feed [min]"]
    feed_LOWER = [1.0]
    feed_UPPER = [1200.0]

    def feed_coeffs(self, A, feed_par):
        return (A/feed_par[0],)

    def feed_rate(self, t, coeffs):
        return coeffs[0]
        

