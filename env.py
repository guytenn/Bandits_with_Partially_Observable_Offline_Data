import numpy as np
from utils import random_cov


class LinearContextualBandit:

    def __init__(self, w, sigma=1, x_norm=1):
        self.K = w.shape[0]
        self.d = w.shape[1]
        self.w = w
        self.x_norm = x_norm
        self.sigma = sigma

    def sample_x(self):
        x = np.random.randn(self.d)
        normalization = np.linalg.norm(x, 2)
        normalization *= self.x_norm
        return x / normalization

    def best_r(self, x):
        rmax = -np.inf
        for a in range(self.K):
            r = x @ self.w[a]
            if rmax < r:
                rmax = r
        return rmax

    def sample_r(self, x, a):
        real_r = x @ self.w[a]
        noise = self.sigma * np.random.randn()
        noisy_r = real_r + noise

        return real_r, noisy_r