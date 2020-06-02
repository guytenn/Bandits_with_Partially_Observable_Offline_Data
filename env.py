import numpy as np
from utils import random_cov


class LinearContextualBandit:

    def __init__(self, w, sigma=1, x_noise=None, T=None, x_norm=1):
        self.K = w.shape[0]
        self.d = w.shape[1]
        self.w = w
        self.x_noise = x_noise
        self.T = T
        self.x_norm = x_norm
        self.sigma = sigma
        if x_noise == 'correlated':
            self.cov = random_cov(np.random.rand(self.d))
            self.mean = 0.01 * (np.random.rand(self.d) - 0.5)

    def sample_x(self):
        # x = np.abs(np.random.randn(self.d))
        # x = 0.5 * (0.5 + np.random.rand(self.d))
        if self.x_noise == 'correlated':
            x = np.random.multivariate_normal(self.mean, self.cov)
        else:
            x = np.random.randn(self.d)
        normalization = np.linalg.norm(x, 2)
        normalization *= self.x_norm
        if self.T:
            normalization *= np.sqrt(self.T)
        # normalization = (3 * np.random.rand() + 1) * np.linalg.norm(x, 2)
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