import numpy as np


class LinearContextualBandit:

    def __init__(self, w, noise):
        self.K = w.shape[0]
        self.d = w.shape[1]
        self.w = w
        self.noise = noise

    def sample_x(self):
        # x = np.abs(np.random.randn(self.d))
        x = 0.5 * (0.5 + np.random.rand(self.d)) / np.sqrt(self.d)
        # normalization = (3 * np.random.rand() + 1) * np.linalg.norm(x, 2)
        return x #/ normalization

    def best_r(self, x):
        rmax = -np.inf
        for a in range(self.K):
            r = x @ self.w[a]
            if rmax < r:
                rmax = r
        return rmax

    def sample_r(self, x, a):
        real_r = x @ self.w[a]
        const = np.min([np.abs(1 - real_r), real_r])
        if self.noise == 'uniform':
            noisy_r = real_r + 2 * const * (np.random.rand() - 0.5)
        elif self.noise == 'bernoulli':
            if (np.random.rand() > 0.5):
                noisy_r = real_r + const
            else:
                noisy_r = real_r - const
        else:
            raise ValueError(f'Unknown noise type: {self.noise}')

        return real_r, noisy_r