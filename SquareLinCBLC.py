import numpy as np
from utils import pinv, proj, I, safe_stack


class SquareLinCBLC:
    def __init__(self, d, K, gamma, mu, l=0.01):
        self.gamma = gamma
        self.mu = mu
        self.l = l
        self.K = K
        self.d = d

        self.ld = l * I(d)
        self.xxt = [np.zeros((d, d)) for _ in range(K)]
        self.Y = [np.zeros(d) for _ in range(K)]

    def step(self, x, M, b):
        y = np.ones(self.K) * np.inf
        if np.prod(M.shape) == 0:
            M = np.zeros((self.K, self.d, self.d))
            b = np.zeros((self.K, self.d))

        for a in range(self.K):
            Ma_inv = np.linalg.pinv(M[a])
            Pa = I(self.d) - Ma_inv @ M[a]
            Vt = self.xxt[a] + x[:, np.newaxis]@x[:, np.newaxis].T
            Ma_ba = Ma_inv @ b[a]
            Pw = np.linalg.pinv(Pa @ (self.ld + Vt) @ Pa) @ (self.Y[a] - Vt @ Ma_ba)
            y[a] = np.clip(x @ Pw + x @ Ma_ba, 0, 1)
        # print(y)
        return y

    def calc_p(self, y):
        b = np.argmax(y)
        p = np.zeros(self.K)
        curr_sum = 0
        for a in range(self.K):
            if a == b:
                continue
            p[a] = 1. / (self.mu + self.gamma*(y[b] - y[a]))
            curr_sum += p[a]
        p[b] = 1 - curr_sum
        return p

    def update(self, x, a, r):
        self.xxt[a] += x[:, np.newaxis] @ x[:, np.newaxis].T
        self.Y[a] += r * x






