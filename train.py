import numpy as np
from SquareLinCBLC import SquareLinCBLC
from utils import calc_gamma
from time import time


class Job:
    def __init__(self, env, M, L, w, **kwargs):
        self.env = env
        self.L = L
        self.M = M[:, 0:L, :]
        self.w = w
        self.S = np.max(np.linalg.norm(w, axis=1))
        self.args = kwargs

        self.mu = self.args['K']

        self.b = np.zeros((self.args['K'], L))
        self.B = 0
        for a in range(self.args['K']):
            self.b[a] = self.M[a] @ self.w[a]
            self.B = np.max([self.B, self.S * np.linalg.norm(self.b[a])])

    def execute(self, T : int):
        start = time.time()
        print(f'Started Job: (T={T}, L={self.L})')
        args = self.args
        regret = 0
        gamma = calc_gamma(T, 1, self.B, args['K'], args['l'], args['d'], self.L, self.S, args['delta'])
        Algo = SquareLinCBLC(args['d'], args['K'], gamma, self.mu, args['l'])
        for _ in range(T):
            x = self.env.sample_x()
            y_hat = Algo.step(x, self.M, self.b)
            p = Algo.calc_p(y_hat)
            a = np.random.choice(range(args['K']), p=p)
            real_r, r = self.env.sample_r(x, a)
            Algo.update(x, a, r)
            regret += self.env.best_r(x) - real_r
        print(f'Done: (T={T}, L={self.L}, regret={regret}, time={time.time()-start}s)')
        return regret

