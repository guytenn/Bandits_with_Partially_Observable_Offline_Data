import numpy as np
from SquareLinCBLC import SquareLinCBLC
from utils import calc_gamma
from time import time


class Job:
    def __init__(self, env, M, w, **kwargs):
        self.env = env
        self.M = M
        self.w = w
        self.S = np.max(np.linalg.norm(w, axis=1))
        self.args = kwargs

        self.mu = self.args['K']

    def execute(self, T, L, gamma_factor):
        start = time()
        print(f'Started Job: (T={T}, L={L}, gamma={gamma_factor})')

        M = self.M[:, 0:L, :]
        b = np.zeros((self.args['K'], L))
        B = 0
        for a in range(self.args['K']):
            b[a] = M[a] @ self.w[a]
            B = np.max([B, self.S * np.linalg.norm(b[a])])

        args = self.args
        regret = 0
        gamma = gamma_factor * calc_gamma(T, 1, B, args['K'], args['l'], args['d'], L, self.S, args['delta'])
        # gamma = gamma_factor * np.sqrt(T)
        Algo = SquareLinCBLC(args['d'], args['K'], gamma, self.mu, args['l'])
        for _ in range(T):
            x = self.env.sample_x()
            y_hat = Algo.step(x, M, b)
            p = Algo.calc_p(y_hat)
            a = np.random.choice(range(args['K']), p=p)
            real_r, r = self.env.sample_r(x, a)
            Algo.update(x, a, r)
            regret += self.env.best_r(x) - real_r
        elapsed_time = time()-start
        print(f'Done: (T={T}, L={L}, gamma={gamma_factor}, regret={regret}, time={elapsed_time}s, time_per_100_iter={100 * elapsed_time / T}s)')
        return regret

