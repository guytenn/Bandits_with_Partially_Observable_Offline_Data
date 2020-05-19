import numpy as np
from SquareLinCBLC import SquareLinCBLC
from utils import calc_gamma
from time import time
from data import MbSampler, DataManager


class Trainer:
    def __init__(self, env, w, **kwargs):
        self.args = kwargs

        self.env = env
        self.d = self.args['d']
        self.K = self.args['K']
        if not self.args['perturbations']:
            self.M = np.random.rand(self.K, self.d, self.d) / np.sqrt(self.d)
        else:
            self.M = None
        self.w = w
        self.S = np.max(np.linalg.norm(w, axis=1))
        self.mu = self.K

    def execute(self, T, L, gamma_factor, data_manager: DataManager = None):
        start = time()
        print(f'Started Job: (T={T}, L={L}, gamma={gamma_factor})')

        if not self.args['perturbations']:
            M = self.M[:, 0:L, :]
            b = np.zeros((self.K, L))
            B = 0
            for a in range(self.K):
                b[a] = M[a] @ self.w[a]
                B = np.max([B, self.S * np.linalg.norm(b[a])])
            sampler = None
        else:
            sampler = MbSampler(data_manager, L, self.d, self.K)
            b = np.array(sampler.b)
            B = np.max(np.linalg.norm(b, axis=1))
            M = None

        args = self.args
        regret = 0
        gamma = gamma_factor * calc_gamma(T, 1, B, self.K, args['l'], self.d, L, self.S, args['delta'])
        # gamma = gamma_factor * np.sqrt(T)
        Algo = SquareLinCBLC(self.d, self.K, gamma, self.mu, args['l'])
        for _ in range(T):
            x = self.env.sample_x()
            if self.args['perturbations']:
                M, _ = sampler.step(x)
            y_hat = Algo.step(x, M, b)
            p = Algo.calc_p(y_hat)
            a = np.random.choice(range(self.K), p=p)
            real_r, r = self.env.sample_r(x, a)
            Algo.update(x, a, r)
            regret += self.env.best_r(x) - real_r
        elapsed_time = time()-start
        print(f'Done: (T={T}, L={L}, gamma={gamma_factor}, regret={regret}, time={elapsed_time}s, time_per_100_iter={100 * elapsed_time / T}s)')
        return regret