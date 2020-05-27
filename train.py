import numpy as np
from SquareLinCB import SquareLinCB
from OFUL import OFUL
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

        args = self.args

        if not args['perturbations']:
            M = self.M[:, 0:L, :]
            b = np.zeros((self.K, L))
            B = 0
            for a in range(self.K):
                b[a] = M[a] @ self.w[a]
                B = np.max([B, self.S * np.linalg.norm(b[a])])
            sampler = None
        else:
            sampler = MbSampler(data_manager, L, self.d, self.K, args['calc_r12'])
            b = np.array(sampler.b)
            B = self.S * np.max(np.linalg.norm(b, axis=1))
            M = None
            elapsed_time = time() - start
            print(f'It took {elapsed_time}s to build dataset')
            start = time()


        regret = 0
        regret_vec = []

        if args['algo'] == 'oful':
            Algo = OFUL(self.d, self.K, L, gamma_factor, self.S, args['l'], args['delta'])
        elif args['algo'] == 'square':
            gamma = gamma_factor * calc_gamma(T, 1, B, self.K, args['l'], self.d, L, self.S, args['delta'])
            Algo = SquareLinCB(self.d, self.K, gamma, self.mu, args['l'])
        else:
            raise ValueError(f'Unknown algorithm {args["algo"]}')

        last_time = start
        for t in range(T):
            if self.args['verbose']:
                elapsed_time = time() - last_time
                if elapsed_time > 60:
                    last_time = time()
                    print(f'Mid-run: (t/T={t}/{T}, L={L}, gamma={gamma_factor}, regret={regret}, time={time() - start}s, time_per_100_iter={100 * elapsed_time / T}s)')
            x = self.env.sample_x()
            if self.args['perturbations']:
                M, _ = sampler.step(x)
            a = Algo.step(x, M, b)
            real_r, r = self.env.sample_r(x, a)
            Algo.update(x, a, r)
            regret += self.env.best_r(x) - real_r
            if args['algo'] == 'oful':
                regret_vec.append(regret)
        elapsed_time = time()-start
        print(f'Done: (T={T}, L={L}, gamma={gamma_factor}, regret={regret}, time={elapsed_time}s, time_per_100_iter={100 * elapsed_time / T}s)')
        if args['algo'] == 'oful':
            return regret_vec
        else:
            return regret