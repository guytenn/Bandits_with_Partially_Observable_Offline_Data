from env import LinearContextualBandit
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
from train import Job
from joblib import Parallel, delayed
from itertools import product

matplotlib.use('Agg')

def main(args):
    d = args['d']
    K = args['K']

    w = np.abs(np.random.rand(K, d)) / np.sqrt(d)
    # w = 0.0001*np.ones(d)
    # w[0] = 1
    # w = np.diag(w)

    env = LinearContextualBandit(w, args['noise'])
    M_global = np.random.rand(K, d, d) / np.sqrt(d)
    # M_global = np.repeat(np.eye(d)[np.newaxis, :, :], K, axis=0)

    T_vals = np.logspace(1, args['N'], args['n_vals']).astype(int)

    n_jobs = min(len(args['L_values']) * len(args['gamma_values']), args['max_jobs'])
    regret_tmp = []
    for t in T_vals:
        print(f'Starting {n_jobs} jobs for t={t}')
        job = Job(env=env, M=M_global, w=w, **args)
        regret_tmp.append(Parallel(n_jobs=n_jobs)(delayed(job.execute)(t, L, gamma)
                                                  for L, gamma in product(args['L_values'], args['gamma_values'])))

    regret = np.zeros((len(T_vals), len(args['L_values']), len(args['gamma_values'])))
    for tt in range(len(T_vals)):
        i = 0
        for ll, gg in product(range(len(args['L_values'])), range(len(args['gamma_values']))):
            regret[tt, ll, gg] = regret_tmp[tt][i]
            i += 1

    # best_regret = np.zeros((len(T_vals), len(args['L_values'])))
    # for ll in range(len(args['L_values'])):
    #     best_regret[:, ll] = regret[:, ll, np.argmin(regret[-1, ll, :])]

    for gg, gamma in enumerate(args['gamma_values']):
        fig, ax = plt.subplots(1)
        x_axis = np.repeat(T_vals[np.newaxis, :], regret[:, :, gg].shape[1], axis=0)
        ax.plot(x_axis.T, regret[:, :, gg])
        # ax.fill_between(t_vals, mean-std/2, mean+std/2, facecolor=colors[ll], alpha=0.1)
        plt.title(f'Regret, d={d}, gamma={gamma}')
        plt.legend([f'L={L}' for L in args['L_values']])
        # plt.show()
        plt.savefig(f'regret_gamma_{gamma}.png')

    save_data = {'data': regret, 'args': args}
    np.save('data.npy', save_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_vals", default=20, type=int)
    parser.add_argument("--N", default=2, type=int)
    parser.add_argument("--d", default=20, type=int)
    parser.add_argument("--K", default=5, type=int)
    parser.add_argument("--l", default=1, type=int)
    parser.add_argument("--n_seeds", default=5, type=int)
    parser.add_argument("--delta", default=0.001, type=float)
    parser.add_argument('--L_values', nargs='+', default=[0], type=int)
    parser.add_argument('--gamma_values', nargs='+', default=[0.2, 0.5, 1, 1.5, 5, 10], type=float)
    parser.add_argument('--max_jobs', default=20, type=int)
    parser.add_argument('--noise', dest='noise', choices=['uniform', 'bernoulli'], default='uniform')
    # parser.add_argument("--boolean", action="store_true")
    args = parser.parse_args().__dict__

    main(args)
