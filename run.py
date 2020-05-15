from env import LinearContextualBandit
import numpy as np
import matplotlib.pyplot as plt
import argparse
from train import Job
from joblib import Parallel, delayed


def main(args):
    d = args['d']
    K = args['K']

    w = np.abs(np.random.rand(K, d)) / np.sqrt(d)
    # w = 0.0001*np.ones(d)
    # w[0] = 1
    # w = np.diag(w)

    env = LinearContextualBandit(w)
    M_global = np.random.rand(K, d, d) / np.sqrt(d)
    # M_global = np.repeat(np.eye(d)[np.newaxis, :, :], K, axis=0)

    T_vals = np.logspace(1, args['N'], args['n_vals']).astype(int)
    n_jobs = min(len(T_vals), args['max_jobs'])
    regret = []
    for L in args['L_values']:
        job = Job(env=env, M=M_global, L=L, w=w, **args)
        regret.append(Parallel(n_jobs=n_jobs)(delayed(job.execute)(t) for t in T_vals))
    regret = np.array(regret)

    fig, ax = plt.subplots(1)
    x_axis = np.repeat(T_vals[np.newaxis, :], len(regret), axis=0)
    ax.plot(x_axis.T, regret.T)
    # ax.fill_between(t_vals, mean-std/2, mean+std/2, facecolor=colors[ll], alpha=0.1)
    plt.title(f'Regret, d={d}')
    plt.legend([f'L={L}' for L in args['L_values']])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_vals", default=20, type=int)
    parser.add_argument("--N", default=2, type=int)
    parser.add_argument("--d", default=20, type=int)
    parser.add_argument("--K", default=5, type=int)
    parser.add_argument("--l", default=1, type=int)
    parser.add_argument("--n_seeds", default=5, type=int)
    parser.add_argument("--delta", default=0.001, type=float)
    parser.add_argument('--L_values', '--list', nargs='+', default=[0], type=int)
    parser.add_argument('--max_jobs', default=20, type=int)
    # parser.add_argument("--boolean", action="store_true")
    args = parser.parse_args().__dict__

    main(args)
