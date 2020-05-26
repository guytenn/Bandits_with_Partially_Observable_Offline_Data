from env import LinearContextualBandit
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
from train import Trainer
from joblib import Parallel, delayed
from itertools import product
import os
import datetime
from data import DataManager

# matplotlib.use('Agg')


def main(args):
    if args['seed'] >= 0:
        np.random.seed(args['seed'])

    d = args['d']
    K = args['K']

    w = np.abs(np.random.rand(K, d)) / np.sqrt(d)

    env = LinearContextualBandit(w, args['noise'])

    if args['perturbations']:
        print(f'Creating Dataset of size N={args["data_size"]}')
        data_manager = DataManager(env, d, K, args['data_size'])
    else:
        data_manager = None

    trainer = Trainer(env=env, w=w, **args)

    if args['algo'] == 'oful':
        T_vals = [10 ** args['N']]
    elif args['algo'] == 'square':
        T_vals = np.logspace(1, args['N'], args['n_vals']).astype(int)
    else:
        raise ValueError(f'Unknown algorithm {args["algo"]}')

    iters = [range(args['n_seeds']), args['L_values'], args['gamma_values']]

    n_jobs = min(args['n_seeds'] * len(args['L_values']) * len(args['gamma_values']), args['max_jobs'])
    regret_tmp = []
    for t in T_vals:
        print(f'Starting {n_jobs} jobs for t={t}')
        regret_tmp.append(Parallel(n_jobs=n_jobs)(delayed(trainer.execute)(t, L, gamma, data_manager=data_manager)
                                                  for seed, L, gamma in product(*iters)))

    # Gather results and save
    folder_name = datetime.datetime.now().__str__()
    os.mkdir(folder_name)

    if args['algo'] == 'oful':
        regret = np.zeros((args['n_seeds'], T_vals[0], len(args['L_values']), len(args['gamma_values'])))
        i = 0
        for seed, ll, gg in product(*[range(len(x)) for x in iters]):
            regret[seed, :, ll, gg] = regret_tmp[0][i]
            i += 1
    elif args['algo'] == 'square':
        regret = np.zeros((args['n_seeds'], len(T_vals), len(args['L_values']), len(args['gamma_values'])))
        for tt in range(len(T_vals)):
            i = 0
            for seed, ll, gg in product(*[range(len(x)) for x in iters]):
                regret[seed, tt, ll, gg] = regret_tmp[tt][i]
                i += 1

    save_data = {'data': regret, 'args': args}
    np.save(f'{folder_name}/data.npy', save_data)

    # Plot
    for gg, gamma in enumerate(args['gamma_values']):
        fig, ax = plt.subplots(1)
        if args['algo'] == 'oful':
            x_axis = np.repeat(np.array(range(T_vals[0]))[np.newaxis, :], len(args['L_values']), axis=0)
        elif args['algo'] == 'square':
            x_axis = np.repeat(T_vals[np.newaxis, :], len(args['L_values']), axis=0)
        ax.plot(x_axis.T, np.mean(regret, axis=0)[:, :, gg])
        # ax.fill_between(t_vals, mean-std/2, mean+std/2, facecolor=colors[ll], alpha=0.1)
        plt.title(f'Regret, d={d}, gamma={gamma}')
        plt.legend([f'L={L}' for L in args['L_values']])
        # plt.show()
        plt.savefig(f'{folder_name}/regret_gamma_{gamma}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', dest='algo', choices=['oful', 'square'], default='oful')
    parser.add_argument("--n_vals", default=20, type=int)
    parser.add_argument("--N", default=3, type=int)
    parser.add_argument("--d", default=30, type=int)
    parser.add_argument("--K", default=5, type=int)
    parser.add_argument("--l", default=1, type=float)
    parser.add_argument("--n_seeds", default=1, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--delta", default=0.01, type=float)
    parser.add_argument('--L_values', nargs='+', default=[0, 5, 10], type=int)
    parser.add_argument('--gamma_values', nargs='+', default=[1], type=float)
    parser.add_argument('--max_jobs', default=20, type=int)
    parser.add_argument('--noise', dest='noise', choices=['uniform', 'bernoulli'], default='uniform')
    parser.add_argument("--perturbations", action="store_true")
    parser.add_argument('--data_size', default=1000, type=int)
    parser.add_argument('--calc_r12', action="store_true")
    args = parser.parse_args().__dict__

    main(args)
