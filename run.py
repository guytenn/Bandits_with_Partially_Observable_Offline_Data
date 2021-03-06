from env import LinearContextualBandit
import numpy as np
import matplotlib.pyplot as plt
import argparse
from train import Trainer
from joblib import Parallel, delayed
from itertools import product
import os
import datetime
from data import DataManager
import yaml

# matplotlib.use('Agg')

def main(args):
    if args['seed'] >= 0:
        np.random.seed(args['seed'])

    d = args['d']
    K = args['K']
    T_vals = [10 ** args['N']]

    w = np.abs(np.random.rand(K, d)) / np.sqrt(d)

    env = LinearContextualBandit(w, sigma=args['sigma'], x_norm=args['x_normalization'])

    for data_size in args["data_sizes"]:
        if args['perturbations']:
            print(f'Creating Dataset of size N={data_size}')
            data_manager = DataManager(env, d, K, data_size)
        else:
            data_manager = None

        trainer = Trainer(env=env, w=w, **args)

        iters = [range(args['n_seeds']), args['L_values'], args['alpha_values']]

        n_jobs = min(args['n_seeds'] * len(args['L_values']) * len(args['alpha_values']), args['max_jobs'])
        regret_tmp = []
        for t in T_vals:
            print(f'Starting {n_jobs} jobs for t={t}')
            regret_tmp.append(Parallel(n_jobs=n_jobs)(delayed(trainer.execute)(t, L, alpha_l, data_manager=data_manager)
                                                      for seed, L, alpha_l in product(*iters)))

        # Gather results and save
        folder_name = datetime.datetime.now().__str__().replace(' ', '_')
        os.mkdir(folder_name)

        regret = np.zeros((args['n_seeds'], T_vals[0], len(args['L_values']), len(args['alpha_values'])))
        i = 0
        for seed, ll, gg in product(*[range(len(x)) for x in iters]):
            regret[seed, :, ll, gg] = regret_tmp[0][i]
            i += 1

        args['data_size'] = data_size
        save_data = {'data': regret, 'args': args}
        np.save(f'{folder_name}/data.npy', save_data)
        with open(f'{folder_name}/args.yml', 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)

        # Plot
        for aa, alpha_l in enumerate(args['alpha_values']):
            fig, ax = plt.subplots(1)
            x_axis = np.repeat(np.array(range(T_vals[0]))[np.newaxis, :], len(args['L_values']), axis=0)
            ax.plot(x_axis.T, np.mean(regret, axis=0)[:, :, aa])
            # ax.fill_between(t_vals, mean-std/2, mean+std/2, facecolor=colors[ll], alpha=0.1)
            plt.title(f'Regret, d={d}, alpha_l={alpha_l}')
            plt.legend([f'L={L}' for L in args['L_values']])
            # plt.show()
            plt.savefig(f'{folder_name}/regret_gamma_{alpha_l}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", default=5, type=int)
    parser.add_argument("--d", default=30, type=int)
    parser.add_argument("--K", default=30, type=int)
    parser.add_argument("--n_seeds", default=5, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--l", default=0.1, type=float)
    parser.add_argument("--delta", default=0.01, type=float)
    parser.add_argument('--L_values', nargs='+', default=[0, 5, 10, 15, 20, 25], type=int)
    parser.add_argument('--alpha_values', nargs='+', default=[0.01], type=float)
    parser.add_argument('--max_jobs', default=40, type=int)
    parser.add_argument("--sigma", default=1, type=float)
    parser.add_argument('--x_normalization', default=1, type=float)
    parser.add_argument("--perturbations", action="store_true")
    parser.add_argument('--data_sizes', nargs='+', default=[10000, 100000, 1000000], type=int)
    parser.add_argument('--calc_r12', action="store_true")
    parser.add_argument('--verbose', action="store_true")
    args = parser.parse_args().__dict__

    main(args)
