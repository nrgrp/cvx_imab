import os
import time
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy as sp


if __name__ == '__main__':
    repeats = 10
    max_beta = 5

    data_dir = '../data'
    alphas = np.load(os.path.join(data_dir, 'alphas.npy'))
    betas = np.load(os.path.join(data_dir, 'betas.npy'))
    rewards = np.load(os.path.join(data_dir, 'rewards.npy'))
    actions = np.load(os.path.join(data_dir, 'actions.npy'))

    n = actions.shape[1]
    m = actions.shape[-1]

    columns = ['alpha', 'beta', 'alpha_hat', 'beta_hat',
               'time', 'll']
    log_df = pd.DataFrame(columns=columns)
    output_dir = '../outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    alpha_hats = []
    beta_hats = []
    times = []
    lls = []
    for rews, acts in tqdm(zip(rewards, actions), total=len(rewards)):
        t = 0

        acts_idx = np.argmax(acts, axis=-1)
        def fn(params):
            alpha, beta = params
            qs = [np.zeros(m)]
            for r_idx, r in enumerate(rews[:-1]):
                qs.append((1 - alpha) * qs[r_idx] + alpha * beta * r)
            qs = np.array(qs)
            log_pi = qs - sp.special.logsumexp(qs, axis=-1, keepdims=True)
            ll = np.sum(log_pi[np.arange(acts_idx.shape[0] - 1), acts_idx[1:]])
            return -ll

        ls = []
        xs = []
        for i in range(repeats):
            alpha_init = np.random.uniform()
            beta_init = max_beta * np.random.uniform()
            constraints = [
                sp.optimize.LinearConstraint(np.array([1, 0]), 0, 1),
                sp.optimize.LinearConstraint(np.array([0, 1]), 0)
            ]

            start_t = time.time()
            prob = sp.optimize.minimize(fn, (alpha_init, beta_init), constraints=constraints, method='COBYLA')
            end_t = time.time()
            t += end_t - start_t

            ls.append(prob.fun)
            xs.append(prob.x)

        alpha_hat, beta_hat = xs[np.argmin(ls)]

        times.append(t)
        lls.append(-np.min(ls))
        alpha_hats.append(alpha_hat)
        beta_hats.append(beta_hat)

    log_df['alpha'] = alphas
    log_df['beta'] = betas
    log_df['alpha_hat'] = alpha_hats
    log_df['beta_hat'] = beta_hats
    log_df['time'] = times
    log_df['ll'] = lls
    log_df.to_csv(os.path.join(output_dir, 'log_direct.csv'), index=False)
