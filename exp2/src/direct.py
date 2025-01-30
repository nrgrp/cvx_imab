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
    rewards = np.load(os.path.join(data_dir, 'rewards.npy'))
    actions = np.load(os.path.join(data_dir, 'actions.npy'))

    n = actions.shape[1]
    m = actions.shape[-1]

    log_df = pd.DataFrame()
    output_dir = '../outputs'
    param_dir = os.path.join(output_dir, 'params_direct')
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)

    alpha_r_hats = []
    beta_r_hats = []
    alpha_a_hats = []
    beta_a_hats = []
    times = []
    lls = []
    for rews, acts in tqdm(zip(rewards, actions), total=len(rewards)):
        t = 0

        acts_idx = np.argmax(acts, axis=-1)
        def fn(params):
            alpha_r = params[:m]
            alpha_a = params[m: 2 * m]
            beta_r = params[2 * m: 3 * m]
            beta_a = params[-m:]
            q_rs = [np.zeros(m)]
            q_as = [np.zeros(m)]
            for idx, (r, a) in enumerate(zip(rews[:-1], acts[:-1])):
                q_rs.append((1 - alpha_r) * q_rs[idx] + alpha_r * beta_r * r)
                q_as.append((1 - alpha_a) * q_as[idx] + alpha_a * beta_a * a)

            q_rs = np.array(q_rs)
            q_as = np.array(q_as)
            qs = q_rs + q_as
            log_pi = qs - sp.special.logsumexp(qs, axis=-1, keepdims=True)
            ll = np.sum(log_pi[np.arange(acts_idx.shape[0] - 1), acts_idx[1:]])
            return -ll

        ls = []
        xs = []
        for i in range(repeats):
            x0 = np.random.uniform(size=4 * m)
            constraints = [
                sp.optimize.LinearConstraint(np.diag(np.hstack((np.ones(2 * m), np.zeros(2 * m)))), 0, 1),
                sp.optimize.LinearConstraint(np.diag(np.hstack((np.zeros(2 * m), np.ones(2 * m)))), 0)
            ]

            start_t = time.time()
            prob = sp.optimize.minimize(fn, x0, constraints=constraints, method='COBYLA')
            end_t = time.time()
            t += end_t - start_t

            ls.append(prob.fun)
            xs.append(prob.x)

        x = xs[np.argmin(ls)]
        alpha_r_hat = x[:m]
        alpha_a_hat = x[m: 2 * m]
        beta_r_hat = x[2 * m: 3 * m]
        beta_a_hat = x[-m:]

        times.append(t)
        lls.append(-np.min(ls))
        alpha_r_hats.append(alpha_r_hat)
        beta_r_hats.append(beta_r_hat)
        alpha_a_hats.append(alpha_a_hat)
        beta_a_hats.append(beta_a_hat)

    log_df['time'] = times
    log_df['ll'] = lls
    log_df.to_csv(os.path.join(output_dir, 'log_direct.csv'), index=False)
    np.save(os.path.join(param_dir, 'alpha_r_hats.npy'), alpha_r_hats)
    np.save(os.path.join(param_dir, 'beta_r_hats.npy'), beta_r_hats)
    np.save(os.path.join(param_dir, 'alpha_a_hats.npy'), alpha_a_hats)
    np.save(os.path.join(param_dir, 'beta_a_hats.npy'), beta_a_hats)