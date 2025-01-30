import os

import numpy as np
import scipy as sp
import pandas as pd


if __name__ == '__main__':
    data_dir = '../data'
    alpha_rs = np.load(os.path.join(data_dir, 'alpha_rs.npy'))
    beta_rs = np.load(os.path.join(data_dir, 'beta_rs.npy'))
    alpha_as = np.load(os.path.join(data_dir, 'alpha_as.npy'))
    beta_as = np.load(os.path.join(data_dir, 'beta_as.npy'))
    rewards = np.load(os.path.join(data_dir, 'rewards.npy'))
    actions = np.load(os.path.join(data_dir, 'actions.npy'))

    n = actions.shape[1]
    m = actions.shape[-1]

    log_df = pd.DataFrame()
    output_dir = '../outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    lls = []
    for alpha_r, beta_r, alpha_a, beta_a, rews, acts in zip(alpha_rs, beta_rs, alpha_as, beta_as, rewards, actions):
        X = []
        Y = []
        G_r = np.array([alpha_r * (1 - alpha_r) ** k * beta_r for k in range(n)]).T
        G_a = np.array([alpha_a * (1 - alpha_a) ** k * beta_a for k in range(n)]).T
        for t in range(n):
            U_r = np.zeros((n, m))
            U_a = np.zeros((n, m))
            if t > 0:
                U_r[:t] = rews[:t][::-1]
                U_a[:t] = acts[:t][::-1]
            x = np.diag(G_r @ U_r) + np.diag(G_a @ U_a)
            X.append(x)
            Y.append(acts[t])
        Y = np.vstack(Y)
        X = np.vstack(X)
        lls.append(np.sum(np.diag(Y @ X.T) - sp.special.logsumexp(X, axis=1)))

    log_df['ll'] = lls
    log_df.to_csv(os.path.join(output_dir, 'log_baseline.csv'), index=False)
