import os

import numpy as np
import scipy as sp
import pandas as pd


if __name__ == '__main__':
    data_dir = '../data'
    alphas = np.load(os.path.join(data_dir, 'alphas.npy'))
    betas = np.load(os.path.join(data_dir, 'betas.npy'))
    rewards = np.load(os.path.join(data_dir, 'rewards.npy'))
    actions = np.load(os.path.join(data_dir, 'actions.npy'))

    n = actions.shape[1]
    m = actions.shape[-1]

    columns = ['alpha', 'beta', 'll']
    log_df = pd.DataFrame(columns=columns)
    output_dir = '../outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    lls = []
    for alpha, beta, rews, acts in zip(alphas, betas, rewards, actions):
        X = []
        Y = []
        G = np.vstack([[alpha * (1 - alpha) ** k * beta for k in range(n)] for _ in range(m)])
        for t in range(n):
            U = np.zeros((n, m))
            if t > 0:
                U[:t] = rews[:t][::-1]
            x = np.diag(G @ U)
            X.append(x)
            Y.append(acts[t])
        Y = np.vstack(Y)
        X = np.vstack(X)
        lls.append(np.sum(np.diag(Y @ X.T) - sp.special.logsumexp(X, axis=1)))

    log_df['alpha'] = alphas
    log_df['beta'] = betas
    log_df['ll'] = lls
    log_df.to_csv(os.path.join(output_dir, 'log_baseline.csv'), index=False)
