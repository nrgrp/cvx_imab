import os
import time
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import numpy as np
import cvxpy as cp
import scipy as sp
import pandas as pd

from utils import diag_op


if __name__ == '__main__':
    p = 5
    s2_repeats = 10
    max_beta = 5
    epsilon = 1e-5

    data_dir = '../data'
    alphas = np.load(os.path.join(data_dir, 'alphas.npy'))
    betas = np.load(os.path.join(data_dir, 'betas.npy'))
    rewards = np.load(os.path.join(data_dir, 'rewards.npy'))
    actions = np.load(os.path.join(data_dir, 'actions.npy'))

    n = actions.shape[1]
    m = actions.shape[-1]

    columns = ['alpha', 'beta', 'alpha_hat', 'beta_hat',
               's1_time', 's2_time', 's1_ll', 'll']
    log_df = pd.DataFrame(columns=columns)
    output_dir = '../outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    alpha_hats = []
    beta_hats = []
    s1_times = []
    s2_times = []
    s1_lls = []
    lls = []
    for rews, acts in tqdm(zip(rewards, actions), total=len(rewards)):
        ### step I ###
        g = cp.Variable(p)
        G = cp.vstack([g, g])

        X = []
        Y = []
        for t in range(n):
            if t < p:
                U = np.zeros((p, m))
                U[:t] = rews[:t][::-1]
            else:
                U = rews[t - p: t][::-1]
            x = diag_op(m) @ cp.vec(G @ U, order='C')
            X.append(x)
            Y.append(acts[t])
        Y = np.vstack(Y)
        X = cp.vstack(X)
        obj = cp.sum(cp.diag(X @ Y.T) - cp.log_sum_exp(X, axis=1))
        constraints = []
        constraints.append(g >= 0)
        constraints.append(cp.diff(g) <= 0)

        s1_prob = cp.Problem(cp.Maximize(obj), constraints)
        assert s1_prob.is_dcp()
        s1_prob.solve()
        s1_times.append(s1_prob.solver_stats.solve_time)
        s1_lls.append(s1_prob.value)

        ### step II ###
        s2_t = 0

        y = g.value
        def fn(params):
            alpha, beta = params
            return np.sum([(alpha * (1 - alpha) ** k * beta - y[k]) ** 2 for k in range(p)])

        s2_ls = []
        s2_xs = []
        tight = False
        for i in range(s2_repeats):
            alpha_init = np.random.uniform()
            beta_init = max_beta * np.random.uniform()
            constraints = [
                sp.optimize.LinearConstraint(np.array([1, 0]), 0, 1),
                sp.optimize.LinearConstraint(np.array([0, 1]), 0)
            ]

            start_t = time.time()
            s2_prob = sp.optimize.minimize(fn, (alpha_init, beta_init), constraints=constraints, method='COBYLA')
            end_t = time.time()
            s2_t += end_t - start_t

            loss = s2_prob.fun
            if np.abs(loss) < n * epsilon:
                alpha_hat, beta_hat = s2_prob.x
                tight = True
                break
            else:
                s2_ls.append(loss)
                s2_xs.append(s2_prob.x)
        if not tight:
            alpha_hat, beta_hat = s2_xs[np.argmin(s2_ls)]

        s2_times.append(s2_t)
        alpha_hats.append(alpha_hat)
        beta_hats.append(beta_hat)

        ### evaluate ###
        X = []
        Y = []
        G_hat = np.vstack([[alpha_hat * (1 - alpha_hat) ** k * beta_hat for k in range(p)] for _ in range(m)])
        for t in range(n):
            if t < p:
                U = np.zeros((p, m))
                U[:t] = rews[:t][::-1]
            else:
                U = rews[t - p: t][::-1]
            x = np.diag(G_hat @ U)
            X.append(x)
            Y.append(acts[t])
        Y = np.vstack(Y)
        X = np.vstack(X)
        lls.append(np.sum(np.diag(Y @ X.T) - sp.special.logsumexp(X, axis=1)))

    log_df['alpha'] = alphas
    log_df['beta'] = betas
    log_df['alpha_hat'] = alpha_hats
    log_df['beta_hat'] = beta_hats
    log_df['s1_time'] = s1_times
    log_df['s2_time'] = s2_times
    log_df['s1_ll'] = s1_lls
    log_df['ll'] = lls
    log_df.to_csv(os.path.join(output_dir, 'log_cvx_truc5.csv'), index=False)
