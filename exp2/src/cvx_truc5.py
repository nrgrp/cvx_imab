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
    rewards = np.load(os.path.join(data_dir, 'rewards.npy'))
    actions = np.load(os.path.join(data_dir, 'actions.npy'))

    n = actions.shape[1]
    m = actions.shape[-1]

    log_df = pd.DataFrame()
    output_dir = '../outputs'
    param_dir = os.path.join(output_dir, 'params_cvx_truc5')
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)

    alpha_r_hats = []
    alpha_a_hats = []
    beta_r_hats = []
    beta_a_hats = []
    s1_times = []
    s2_times = []
    s1_lls = []
    lls = []
    for rews, acts in tqdm(zip(rewards, actions), total=len(rewards)):
        ### step I ###
        G_r = cp.Variable((m, p))
        G_a = cp.Variable((m, p))

        X = []
        Y = []
        for t in range(n):
            if t < p:
                U_r = np.zeros((p, m))
                U_r[:t] = rews[:t][::-1]
                U_a = np.zeros((p, m))
                U_a[:t] = acts[:t][::-1]
            else:
                U_r = rews[t - p: t][::-1]
                U_a = acts[t - p: t][::-1]
            x = diag_op(m) @ (cp.vec(G_r @ U_r, order='C') + cp.vec(G_a @ U_a, order='C'))
            X.append(x)
            Y.append(acts[t])
        Y = np.vstack(Y)
        X = cp.vstack(X)
        obj = cp.sum(cp.diag(X @ Y.T) - cp.log_sum_exp(X, axis=1))
        constraints = []
        constraints.append(G_r[:, -1] >= 0)
        constraints.append(cp.diff(G_r, axis=1) <= 0)
        constraints.append(G_a[:, -1] >= 0)
        constraints.append(cp.diff(G_a, axis=1) <= 0)

        s1_prob = cp.Problem(cp.Maximize(obj), constraints)
        assert s1_prob.is_dcp()
        try:
            s1_prob.solve()
        except cp.error.SolverError:
            s1_times.append(np.nan)
            s1_lls.append(np.nan)
            s2_times.append(np.nan)
            lls.append(np.nan)
            alpha_r_hats.append(np.repeat(np.nan, m))
            beta_r_hats.append(np.repeat(np.nan, m))
            alpha_a_hats.append(np.repeat(np.nan, m))
            beta_a_hats.append(np.repeat(np.nan, m))
            continue
        s1_times.append(s1_prob.solver_stats.solve_time)
        s1_lls.append(s1_prob.value)

        ### step II ###
        s2_t = 0

        alpha_r_hat = []
        beta_r_hat = []
        alpha_a_hat = []
        beta_a_hat = []
        for G_idx, G in enumerate([G_r, G_a]):
            for y in G.value:
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
                    s2_prob = sp.optimize.minimize(fn, (alpha_init, beta_init),
                                                   constraints=constraints, method='COBYLA')
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
                if G_idx == 0:
                    alpha_r_hat.append(alpha_hat)
                    beta_r_hat.append(beta_hat)
                else:
                    alpha_a_hat.append(alpha_hat)
                    beta_a_hat.append(beta_hat)

        alpha_r_hat = np.array(alpha_r_hat)
        beta_r_hat = np.array(beta_r_hat)
        alpha_a_hat = np.array(alpha_a_hat)
        beta_a_hat = np.array(beta_a_hat)

        s2_times.append(s2_t)
        alpha_r_hats.append(alpha_r_hat)
        beta_r_hats.append(beta_r_hat)
        alpha_a_hats.append(alpha_a_hat)
        beta_a_hats.append(beta_a_hat)

        ### evaluate ###
        X = []
        Y = []
        G_r_hat = np.array([alpha_r_hat * (1 - alpha_r_hat) ** k * beta_r_hat for k in range(n)]).T
        G_a_hat = np.array([alpha_a_hat * (1 - alpha_a_hat) ** k * beta_a_hat for k in range(n)]).T

        for t in range(n):
            U_r = np.zeros((n, m))
            U_a = np.zeros((n, m))
            if t > 0:
                U_r[:t] = rews[:t][::-1]
                U_a[:t] = acts[:t][::-1]
            x = np.diag(G_r_hat @ U_r + G_a_hat @ U_a)
            X.append(x)
            Y.append(acts[t])
        Y = np.vstack(Y)
        X = np.vstack(X)
        lls.append(np.sum(np.diag(Y @ X.T) - sp.special.logsumexp(X, axis=1)))

    log_df['s1_time'] = s1_times
    log_df['s2_time'] = s2_times
    log_df['s1_ll'] = s1_lls
    log_df['ll'] = lls
    log_df.to_csv(os.path.join(output_dir, 'log_cvx_truc5.csv'), index=False)
    np.save(os.path.join(param_dir, 'alpha_r_hats.npy'), alpha_r_hats)
    np.save(os.path.join(param_dir, 'beta_r_hats.npy'), beta_r_hats)
    np.save(os.path.join(param_dir, 'alpha_a_hats.npy'), alpha_a_hats)
    np.save(os.path.join(param_dir, 'beta_a_hats.npy'), beta_a_hats)