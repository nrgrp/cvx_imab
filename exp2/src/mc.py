import os
import time

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import pymc as pm
import pytensor
import pytensor.tensor as pt
import arviz as az


if __name__ == '__main__':
    draws = 5000
    tune = 2000
    beta_prior_halfnormal_sigma = 2

    data_dir = '../data'
    rewards = np.load(os.path.join(data_dir, 'rewards.npy'))
    actions = np.load(os.path.join(data_dir, 'actions.npy'))

    n = actions.shape[1]
    m = actions.shape[-1]

    mclog_df = pd.DataFrame()
    log_df = pd.DataFrame()
    output_dir = '../outputs'
    mclog_dir = os.path.join(output_dir, 'mc_val')
    fig_dir = os.path.join(mclog_dir, 'figs')
    param_dir = os.path.join(output_dir, 'params_mc')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)

    alpha_r_hats = []
    alpha_a_hats = []
    beta_r_hats = []
    beta_a_hats = []
    times = []
    lls = []
    for r_idx, (rews, acts) in enumerate(zip(rewards, actions)):
        ys_pt = pt.as_tensor_variable(np.argmax(acts, axis=1), dtype='int32')
        acts_pt = pt.as_tensor_variable(acts[:-1], dtype='int32')
        rews_pt = pt.as_tensor_variable(rews[:-1], dtype='float32')

        def get_ll(alpha_r, alpha_a):
            q_rs = pt.zeros((m,), dtype='float64')
            q_rs, _ = pytensor.scan(
                fn=lambda r, q, lr: pt.set_subtensor(q[:], q[:] + lr * (r - q[:])),
                sequences=[rews_pt], outputs_info=[q_rs], non_sequences=[alpha_r]
            )
            q_as = pt.zeros((m,), dtype='float64')
            q_as, _ = pytensor.scan(
                fn=lambda r, q, lr: pt.set_subtensor(q[:], q[:] + lr * (r - q[:])),
                sequences=[acts_pt], outputs_info=[q_as], non_sequences=[alpha_a]
            )
            qs = q_rs + q_as
            log_pi = qs - pt.logsumexp(qs, axis=-1, keepdims=True)
            ll = pt.sum(log_pi[pt.arange(ys_pt.shape[0] - 1), ys_pt[1:]])
            return ll

        with pm.Model() as model:
            alpha_r = pm.Uniform(name="alpha_r", lower=0, upper=1, size=m)
            beta_r = pm.HalfNormal(name="beta_r", sigma=beta_prior_halfnormal_sigma, size=m)
            alpha_a = pm.Uniform(name="alpha_a", lower=0, upper=1, size=m)
            beta_a = pm.HalfNormal(name="beta_a", sigma=beta_prior_halfnormal_sigma, size=m)
            rews_pt *= beta_r
            acts_pt *= beta_a

            like = pm.Potential(name="like", var=get_ll(alpha_r, alpha_a))

            start_time = time.time()
            tr = pm.sample(draws=draws, tune=tune, nuts_sampler='numpyro')
            end_time = time.time()
        times.append(end_time - start_time)

        mc_log = az.summary(tr)
        mc_log['repeat'] = r_idx
        mc_log = mc_log.reset_index(names='param')
        mclog_df = pd.concat((mclog_df, mc_log))
        mclog_df.to_csv(os.path.join(mclog_dir, 'mc_log.csv'), index=False)

        az.plot_trace(tr)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'trace_{r_idx}.png'))
        plt.close()

        az.plot_posterior(tr)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'posterior_{r_idx}.png'))
        plt.close()

        mc_log = mc_log.set_index('param')
        alpha_r_hat = np.array([mc_log.loc[f'alpha_r[{k}]', 'mean'] for k in range(m)])
        beta_r_hat = np.array([mc_log.loc[f'beta_r[{k}]', 'mean'] for k in range(m)])
        alpha_a_hat = np.array([mc_log.loc[f'alpha_a[{k}]', 'mean'] for k in range(m)])
        beta_a_hat = np.array([mc_log.loc[f'beta_a[{k}]', 'mean'] for k in range(m)])
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

    log_df['time'] = times
    log_df['ll'] = lls
    log_df.to_csv(os.path.join(output_dir, 'log_mc.csv'), index=False)
    np.save(os.path.join(param_dir, 'alpha_r_hats.npy'), alpha_r_hats)
    np.save(os.path.join(param_dir, 'beta_r_hats.npy'), beta_r_hats)
    np.save(os.path.join(param_dir, 'alpha_a_hats.npy'), alpha_a_hats)
    np.save(os.path.join(param_dir, 'beta_a_hats.npy'), beta_a_hats)