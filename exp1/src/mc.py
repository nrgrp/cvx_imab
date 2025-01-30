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
    alphas = np.load(os.path.join(data_dir, 'alphas.npy'))
    betas = np.load(os.path.join(data_dir, 'betas.npy'))
    rewards = np.load(os.path.join(data_dir, 'rewards.npy'))
    actions = np.load(os.path.join(data_dir, 'actions.npy'))

    n = actions.shape[1]
    m = actions.shape[-1]

    mclog_df = pd.DataFrame()
    columns = ['alpha', 'beta', 'alpha_hat', 'beta_hat',
               'time', 'll']
    log_df = pd.DataFrame(columns=columns)
    output_dir = '../outputs'
    mclog_dir = os.path.join(output_dir, 'mc_val')
    fig_dir = os.path.join(mclog_dir, 'figs')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    alpha_hats = []
    beta_hats = []
    times = []
    lls = []
    for r_idx, (rews, acts) in enumerate(zip(rewards, actions)):
        acts_pt = pt.as_tensor_variable(np.argmax(acts, axis=1), dtype='int32')
        rews_pt = pt.as_tensor_variable(rews[:-1], dtype='float32')

        def get_ll(alpha):
            qs = pt.zeros((m,), dtype='float64')
            qs, _ = pytensor.scan(
                fn=lambda r, q, lr: pt.set_subtensor(q[:], q[:] + lr * (r - q[:])),
                sequences=[rews_pt], outputs_info=[qs], non_sequences=[alpha]
            )
            log_pi = qs - pt.logsumexp(qs, axis=-1, keepdims=True)
            ll = pt.sum(log_pi[pt.arange(acts_pt.shape[0] - 1), acts_pt[1:]])
            return ll

        with pm.Model() as model:
            alpha = pm.Uniform(name="alpha", lower=0, upper=1)
            beta = pm.HalfNormal(name="beta", sigma=beta_prior_halfnormal_sigma)
            rews_pt *= beta

            like = pm.Potential(name="like", var=get_ll(alpha))

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

        alpha_hat, beta_hat = mc_log['mean']
        alpha_hats.append(alpha_hat)
        beta_hats.append(beta_hat)

        ### evaluate ###
        X = []
        Y = []
        G_hat = np.vstack([[alpha_hat * (1 - alpha_hat) ** k * beta_hat for k in range(n)] for _ in range(m)])
        for t in range(n):
            U = np.zeros((n, m))
            if t > 0:
                U[:t] = rews[:t][::-1]
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
    log_df['time'] = times
    log_df['ll'] = lls
    log_df.to_csv(os.path.join(output_dir, 'log_mc.csv'), index=False)
