import os

import numpy as np

from utils import Bandit


if __name__ == '__main__':
    repeats = 1000
    num_steps = 200
    p_r = np.array([0.30, 0.27, 0.95, 0.67, 0.69, 0.29, 0.42, 0.05, 0.73, 1.00])
    p_tr = 0.02
    max_alpha_r = 1
    max_beta_r = 10
    max_alpha_a = 1
    max_beta_a = 5
    np.random.seed(10015)

    m = p_r.shape[0]

    output_dir = '../data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    envr = Bandit(p_r, p_tr)

    alpha_rs = []
    beta_rs = []
    alpha_as = []
    beta_as = []
    rewards = []  # (repeat, step, reward)
    actions = []  # (repeat, step, action)
    for rp in range(repeats):
        alpha_r = max_alpha_r * np.random.uniform(size=m)
        beta_r = max_beta_r * np.random.uniform(size=m)
        alpha_a = max_alpha_a * np.random.uniform(size=m)
        beta_a = max_beta_a * np.random.uniform(size=m)
        alpha_rs.append(alpha_r)
        beta_rs.append(beta_r)
        alpha_as.append(alpha_a)
        beta_as.append(beta_a)

        reward, action = envr.generate_data(alpha_r, beta_r, alpha_a, beta_a, num_steps)
        action_vec = []
        for a in action:
            a_vec = np.zeros(envr.num_arms)
            a_vec[a] = 1
            action_vec.append(a_vec)

        rewards.append(reward)
        actions.append(action_vec)
    np.save(os.path.join(output_dir, 'alpha_rs.npy'), alpha_rs)
    np.save(os.path.join(output_dir, 'beta_rs.npy'), beta_rs)
    np.save(os.path.join(output_dir, 'alpha_as.npy'), alpha_as)
    np.save(os.path.join(output_dir, 'beta_as.npy'), beta_as)
    np.save(os.path.join(output_dir, 'rewards.npy'), rewards)
    np.save(os.path.join(output_dir, 'actions.npy'), actions)
