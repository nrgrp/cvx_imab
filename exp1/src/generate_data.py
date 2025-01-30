import os

import numpy as np

from utils import Bandit


if __name__ == '__main__':
    repeats = 1000
    num_steps = 200
    p_r = np.array([0.9, 0.1])
    p_tr = 0.02
    max_beta = 5
    np.random.seed(10015)

    output_dir = '../data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    envr = Bandit(p_r, p_tr)

    alphas = []
    betas = []
    rewards = []  # (repeat, step, reward)
    actions = []  # (repeat, step, action)
    for rp in range(repeats):
        alpha = np.random.uniform()
        beta = max_beta * np.random.uniform()
        alphas.append(alpha)
        betas.append(beta)

        reward, action = envr.generate_data(alpha, beta, num_steps)
        action_vec = []
        for a in action:
            a_vec = np.zeros(envr.num_arms)
            a_vec[a] = 1
            action_vec.append(a_vec)

        rewards.append(reward)
        actions.append(action_vec)
    np.save(os.path.join(output_dir, 'alphas.npy'), alphas)
    np.save(os.path.join(output_dir, 'betas.npy'), betas)
    np.save(os.path.join(output_dir, 'rewards.npy'), rewards)
    np.save(os.path.join(output_dir, 'actions.npy'), actions)
