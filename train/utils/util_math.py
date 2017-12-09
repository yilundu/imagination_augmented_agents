import numpy as np


def softmax(x):
    if isinstance(x, np.ndarray):
        return np.exp(x) / np.sum(np.exp(x), axis = 0)
    elif isinstance(x, list) or isinstance(x, tuple):
        x = np.array(x)
        return np.exp(x) / np.sum(np.exp(x), axis = 0).tolist()
    else:
        raise NotImplementedError


# reference: https://github.com/alexis-jacq/Pytorch-Sketch-RNN/blob/master/sketch_rnn.py
def sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, greedy = False):
    if greedy:
        return mu_x, mu_y
    mean = [mu_x, mu_y]
    covariance = [
        [sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],
        [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]
    ]
    sample = np.random.multivariate_normal(mean, covariance, 1)
    return sample[0][0], sample[0][1]
