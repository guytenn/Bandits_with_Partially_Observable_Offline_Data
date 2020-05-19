import numpy as np


def pinv(M):
    return M.T@np.linalg.pinv(M@M.T)


def proj(M):
    return pinv(M)@M


def I(d):
    return np.eye(d)


def safe_stack(x):
    if np.any(x):
        return np.stack(x)
    return None


def vovk_regret(l, u, B, Y, S, d, t):
    return (l/2)*u**2 + B**2 + ((d * Y**2) / 2) * np.log(1 + (t * S**2)/(l * d))


def calc_gamma(t, w, B, K, l, d, L, S, delta):
    return 2 * np.sqrt(t / (vovk_regret(l, w, B, 1, S, d - L, t) + (16. / K) * np.log(1/delta)))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()