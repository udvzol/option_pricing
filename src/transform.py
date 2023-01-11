from abc import abstractmethod

import numpy as np
from scipy.stats import lognorm


class StateTransform:
    def __init__(self, rmin, rmax, bits, calc_inv=True):
        self.rmin = rmin
        self.rmax = rmax
        self.n = bits
        self.a = rmin
        self.b = (rmax - rmin) / (2 ** bits - 1)
        if calc_inv:
            x = -rmin / (rmax - rmin) * (2 ** bits - 1)
            self.inverse = StateTransform(x, 2 ** bits - 1 + x, np.log2(rmax - rmin + 1), calc_inv=False)

    def dec(self, decimal):
        decimal = (decimal - self.a) / self.b
        return decimal

    def dec_to_bin(self, decimal):
        """
        It is floored if not integer.
        :param decimal:
        :return:
        """
        r = []
        for i in range(self.n):
            r.append(int(decimal % 2))
            decimal //= 2
        return r[::-1]

    @abstractmethod
    def bin_to_dec(binary):
        return int(binary, 2)

    def state_dict(self):
        d = dict()
        for i in range(2 ** self.n):
            d["".join([str(k) for k in self.dec_to_bin(i)])] = i * self.b + self.a
        return d

    def get_coef(self):
        return self.a, self.b

    def twos_complement(self, decimal):
        b = self.dec_to_bin(decimal)
        b_c = [str(abs(i - 1)) for i in b]
        dec_c = StateTransform.bin_to_dec("".join(b_c)) + 1
        return self.dec_to_bin(dec_c)

    def ones_complement(self, decimal):
        b = self.dec_to_bin(decimal)
        b_c = [str(abs(i - 1)) for i in b]
        dec_c = StateTransform.bin_to_dec("".join(b_c))
        return self.dec_to_bin(dec_c)


def lognormal_distribution(S, t, r, vol):
    """

    :param S:
    :param t:
    :param K:
    :param r:
    :param vol:
    :return:
    """
    mu = (r - vol ** 2 / 2) * t + np.log(S)
    s = vol * np.sqrt(t)
    scale = np.exp(mu)

    # Distribution
    lnd = lognorm(s=s, scale=scale)
    return lnd


def prob_part(distribution, x):
    """
    :param distribution: scipy.stats
    :param n: number of qubits
    :return:
    """
    xmax = max(x)
    xmin = min(x)
    d = np.mean(np.diff(x)) / 2
    probs = []
    for i in range(len(x)):
        probs.append((distribution.cdf(x[i] + d) - distribution.cdf(x[i] - d))  # / (
                     # distribution.cdf(xmax + d) - distribution.cdf(xmin - d))
                     )
    return probs


def prob_to_price(prob, xmax, K, r, t, c=0.01):
    return (prob + c - 1 / 2) / c * (xmax - K) / 2 * np.exp(-r * t)
