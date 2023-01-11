import numpy as np
from scipy.stats import norm


def BS(St, t, K, r, s):
    '''This function returns the option price.
    St-present stock value, K-strike price, t-time to maturity, r-risk free interest-rate, s-volatility '''
    return St * norm.cdf(d1(St, t, K, r, s)) - K * np.exp(-r * t) * norm.cdf(d2(St, t, K, r, s))


def d1(St, t, K, r, s):
    a = np.log(St / K) + (r + s ** 2 / 2) * t
    return a / s / np.sqrt(t)


def d2(St, t, K, r, s):
    return d1(St, t, K, r, s) - s * np.sqrt(t)


def d(S, K, t, r, s):
    return 1 / s / np.sqrt(t) * np.log(K * np.exp(-r * t) / S) + s / 2 * np.sqrt(t)


def option_var(S, K, t, r, s):
    C2 = S ** 2 * np.exp(s ** 2 * t) * norm.cdf(-d(S, K, t, r, s) + 2 * s * np.sqrt(t)) - 2 * K * S * np.exp(
        -r * t) * norm.cdf(-d(S, K, t, r, s) + s * np.sqrt(t)) + K ** 2 * np.exp(-2 * r * t) * norm.cdf(
        -d(S, K, t, r, s))
    return C2 - BS(S, t, K, r, s) ** 2


def cubic_coeff(xu, xu_):
    """Calculate the coefficients of the cubic polynomial that fits a linear at xu and xu_"""
    a = 2 * xu_ / ((xu_ ** 3 - xu ** 3) + 3 * (xu ** 2 * xu_ - xu * xu_ ** 2))
    b = -3 / 2 * a * (xu + xu_)
    c = -3 * a * xu ** 2 - 2 * b * xu + 1
    d = -a * xu_ ** 3 - b * xu_ ** 2 - c * xu_
    return a, b, c, d


def int_sin_x(a, x, order=0):
    '''
    Indefinite integrals of sin(ax)*x^(order)

    '''
    if order == 1:
        return np.sin(a * x) / a ** 2 - x * np.cos(a * x) / a

    elif order == 0:
        return -1 / a * np.cos(a * x)

    elif order == 2:
        return 2 * x / a ** 2 * np.sin(a * x) - (x ** 2 / a - 2 / a ** 3) * np.cos(a * x)

    elif order == 3:
        return (3 * x ** 2 / a ** 2 - 6 / a ** 4) * np.sin(a * x) - (x ** 3 / a - 6 * x / a ** 3) * np.cos(a * x)


def int_cos_x(a, x, order=0):
    '''
    Indefinite integrals of cos(ax)*x^(order)
    '''
    if order == 1:
        return np.cos(a * x) / a ** 2 + x * np.sin(a * x) / a

    elif order == 0:
        return 1 / a * np.sin(a * x)

    elif order == 2:
        return 2 * x / a ** 2 * np.cos(a * x) + (x ** 2 / a - 2 / a ** 3) * np.sin(a * x)

    elif order == 3:
        return (3 * x ** 2 / a ** 2 - 6 / a ** 4) * np.cos(a * x) + (x ** 3 / a - 6 * x / a ** 3) * np.sin(a * x)


def ccos(a, b, c, d, n, xu, T):
    w = n * 2 * np.pi / T
    # The definite integrals.
    # The linear part at zero written out explicitly.
    term_1 = int_cos_x(w, xu, 1) * (1 - c) + c * int_cos_x(w, T, 1) - 1 / w ** 2
    term_0 = d * (int_cos_x(w, T, 0) - int_cos_x(w, xu, 0))
    term_2 = b * (int_cos_x(w, T, 2) - int_cos_x(w, xu, 2))
    term_3 = a * (int_cos_x(w, T, 3) - int_cos_x(w, xu, 3))
    return 2 / T * (term_0 + term_1 + term_2 + term_3)


def csin(a, b, c, d, n, xu, T):
    w = n * 2 * np.pi / T
    term_1 = int_sin_x(w, xu, 1) * (1 - c) + c * int_sin_x(w, T, 1)
    term_0 = d * (int_sin_x(w, T, 0) - int_sin_x(w, xu, 0))
    term_2 = b * (int_sin_x(w, T, 2) - int_sin_x(w, xu, 2))
    term_3 = a * (int_sin_x(w, T, 3) - int_sin_x(w, xu, 3))
    return 2 / T * (term_0 + term_1 + term_2 + term_3)
