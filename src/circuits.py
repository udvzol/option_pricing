import sys

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from . import formulas
from .gate import CCRYGate


def comparator(qreg, t):
    n = len(qreg)
    areg = QuantumRegister(n, "ac")
    qc = QuantumCircuit(qreg, areg, name="comparator")
    if t[-1] == 1:
        qc.cx(qreg[0], areg[0])
    for i in range(1, n):
        # From the least significant
        if t[-i - 1] > 0:
            # OR - actually it exists in qiskit but I write it out
            qc.x(qreg[i])
            qc.x(areg[i - 1])
            qc.x(areg[i])
            qc.ccx(qreg[i], areg[i - 1], areg[i])
            qc.x(qreg[i])
            qc.x(areg[i - 1])
        else:
            qc.ccx(qreg[i], areg[i - 1], areg[i])
    return qc


def payoff_circuit(qreg, acmpreg, areg, f0, f1):
    qc = QuantumCircuit(qreg, acmpreg, areg, name="payoff")
    qc.cry(2 * f0, acmpreg[-1], areg[0])
    for i in range(len(qreg)):
        qc.append(CCRYGate(2 * 2 ** i * f1), [qreg[i], acmpreg[-1], areg[0]])
    return qc


def payoff_circuit_with_constant(qreg, acmpreg, areg, f0, f1, fp):
    qc = QuantumCircuit(qreg, acmpreg, areg, name="payoff")
    qc.ry(2 * fp, areg[0])
    qc.cry(2 * f0, acmpreg[-1], areg[0])
    for i in range(len(qreg)):
        qc.append(CCRYGate(2 * 2 ** i * f1), [qreg[i], acmpreg[-1], areg[0]])
    return qc


def payoff_circuit_fourier(qreg, acmpreg, areg, delta, xl, omega, beta, n):
    qc = QuantumCircuit(qreg, acmpreg, areg, name="payoff")
    qc.cry(n * omega * xl - beta, acmpreg[-1], areg[0])
    if n > 0:
        for i in range(len(qreg)):
            qc.append(CCRYGate(2 ** i * delta * n * omega), [qreg[i], acmpreg[-1], areg[0]])

    return qc


def initialize(n, probs):
    import numpy as np

    def left_prob(probs, theta):
        m = round(np.log2(len(probs)))
        a_left = sum(probs[:len(probs) // 2])
        a_right = 1 - a_left
        theta[m - 1].append(np.arccos(np.sqrt(a_left)) * 2)
        if len(probs) > 2:
            # Here the order is important
            left_prob(probs[:len(probs) // 2] / a_left, theta)
            left_prob(probs[len(probs) // 2:] / a_right, theta)

    def ccry(clist, data, thetalist, circuit, flip):
        # clist from the most significant
        if len(clist) > 0:
            # This is the transformation for the conditional rotation
            theta_left = (thetalist[:len(thetalist) // 2] + thetalist[len(thetalist) // 2:]) / 2
            theta_right = (thetalist[:len(thetalist) // 2] - thetalist[len(thetalist) // 2:]) / 2
            # If we apply the left and right part in a reverse order two CNOTs cancel. See Shende et al. page 11.
            if flip % 2 == 0:
                ccry(clist[1:], data, theta_left, circuit, flip + 2)
                circuit.cx(clist[0], data)
                ccry(clist[1:], data, theta_right, circuit, flip + 1)
                # If we are in an inner loop these will cancel out.
                if flip == 0:
                    circuit.cx(clist[0], data)
            else:
                # If we are in an inner loop this will cancel out with the one above.
                if flip == 0:
                    circuit.cx(clist[0], data)
                ccry(clist[1:], data, theta_right, circuit, flip + 1)
                circuit.cx(clist[0], data)
                ccry(clist[1:], data, theta_left, circuit, flip + 2)
        else:
            circuit.ry(thetalist[0], data)

    qreg_q = QuantumRegister(n, 'q')
    circuit = QuantumCircuit(qreg_q)

    theta = [[] for _ in range(n)]
    left_prob(probs, theta)

    for i in range(n):
        # Reverse order!!
        ccry(qreg_q[:n - i - 1:-1],
             qreg_q[n - i - 1],
             np.array(theta[n - i - 1]),
             circuit, flip=0)
    return circuit


class EuropeanCallOptionBase:
    def __init__(self, n, S, t, K, r, vol, w=4, sym_distribution=False, renorm_distribution=True):
        self.n = n
        self.S = S
        self.t = t
        self.K = K
        self.r = r
        self.vol = vol
        self.w = w
        self.st = None
        self.lnd = None
        self.x, self.probs = self.distribution(symmetric=sym_distribution, renorm=renorm_distribution)

        self.disc = np.exp(-self.r * self.t)

    def distribution(self, symmetric, renorm):
        from . import transform

        lnd = transform.lognormal_distribution(self.S, self.t, self.r, self.vol)

        xmin = max(lnd.mean() - self.w * lnd.std(), 0)
        if symmetric:
            xmax = lnd.mean() + self.w * lnd.std()
        else:
            xmax = lnd.mean() + self.w * lnd.std() - min(lnd.mean() - self.w * lnd.std(), 0)
        x = np.linspace(xmin, xmax, 2 ** self.n)

        probs = transform.prob_part(lnd, x)
        if renorm:
            probs = probs / sum(probs)

        self.st = transform.StateTransform(xmin, xmax, self.n)
        self.lnd = lnd
        return [x, probs]

    def BS_price(self):
        return formulas.BS(St=self.S, s=self.vol, K=self.K, r=self.r, t=self.t)

    def BS_var(self):
        return formulas.option_var(S=self.S, s=self.vol, K=self.K, r=self.r, t=self.t)

    def payoff(self, x):
        return (x > self.K).astype(int) * (x - self.K)

    def discrete_price(self):
        return sum(self.payoff(self.x) * self.probs) * self.disc

    def std_price(self):
        """Already scaled with leverage and discounted. This is really the std of price on the quantum computer."""
        prob = self.ideal_prob()
        return self.error_to_price(np.sqrt((1 - prob) * prob)) * self.disc


class EuropeanCallOptionFourier(EuropeanCallOptionBase):
    '''
    European call option class with Fourier series based function evalution technique. (arXiv:2105.09100v4)
    '''

    def __init__(self, n, S, t, K, r, vol, T=None, w=4, sym_distribution=False, renorm=True):
        super().__init__(n, S, t, K, r, vol, w, sym_distribution, renorm)
        self.xu = self.st.rmax - self.K
        if T:
            self.T = T
        else:
            self.T = 2 * self.xu
        if self.xu >= self.T:
            print("Linear part bigger", file=sys.stderr)

    def circuit(self, barriers=False, uncompute=False, fourier_n=None, fourier_beta=None):
        Kt = self.st.dec(self.K)
        Kb = self.st.ones_complement(min(max(Kt, 0), 2 ** self.n - 1))

        qri = QuantumRegister(self.n, "i")
        qra = QuantumRegister(self.n, "a_comparator")
        qra2 = QuantumRegister(1, "a_payoff")

        circuit = QuantumCircuit(qri, qra, qra2)
        initializer = initialize(self.n, self.probs)
        circuit.compose(initializer, inplace=True)

        if barriers:
            circuit.barrier()
        comparator_circuit = comparator(qri, Kb)
        circuit.compose(comparator_circuit, inplace=True)

        if barriers:
            circuit.barrier()

        qc = payoff_circuit_fourier(qri, qra, qra2, self.st.b, -self.K + self.st.rmin, 2 * np.pi / self.T, fourier_beta,
                                    fourier_n)
        # With int's created by numpy the type is numpy.int64 which is not handled correctly in compose.
        qubit_list = [i for i in range(2 * self.n)] + [self.n + self.n]
        circuit.compose(qc, qubits=qubit_list, inplace=True)

        if uncompute:
            circuit.compose(comparator_circuit.inverse(), inplace=True)

        return circuit

    def probs_to_price(self, p0, pcos, psin):
        '''
            pcos is where $\\beta$ is 0.
            psin is where $\\beta$ is $\pi/2$
        '''
        coefs = self.fourier_coeffs(len(pcos) + 1)
        cos_terms = p0 - 2 * pcos
        sin_term = p0 - 2 * psin
        return (coefs[0] @ cos_terms + coefs[1] @ sin_term + self.xu / 2 * p0) * self.disc

    def fourier_coeffs(self, kmax):
        a, b, c, d = formulas.cubic_coeff(self.xu, self.T)

        fcsin = lambda k: formulas.csin(a, b, c, d, k, self.xu, self.T)
        fccos = lambda k: formulas.ccos(a, b, c, d, k, self.xu, self.T)

        k = np.arange(1, kmax + 1)
        coeff_sin = np.array(list(map(fcsin, k)))
        coeff_cos = np.array(list(map(fccos, k)))

        return coeff_cos, coeff_sin

    def ideal_terms(self, kmax):
        cos_terms = np.array(
            [((self.probs * (self.x > self.K).astype(int)) @ np.cos(n * 2 * np.pi / self.T * (self.x - self.K))) for n
             in range(1, kmax + 1)])
        sin_terms = np.array(
            [((self.probs * (self.x > self.K).astype(int)) @ np.sin(n * 2 * np.pi / self.T * (self.x - self.K))) for n
             in range(1, kmax + 1)])
        zero_term = np.sum(self.probs * (self.x > self.K).astype(int))
        return zero_term, cos_terms, sin_terms

    def ideal_probs(self, kmax):
        "First all the cos than all the sin term probabilities"
        ideal_terms = self.ideal_terms(kmax)
        return ideal_terms[0], (ideal_terms[0] - ideal_terms[1]) / 2, (ideal_terms[0] - ideal_terms[2]) / 2

    def appr_payoff(self, x, kmax):
        coefs = self.fourier_coeffs(kmax)
        cos_terms = coefs[0] @ np.array([np.cos(n * 2 * np.pi / self.T * (x - self.K)) for n in range(1, kmax + 1)])
        sin_terms = coefs[1] @ np.array([np.sin(n * 2 * np.pi / self.T * (x - self.K)) for n in range(1, kmax + 1)])
        return (cos_terms + sin_terms + self.xu / 2) * (x > self.K).astype(int)

    def ideal_price(self, kmax):
        ideal_terms = self.ideal_terms(kmax)
        coefs = self.fourier_coeffs(kmax)
        return (coefs[0] @ ideal_terms[1] + coefs[1] @ ideal_terms[2] + self.xu / 2 * ideal_terms[0]) * self.disc


class EuropeanCallOption(EuropeanCallOptionBase):
    '''
        European call option class with rescaled function evalution technique. (Stamatopoulos et al., Quantum, 4, 5 2020)
    '''

    def __init__(self, n, S, t, K, r, vol, c=0.01, w=4, sym_distribution=False, renorm=True):
        self.c = c
        super().__init__(n, S, t, K, r, vol, w, sym_distribution, renorm)
        self.leverage = self.prob_to_price(1) - self.prob_to_price(0)

    def circuit(self, barriers=False, uncompute=False):
        Kt = self.st.dec(self.K)
        Kb = self.st.ones_complement(min(max(Kt, 0), 2 ** self.n - 1))

        # Rescaling
        c = self.c
        fp = np.pi / 4 - c
        f0 = -2 * c * Kt / (2 ** self.n - 1 - Kt)
        f1 = 2 * c / (2 ** self.n - 1 - Kt)

        qri = QuantumRegister(self.n, "i")
        qra = QuantumRegister(self.n, "a_comparator")
        qra2 = QuantumRegister(1, "a_payoff")

        circuit = QuantumCircuit(qri, qra, qra2)
        initializer = initialize(self.n, self.probs)
        circuit.compose(initializer, inplace=True)

        if barriers:
            circuit.barrier()
        comparator_circuit = comparator(qri, Kb)
        circuit.compose(comparator_circuit, inplace=True)

        if barriers:
            circuit.barrier()
        circuit.ry(2 * fp, qra2[0])
        qc = payoff_circuit(qri, qra, qra2, f0, f1)

        # With int's created by numpy the type is numpy.int64 which is not handled correctly in compose.
        qubit_list = [i for i in range(2 * self.n)] + [self.n + self.n]
        circuit.compose(qc, qubits=qubit_list, inplace=True)

        if uncompute:
            circuit.compose(comparator_circuit.inverse(), inplace=True)

        return circuit

    def prob_to_price(self, prob):
        return (prob + self.c - 1 / 2) / self.c * (self.st.rmax - self.K) / 2 * self.disc

    def price_to_prob(self, price):
        return 2 * self.c * (price / (self.st.rmax - self.K) / self.disc - 0.5) + 0.5

    def error_to_price(self, error):
        """error multiplied by leverage"""
        return error / self.c * (self.st.rmax - self.K) / 2 * self.disc

    def scaled_payoff(self, x):
        return self.c * ((x > self.K).astype(int) * 2 * (x - self.K) / (np.max(self.x) - self.K) - 1)

    def appr_payoff(self, x):
        return np.sin(self.scaled_payoff(
            x) + np.pi / 4) ** 2  # Jó lenne, ha így lenne, de nem: * (x > self.K).astype(int) + (0.5 - self.c) * (x < self.K).astype(int)

    def ideal_prob(self):
        return sum(self.appr_payoff(self.x) * self.probs)

    def ideal_price(self):
        """The same price the simulation of the circuit would give"""
        return self.prob_to_price(self.ideal_prob())
