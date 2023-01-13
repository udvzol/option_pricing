import numpy as np
from qiskit import Aer, transpile, QuantumCircuit
from qiskit import execute
from qiskit.circuit.library import GroverOperator
from scipy.optimize import curve_fit


def simulate_prob(circuit: QuantumCircuit, qubits, shots=1, *args, **kwargs):
    if "measure" in circuit.count_ops().keys():
        qubits = [circuit.get_instructions('measure')[i][1][0] for i in qubits]
    temp = circuit.remove_final_measurements(inplace=False)
    # temp.snapshot_probabilities("prob", qubits)
    temp.save_probabilities_dict(qubits=qubits)

    def simulate(circuit, shots=1024, *args, **kwargs):
        if "backend" in kwargs.keys():
            backend = kwargs.pop("backend")
        else:
            backend = Aer.get_backend('aer_simulator')
        job = execute(circuit, backend=backend, shots=shots, *args, **kwargs)
        # job_monitor(job)
        # plot_histogram(job.result().get_counts(circuit))
        return job.result()

    res = simulate(temp, shots, *args, **kwargs)
    return res.data()["probabilities"]


class QAError:
    """
    Calculate or simulate the error of the QAE algorithm
    """
    backend = Aer.get_backend('aer_simulator')

    def __init__(self, oracle, state_prep, good_state, basis_gates, zero_reflection=None, reflection_qubits=None,
                 device="CPU"):
        self.state_prep = state_prep
        self.good_state = good_state
        self.basis_gates = basis_gates
        self.device = device
        self.grover = GroverOperator(oracle=oracle, state_preparation=state_prep, zero_reflection=zero_reflection,
                                     reflection_qubits=reflection_qubits)
        self.grover_operator = transpile(self.grover, basis_gates=self.basis_gates)
        self.ideal_prob = simulate_prob(
            self.state_prep, self.good_state, noise_model=None, device=self.device, optimization_level=0).get(1, 0)

    def zero_reflection(registers, control_reg, target_reg, ancilla_reg, mode):
        circuit = QuantumCircuit(*registers)
        circuit.x(registers[control_reg])
        circuit.x(registers[target_reg])
        circuit.h(registers[target_reg])
        circuit.mcx(control_qubits=registers[control_reg], target_qubit=registers[target_reg],
                    ancilla_qubits=registers[ancilla_reg],
                    mode=mode)
        circuit.h(registers[target_reg])
        circuit.x(registers[control_reg])
        circuit.x(registers[target_reg])

        return circuit

    def circuit_sequence(self, klist):
        """
        Prepares the amplitude estimation circuits in the schedule klist.
        """
        circuits = []
        state_prep_temp = transpile(self.state_prep, basis_gates=self.basis_gates, optimization_level=3)
        for k in klist:
            circuits.append(state_prep_temp.compose(self.grover_operator.power(k), inplace=False))

        return circuits

    def fit_pcoh(self, noise_model, kmax, coupling_map=None):
        self.basis_gates = noise_model.basis_gates if noise_model else None

        def f(x, p0, pQ):
            return p0 * np.sin((2 * x + 1) * np.arcsin(np.sqrt(self.ideal_prob))) ** 2 * pQ ** x + (
                        1 - p0 * pQ ** x) * 0.5

        probs = []
        for k in range(kmax):
            circuit_temp = transpile(
                self.state_prep.compose(self.grover.power(k), inplace=False).measure_all(inplace=False),
                basis_gates=self.basis_gates, coupling_map=coupling_map, optimization_level=3)
            p = simulate_prob(
                circuit_temp, self.good_state, noise_model=noise_model,
                method="density_matrix", device=self.device, optimization_level=0).get(1, 0)
            probs.append(p)
        par, err = curve_fit(f, np.arange(kmax), probs, p0=[1, 0.9], bounds=(0, 1))
        return par, np.sqrt(np.diag(err)), probs

    def Fisher_I(mk, Nk, t, p, p0=1):
        n = 2 * mk + 1
        denum = np.power(p, -2 * mk, dtype=np.longdouble) / p0 ** 2 - np.cos(
            2 * n * t) ** 2  # exp(2*k*mk)-cos(2*n*t)**2

        I11 = sum(Nk * n ** 2 / np.sin(2 * t) ** 2 * 4 * np.sin(2 * n * t) ** 2 / denum)
        I12 = sum(Nk * mk * n / np.sin(2 * t) * np.sin(4 * n * t) / denum)
        I22 = sum(Nk * mk ** 2 * np.cos(2 * n * t) ** 2 / denum)
        m = np.array([[I11, I12], [I12, I22]])
        return m

    def est_error(mk, Nk, t, p, p0=1):
        '''
        The Cramér-Rao bound on variance of the estimated probability. (Tanaka et al., Quantum Information Processing, 20(9):3–14, 2021.)
        '''
        return np.sqrt(np.linalg.inv(QAError.Fisher_I(mk, Nk, t, p, p0).astype(np.float64))[0, 0])


class CRCalculator:
    '''
    Cramér-Rao error bound calculator base class for different schedules.
    '''

    def __init__(self, N, t0):
        self.N = N
        self.t0 = t0
        self.p0 = np.sin(t0) ** 2
        self.pcoh = None
        self.thr = None
        self.schedule = None
        self.Ntot = None

    def set_t0(self, t0):
        self.t0 = t0
        self.p0 = np.sin(t0) ** 2

    def set_pcoh(self, pcoh):
        self.pcoh = pcoh
        self.thr = 1 / (1 - self.pcoh) / 2 - 0.5 if pcoh < 1 else np.inf
        self.Nthr = self.N * np.sum(2 * self.make_schedule(self.schedule_len(self.thr)) + 1) if pcoh < 1 else np.inf

    def set_schedule(self, schedule=None, length=None, mkmax=None):
        if schedule:
            self.schedule = schedule
        elif length:
            self.schedule = self.make_schedule(length)
        elif mkmax:
            self.schedule = self.make_schedule(self.schedule_len(mkmax))
        else:
            self.set_schedule(mkmax=self.thr)

        self.Ntot = self.N * np.cumsum(2 * self.schedule + 1)

    def pcoh_for_thr(self, mkmax):
        return 1 - 1 / (2 * mkmax + 1)

    def get_errors(self, idx=None):
        if idx:
            return np.array([QAError.est_error(self.schedule[:i], self.N, self.t0, self.pcoh) for i in
                             idx])
        else:
            return np.array([QAError.est_error(self.schedule[:i], self.N, self.t0, self.pcoh) for i in
                             range(2, len(self.schedule) + 1)])

    def Fisher(self, idx=None):
        if idx:
            return np.array([QAError.Fisher_I(self.schedule[:i], self.N, self.t0, self.pcoh) for i in idx])
        else:
            return np.array([QAError.Fisher_I(self.schedule[:i], self.N, self.t0, self.pcoh) for i in
                             range(1, len(self.schedule) + 1)])


class CRCalculatorLinear(CRCalculator):
    '''
    Cramér-Rao error bound calculator for linear schedule
    '''

    def __init__(self, N=1, t0=0.7):
        super().__init__(N, t0)
        # for the asymptotic scaling of the std.
        # The scaling N->\infty is: pref*s0/N^power
        self.power = 3 / 4
        self.pref = np.sqrt(3 / 4)
        self.name = "linear"

    def make_schedule(self, length):
        return np.arange(length)

    def schedule_len(self, mkmax):
        return int(mkmax) + 1


class CRCalculatorExponential(CRCalculator):
    '''
        Cramér-Rao error bound calculator for exponential schedule
    '''

    def __init__(self, N=1, t0=0.7):
        super().__init__(N, t0)
        # for the asymptotic scaling of the std.
        self.power = 1
        self.pref = np.sqrt(3)
        self.name = "exponential"

    def make_schedule(self, length):
        return np.insert(2 ** np.arange(length - 1), 0, 0)

    def schedule_len(self, mkmax):
        if mkmax < 1:
            return 1
        else:
            return int(np.log2(mkmax)) + 2

    def Ntot_to_mkmax(self, Ntot):
        """I have this approximate solution for the inverse.
        The real solution contains lambert W function but is uncomputable for Ntot > 50"""
        if Ntot > 1:
            return 2 ** (int(np.log2(Ntot)) - 2)
        elif Ntot == 1:
            return 0
        else:
            return np.nan


class CRCalculatorQuadratic(CRCalculator):
    '''
        Cramér-Rao error bound calculator for quadratic schedule
    '''

    def __init__(self, N=1, t0=0.7):
        super().__init__(N, t0)
        # for the asymptotic scaling of the std.
        self.power = 5 / 6
        self.pref = np.sqrt(15 / 12 * (2 / 3) ** (5 / 3))
        self.name = "quadratic"

    def make_schedule(self, length):
        return np.arange(length) ** 2

    def schedule_len(self, mkmax):
        return int(np.sqrt(mkmax)) + 1
