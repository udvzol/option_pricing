from qiskit import QuantumRegister
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import RYGate


class CCRYGate(ControlledGate):
    '''
    Double controlled RY gate.
    '''

    def __init__(self, theta, label=None, ctrl_state=None):
        """Create new CCRY gate."""
        super().__init__('ccry', 3, [theta], num_ctrl_qubits=2, label=label,
                         ctrl_state=ctrl_state, base_gate=RYGate(theta))

    def _define(self):
        """
        gate cry(lambda) a,b,c
        { u3(lambda/4,0,0) c; cx a,c;
          u3(-lambda/4,0,0) c; cx b,c;
          u3(lambda/4,0,0) c; cx a,c;
          u3(-lambda/4,0,0) c; cx b,c;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from qiskit.circuit.library.standard_gates.u3 import U3Gate
        from qiskit.circuit.library.standard_gates.x import CXGate
        q = QuantumRegister(3, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (U3Gate(self.params[0] / 4, 0, 0), [q[2]], []),
            (CXGate(), [q[0], q[2]], []),
            (U3Gate(-self.params[0] / 4, 0, 0), [q[2]], []),
            (CXGate(), [q[1], q[2]], []),
            (U3Gate(self.params[0] / 4, 0, 0), [q[2]], []),
            (CXGate(), [q[0], q[2]], []),
            (U3Gate(-self.params[0] / 4, 0, 0), [q[2]], []),
            (CXGate(), [q[1], q[2]], [])
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc
