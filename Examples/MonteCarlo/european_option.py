# import useful libraries

from scipy.stats import norm
import numpy as np
from numpy import sin, cos
import operator


# import internal libraries
import quasar
from qcware import forge
from qcware.forge import qio, qutils
from qcware.forge.montecarlo.nisqAE import make_schedule, run_schedule, compute_mle


class EuropeanOption:
    """Quantum European Option pricing class
    .
    Init args:
        n_asset_prices : Number of different Asset Prices generated
        initial_asset_price : Initial Asset Price
        strike : Strike
        rate : interest rate
        sigma : Volatility
        expiry : Time to expiry
        option_type : 'C' = European Call Option, 'P' = European Put Option
        epsilon : precision parameter for the estimate
        mode : 'parallel' = optimized depth, 'sequential' = optimized number of qubits

    Outputs:
        Circuits used for pricing
            self.probability_loader_circuit :
            self.initial_circuit : 'oracle' circuit A
            self.iteration_circuit : iteration circuit (S_chi A^{inv} S_0 A) used for AE

        self.oracle_calls : number of calls to the oracle (pricing circuit)
        self.shots : number of shots run
        self.theta : estimate of angle theta
        self.expectation : estimate of expectation of sin(theta)**2
        self.price : final estimated price

        self.schedule : schedule used for AE
        self.schedule_type : scheduleType
        self.max_depth : maxDepth of circuit
        self.beta : beta parameter for power-law schedule
        self.n_shots : number of shots for each circuit in the schedule


    """

    def __init__(
        self,
        n_asset_prices,
        initial_asset_price,
        strike,
        rate,
        sigma,
        expiry,
        option_type,
        epsilon=0.005,
        mode="sequential",
    ):

        # initializing classical parameters
        self.n_asset_prices = n_asset_prices
        self.initial_asset_price = initial_asset_price
        self.strike = strike
        self.rate = rate
        self.sigma = sigma
        self.expiry = expiry
        self.option_type = option_type
        self.epsilon = epsilon
        self.mode = mode

        # initializing useful math quantities
        self.mu = (self.rate - 0.5 * self.sigma ** 2) * self.expiry + np.log(
            self.initial_asset_price
        )
        self.mean = np.exp(self.mu + 0.5 * self.expiry * self.sigma ** 2)  # -1
        self.variance = (np.exp(self.expiry * self.sigma ** 2) - 1) * np.exp(
            2 * self.mu + self.expiry * self.sigma ** 2
        )
        self.asset_distribution = np.linspace(
            max(self.mean - 2 * np.sqrt(self.variance), 0),
            self.mean + 2 * np.sqrt(self.variance),
            self.n_asset_prices,
        )

        # computing Black-Scholes model price
        self.bsm = self._bsm()

        # creating the quantum circuits
        self.probability_loader_circuit = self._generate_probability_loader_circuit()
        self.initial_circuit = self._generate_initial_circuit()
        self.iteration_circuit = self._generate_iteration_circuit()

    # Classical European Black-Scholes Model
    def _bsm(self):
        d1 = (
            np.log(self.initial_asset_price / self.strike)
            + (self.rate + self.sigma ** 2 / 2) * self.expiry
        ) / (self.sigma * np.sqrt(self.expiry))
        d2 = (
            np.log(self.initial_asset_price / self.strike)
            + (self.rate - self.sigma ** 2 / 2) * self.expiry
        ) / (self.sigma * np.sqrt(self.expiry))
        if self.option_type == "C":
            self.bsm = (self.initial_asset_price * norm.cdf(d1)) - (
                self.strike * np.exp(-self.rate * self.expiry) * norm.cdf(d2)
            )
        elif self.option_type == "P":
            self.bsm = -(self.initial_asset_price * norm.cdf(-d1)) + (
                self.strike * np.exp(-self.rate * self.expiry) * norm.cdf(-d2)
            )
        else:
            print("option_type must be " "C" " or " "P")
        return self.bsm

    # Function to generate a normalized Normal Law
    def _normalized_lognormal(self):

        ds = self.asset_distribution[1] - self.asset_distribution[0]
        normal_law = (
            1
            / (self.asset_distribution * self.sigma * np.sqrt(2 * np.pi * self.expiry))
            * np.exp(
                -np.power(np.log(self.asset_distribution) - self.mu, 2.0)
                / (2 * np.power(self.sigma, 2.0) * self.expiry)
            )
        )
        normalized_lognormal = np.sqrt(normal_law * ds / np.sum(normal_law * ds))

        return normalized_lognormal

    # Generate the circuit for probability distribution
    def _generate_probability_loader_circuit(self):

        # Generate probability Distributor
        normalized_lognormal = self._normalized_lognormal()

        # Load the probability Distribution in the quantum circuit via the data loader
        self.probability_loader_circuit = qio.loader(
            normalized_lognormal, mode="parallel", initial=True
        )
        # print('none',self.probability_loader_circuit)
        return self.probability_loader_circuit

    # Generate the initial circuit
    def _generate_initial_circuit(self):

        # Computation of angles needed for the payoff
        payoff_final = np.zeros(len(self.asset_distribution))
        payoff_angle = np.zeros(len(self.asset_distribution))
        count = 0
        for i in range(len(self.asset_distribution)):
            if self.asset_distribution[i] > self.strike and self.option_type == "C":
                payoff_angle[i] = np.arcsin(
                    np.sqrt(
                        (self.asset_distribution[i] - self.strike)
                        / (
                            self.asset_distribution[len(self.asset_distribution) - 1]
                            - self.strike
                        )
                    )
                )
                payoff_final[i] = self.asset_distribution[i] - self.strike
                count += 1
            if self.asset_distribution[i] < self.strike and self.option_type == "P":
                payoff_angle[i] = np.arcsin(
                    np.sqrt(
                        (self.strike - self.asset_distribution[i])
                        / (self.strike - self.asset_distribution[0])
                    )
                )
                payoff_final[i] = self.strike - self.asset_distribution[i]
                count += 1
        self._count = count

        if self.option_type == "C":
            self._asset_limit = self.asset_distribution[
                len(self.asset_distribution) - 1
            ]
        if self.option_type == "P":
            self._asset_limit = self.asset_distribution[0]

        circ = quasar.Circuit()
        circ = self.probability_loader_circuit.copy()
        # Generate the gate for the rotation needed for Payoff Computation
        for i in range(count):
            if self.option_type == "C" and self.mode == "sequential":
                circ.add_gate(
                    quasar.Gate.Ry(
                        theta=payoff_angle[self.n_asset_prices - count + i] / 2
                    ),
                    self.n_asset_prices,
                    time_placement="next",
                )
                circ.add_gate(
                    quasar.Gate.CX,
                    (self.n_asset_prices - count + i, self.n_asset_prices),
                    time_placement="next",
                )
                circ.add_gate(
                    quasar.Gate.Ry(
                        theta=-payoff_angle[self.n_asset_prices - count + i] / 2
                    ),
                    self.n_asset_prices,
                    time_placement="next",
                )
                circ.add_gate(
                    quasar.Gate.CX,
                    (self.n_asset_prices - count + i, self.n_asset_prices),
                    time_placement="next",
                )
            if self.option_type == "P" and self.mode == "sequential":
                circ.add_gate(
                    quasar.Gate.Ry(theta=payoff_angle[i] / 2),
                    self.n_asset_prices,
                    time_placement="next",
                )
                circ.add_gate(
                    quasar.Gate.CX, (i, self.n_asset_prices), time_placement="next"
                )
                circ.add_gate(
                    quasar.Gate.Ry(theta=-payoff_angle[i] / 2),
                    self.n_asset_prices,
                    time_placement="next",
                )
                circ.add_gate(
                    quasar.Gate.CX, (i, self.n_asset_prices), time_placement="next"
                )
            if self.option_type == "C" and self.mode == "parallel":
                circ.add_gate(
                    quasar.Gate.Ry(
                        theta=payoff_angle[self.n_asset_prices - count + i] / 2
                    ),
                    self.n_asset_prices + i,
                    time_placement="next",
                )
                circ.add_gate(
                    quasar.Gate.CX,
                    (self.n_asset_prices - count + i, self.n_asset_prices + i),
                    time_placement="next",
                )
                circ.add_gate(
                    quasar.Gate.Ry(
                        theta=-payoff_angle[self.n_asset_prices - count + i] / 2
                    ),
                    self.n_asset_prices + i,
                    time_placement="next",
                )
                circ.add_gate(
                    quasar.Gate.CX,
                    (self.n_asset_prices - count + i, self.n_asset_prices + i),
                    time_placement="next",
                )
            if self.option_type == "P" and self.mode == "parallel":
                circ.add_gate(
                    quasar.Gate.Ry(theta=payoff_angle[i] / 2),
                    self.n_asset_prices + i,
                    time_placement="next",
                )
                circ.add_gate(
                    quasar.Gate.CX, (i, self.n_asset_prices + i), time_placement="next"
                )
                circ.add_gate(
                    quasar.Gate.Ry(theta=-payoff_angle[i] / 2),
                    self.n_asset_prices + i,
                    time_placement="next",
                )
                circ.add_gate(
                    quasar.Gate.CX, (i, self.n_asset_prices + i), time_placement="next"
                )

        self.initial_circuit = circ.copy()
        return self.initial_circuit

    # Create the iteration circuit for the Amplitude Estimation
    def _generate_iteration_circuit(self):

        circ = quasar.Circuit()
        circ = self.initial_circuit.copy()

        # Define the Target Qubits
        if self.mode == "parallel":
            target_qubits = list(
                range(self.n_asset_prices, self.n_asset_prices + self._count)
            )
        elif self.mode == "sequential":
            target_qubits = [self.n_asset_prices]

        self.target_qubits = target_qubits

        # Define the target states
        target_states = [2 ** k for k in range(0, len(target_qubits))]
        self.target_states = target_states

        minusA = circ.slice(
            times=list(range(1, circ.ntime))
        )  # The minus phase doesn't matter, we put A itself, removing the X gate in the beginning

        S_chi = quasar.Circuit()
        for qubit in target_qubits:
            S_chi.Z(qubit)

        A_inverse = circ.adjoint().slice(
            times=list(range(0, circ.ntime - 1))
        )  # removing the X gate in the end

        S_0 = quasar.Circuit()
        for qubit in target_qubits:
            S_0.CZ(0, qubit)  # S_0 operator
        S_0.Z(0)

        self.iteration_circuit = quasar.Circuit.join_in_time(
            [S_chi, A_inverse, S_0, minusA]
        )  # the whole iteration

        return self.iteration_circuit

    def price(
        self,
        schedule=None,
        schedule_type="linear",
        max_depth=int(10),
        beta=0.3,
        n_shots=int(10),
    ):

        self.schedule = schedule
        self.schedule_type = schedule_type
        self.max_depth = max_depth
        self.beta = beta
        self.n_shots = n_shots

        # make schedule if it is not given
        if self.schedule is None:
            self.schedule = make_schedule(
                self.epsilon,
                self.schedule_type,
                self.max_depth,
                self.beta,
                self.n_shots,
            )

        # compute total number of oracle calls and shots
        if self.n_shots is not None:
            oracle_calls_counter = 0
            shots_counter = 0
            for [power, n_shots] in self.schedule:
                oracle_calls_counter += n_shots * (2 * power + 1)
                shots_counter += n_shots
            self.oracle_calls = oracle_calls_counter
            self.shots = shots_counter
        else:
            self.oracle_calls = None
            self.shots = None

        # get the quantum results
        self.results = run_schedule(
            self.initial_circuit,
            self.iteration_circuit,
            self.target_qubits,
            self.target_states,
            self.schedule,
        )

        # compute the parameter theta with MLE
        self.theta = compute_mle(self.results, self.epsilon)

        # compute the expetation
        self.expectation = np.sin(self.theta) ** 2

        # compute the quantum price
        price = 0
        if self.option_type == "C":
            price = self.expectation * (self._asset_limit - self.strike)
        if self.option_type == "P":
            price = self.expectation * (self.strike - self._asset_limit)
        self.quantum_price = price

        return self.quantum_price
