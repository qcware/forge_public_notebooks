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
        nAssetPrices : Number of different Asset Prices generated
        S0 : Initial Asset Price
        K : Strike
        r : interest rate
        sigma : Volatility
        T : Time to maturity
        optionType : 'C' = European Call Option, 'P' = European Put Option
        epsilon : precision parameter for the estimate
        mode : 'parallel' = optimized depth, 'sequential' = optimized number of qubits

    Outputs:
        Circuits used for pricing
            self.probability_loader_circuit :
            self.initial_circuit : 'oracle' circuit A
            self.iteration_circuit : iteration circuit (S_chi A^{inv} S_0 A) used for AE

        self.samples : number of samples
        self.theta : estimate of angle theta
        self.expectation : estimate of expectation of sin(theta)**2
        self.price : final estimated price

        self.schedule : schedule used for AE
        self.scheduleType : scheduleType
        self.maxDepth : maxDepth of circuit
        self.beta : beta parameter for power-law schedule
        self.nShots : nShots for each circuit


    """

    def __init__(
        self,
        nAssetPrices,
        S0,
        K,
        r,
        sigma,
        T,
        optionType,
        epsilon=0.005,
        mode="sequential",
    ):

        # initializing classical parameters
        self.nAssetPrices = nAssetPrices
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.optionType = optionType
        self.epsilon = epsilon
        self.mode = mode

        # initializing useful math quantities
        self.mu = (self.r - 0.5 * self.sigma ** 2) * self.T + np.log(self.S0)
        self.mean = np.exp(self.mu + 0.5 * self.T * self.sigma ** 2)  # -1
        self.variance = (np.exp(self.T * self.sigma ** 2) - 1) * np.exp(
            2 * self.mu + self.T * self.sigma ** 2
        )
        self.S = np.linspace(
            max(self.mean - 2 * np.sqrt(self.variance), 0),
            self.mean + 2 * np.sqrt(self.variance),
            self.nAssetPrices,
        )

        # computing Black-Scholes model price
        self.BSM = self._BSM()

        # creating the quantum circuits
        self.probability_loader_circuit = self._generate_probability_loader_circuit()
        self.initial_circuit = self._generate_initial_circuit()
        self.iteration_circuit = self._generate_iteration_circuit()

    # Classical European Black-Scholes Model
    def _BSM(self):
        d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        d2 = (np.log(self.S0 / self.K) + (self.r - self.sigma ** 2 / 2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        if self.optionType == "C":
            self.BSM = (self.S0 * norm.cdf(d1)) - (
                self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            )
        elif self.optionType == "P":
            self.BSM = -(self.S0 * norm.cdf(-d1)) + (
                self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
            )
        else:
            print("optionType must be " "C" " or " "P")
        return self.BSM

    # Function to generate a normalized Normal Law
    def _normalizedLogNormal(self):

        dS = self.S[1] - self.S[0]
        normalLaw = (
            1
            / (self.S * self.sigma * np.sqrt(2 * np.pi * self.T))
            * np.exp(
                -np.power(np.log(self.S) - self.mu, 2.0)
                / (2 * np.power(self.sigma, 2.0) * self.T)
            )
        )
        normalizedLogNormal = np.sqrt(normalLaw * dS / np.sum(normalLaw * dS))

        return normalizedLogNormal

    # Generate the circuit for probability distribution
    def _generate_probability_loader_circuit(self):

        # Generate probability Distributor
        normalizedLogNormal = self._normalizedLogNormal()

        # Load the probability Distribution in the quantum circuit via the data loader
        self.probability_loader_circuit = qio.loader(
            normalizedLogNormal, mode="parallel", initial=True
        )
        # print('none',self.probability_loader_circuit)
        return self.probability_loader_circuit

    # Generate the initial circuit
    def _generate_initial_circuit(self):

        # Computation of angles needed for the payoff
        payoffFinal = np.zeros(len(self.S))
        payoffAngle = np.zeros(len(self.S))
        count = 0
        for i in range(len(self.S)):
            if self.S[i] > self.K and self.optionType == "C":
                payoffAngle[i] = np.arcsin(
                    np.sqrt((self.S[i] - self.K) / (self.S[len(self.S) - 1] - self.K))
                )
                payoffFinal[i] = self.S[i] - self.K
                count += 1
            if self.S[i] < self.K and self.optionType == "P":
                payoffAngle[i] = np.arcsin(
                    np.sqrt((self.K - self.S[i]) / (self.K - self.S[0]))
                )
                payoffFinal[i] = self.K - self.S[i]
                count += 1
        self._count = count

        if self.optionType == "C":
            self._assetLimit = self.S[len(self.S) - 1]
        if self.optionType == "P":
            self._assetLimit = self.S[0]

        circ = quasar.Circuit()
        circ = self.probability_loader_circuit.copy()
        # Generate the gate for the rotation needed for Payoff Computation
        for i in range(count):
            if self.optionType == "C" and self.mode == "sequential":
                circ.add_gate(
                    quasar.Gate.Ry(
                        theta=payoffAngle[self.nAssetPrices - count + i] / 2
                    ),
                    self.nAssetPrices,
                    time_placement="next",
                )
                circ.add_gate(
                    quasar.Gate.CX,
                    (self.nAssetPrices - count + i, self.nAssetPrices),
                    time_placement="next",
                )
                circ.add_gate(
                    quasar.Gate.Ry(
                        theta=-payoffAngle[self.nAssetPrices - count + i] / 2
                    ),
                    self.nAssetPrices,
                    time_placement="next",
                )
                circ.add_gate(
                    quasar.Gate.CX,
                    (self.nAssetPrices - count + i, self.nAssetPrices),
                    time_placement="next",
                )
            if self.optionType == "P" and self.mode == "sequential":
                circ.add_gate(
                    quasar.Gate.Ry(theta=payoffAngle[i] / 2),
                    self.nAssetPrices,
                    time_placement="next",
                )
                circ.add_gate(
                    quasar.Gate.CX, (i, self.nAssetPrices), time_placement="next"
                )
                circ.add_gate(
                    quasar.Gate.Ry(theta=-payoffAngle[i] / 2),
                    self.nAssetPrices,
                    time_placement="next",
                )
                circ.add_gate(
                    quasar.Gate.CX, (i, self.nAssetPrices), time_placement="next"
                )
            if self.optionType == "C" and self.mode == "parallel":
                circ.add_gate(
                    quasar.Gate.Ry(
                        theta=payoffAngle[self.nAssetPrices - count + i] / 2
                    ),
                    self.nAssetPrices + i,
                    time_placement="next",
                )
                circ.add_gate(
                    quasar.Gate.CX,
                    (self.nAssetPrices - count + i, self.nAssetPrices + i),
                    time_placement="next",
                )
                circ.add_gate(
                    quasar.Gate.Ry(
                        theta=-payoffAngle[self.nAssetPrices - count + i] / 2
                    ),
                    self.nAssetPrices + i,
                    time_placement="next",
                )
                circ.add_gate(
                    quasar.Gate.CX,
                    (self.nAssetPrices - count + i, self.nAssetPrices + i),
                    time_placement="next",
                )
            if self.optionType == "P" and self.mode == "parallel":
                circ.add_gate(
                    quasar.Gate.Ry(theta=payoffAngle[i] / 2),
                    self.nAssetPrices + i,
                    time_placement="next",
                )
                circ.add_gate(
                    quasar.Gate.CX, (i, self.nAssetPrices + i), time_placement="next"
                )
                circ.add_gate(
                    quasar.Gate.Ry(theta=-payoffAngle[i] / 2),
                    self.nAssetPrices + i,
                    time_placement="next",
                )
                circ.add_gate(
                    quasar.Gate.CX, (i, self.nAssetPrices + i), time_placement="next"
                )

        self.initial_circuit = circ.copy()
        return self.initial_circuit

    # Create the iteration circuit for the Amplitude Estimation
    def _generate_iteration_circuit(self):

        circ = quasar.Circuit()
        circ = self.initial_circuit.copy()

        # Define the Target Qubits
        if self.mode == "parallel":
            targetQubits = list(
                range(self.nAssetPrices, self.nAssetPrices + self._count)
            )
        elif self.mode == "sequential":
            targetQubits = [self.nAssetPrices]

        self.targetQubits = targetQubits

        # Define the target states
        targetStates = [2 ** k for k in range(0, len(targetQubits))]
        self.targetStates = targetStates

        minusA = circ.slice(
            times=list(range(1, circ.ntime))
        )  # The minus phase doesn't matter, we put A itself, removing the X gate in the beginning

        S_chi = quasar.Circuit()
        for qubit in targetQubits:
            S_chi.Z(qubit)

        A_inverse = circ.adjoint().slice(
            times=list(range(0, circ.ntime - 1))
        )  # removing the X gate in the end

        S_0 = quasar.Circuit()
        for qubit in targetQubits:
            S_0.CZ(0, qubit)  # S_0 operator
        S_0.Z(0)

        self.iteration_circuit = quasar.Circuit.join_in_time(
            [S_chi, A_inverse, S_0, minusA]
        )  # the whole iteration

        return self.iteration_circuit

    def price(
        self,
        schedule=None,
        scheduleType="linear",
        maxDepth=int(10),
        beta=0.3,
        nShots=int(10),
    ):

        self.schedule = schedule
        self.scheduleType = scheduleType
        self.maxDepth = maxDepth
        self.beta = beta
        self.nShots = nShots

        # make schedule if it is not given
        if self.schedule is None:
            self.schedule = make_schedule(
                self.epsilon, self.scheduleType, self.maxDepth, self.beta, self.nShots
            )

        # compute total number of samples
        if self.nShots is not None:
            samples = 0
            for [power, nShots] in self.schedule:
                samples += nShots * (2 * power + 1)
            self.samples = samples
        else:
            self.samples = None

        # get the quantum results
        self.results = run_schedule(
            self.initial_circuit,
            self.iteration_circuit,
            self.targetQubits,
            self.targetStates,
            self.schedule,
        )

        # compute the parameter theta with MLE
        self.theta = compute_mle(self.results, self.epsilon)

        # compute the expetation
        self.expectation = np.sin(self.theta) ** 2

        # compute the quantum price
        price = 0
        if self.optionType == "C":
            price = self.expectation * (self._assetLimit - self.K)
        if self.optionType == "P":
            price = self.expectation * (self.K - self._assetLimit)
        self.quantumprice = price

        return self.quantumprice
