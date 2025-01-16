import numpy
from .cosmology import Y_x_eq
from scipy.special import kn  # Bessel function


class DecayToX:  # 1->2 decay where X is massless
    """
    A class for the 2-body decay processes of a fermion particle into another massive fermion and a massless scalar DM
    collisionTerm is used for the fBE solution, while the rate is used for the nBE solution.
    """

    def __init__(self, m1: float, m2: float, g_1: float, Msquared: float, Gamma: float):
        """
        Parameters
        ----------
        m1: float
            The mass of the decaying particle [GeV]
        m2: float
            The mass of the daughter particle (not DM) [GeV]
        g_1: float
            The number of dof of the DM particle [1]
        Msquared: float
            The value of the amplitude squared SUMMED over all final states (constant in this case) [1]
        Gamma: float
            The value of the decay width (can be actually computed from Msquared) [GeV^4]
        """
        self._m1 = m1
        self._m2 = m2
        self._mu = m2 / m1  # ratio of masses [1]
        self._g_1 = g_1
        self._Msquared = Msquared
        self._Gamma = Gamma

    def collisionTerm(
        self, x: float, q: np.array, f: np.array, feq: np.array
    ) -> np.array:
        """
        The collision term below is for the production of a (scalar) massless particle in the decay with the mother particle and the other daugther particle being fermions

        Parameters
        ----------
        x: float
            The inverse unit of temperature x = m/T [1]
        q: np.array
            The vector of momenta of particle X divided by T [1]
        f: np.array
            The distribution function of X particle (vector should be the same size as q) [1]
        feq: np.array
            The corresponding equilibrium distribution of X particle [1]
        """

        # The following structure comes from the integration limits
        Elim1 = np.max(
            [
                x * np.ones(q.shape),
                x * self._mu + q,
                x**2
                * (1 + 4 * q**2 / (x**2 * (1 - self._mu**2)) - self._mu**2)
                / 4
                / q,
            ],
            axis=0,
        )  # [1]
        Elim2 = np.max(
            [x - q, x * self._mu * np.ones(q.shape), x**2 / 4 / q], axis=0
        )  # [1]

        return (
            (2 * self._g_1 * self._Msquared * x / self._m1 / 8 / np.pi / q**3)
            * (np.log((1 + np.exp(-Elim1)) / (1 + np.exp(-Elim2))))
            * (f - feq)
        )

    def rate(self, x: float, Y: float) -> float:
        """
        The rate of decay that enters the number density Boltzmann equation (with the MB distribution for the decaying particle)

        Parameters
        ----------
        x: float
            The inverse unit of temperature x = m/T
        Y: float
            The comoving abundance of particle X at the given value of x
        """
        return (
            8
            * self._Gamma
            * self._m1**3
            * (kn(1, x) / x)
            * (1 - Y / Y_x_eq(self._m1 / x))
            / (2 * np.pi) ** 2
        )
