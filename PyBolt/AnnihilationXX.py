import numpy as np
from .cosmology import s_ent, gtilda, Y_x_eq_massive
from scipy.integrate import quad, solve_ivp, fixed_quad
from scipy.interpolate import interp1d
import time
from tqdm import tqdm
import logging
from scipy.special import kn 


class AnnihilationXX:  # 2->2X annihilation
    """
    A class for computing Scalar Singlet Dark Matter (SSDM) annihilation.

    Author: Adam Gomułka
    """

    def __init__(
        self,
        m1: float,
        g_1: float,
        lambda_S: float,
        x_min: float = 10,
        x_max: float = 200,
        num_points: int = 100,
    ):
        """
        Parameters
        ----------
        m1 : float
            The mass of the annihilating particles.
        g_1 : float
            The number of degrees of freedom of the annihilated particles.
        lambda_S : float
            The coupling constant.
        x_min : float
            The minimum value of x = m/T.
        x_max : float
            The maximum value of x = m/T.
        num_points : int
            The number of points in the x grid.
        """
        self._m1 = m1
        self._g_1 = g_1
        self._lambda_S = lambda_S
        self._x_table = np.logspace(np.log10(x_min), np.log10(x_max), num_points)
        self._sigma_v_table = self._tabulate_sigma_v()
        self._sigma_v_interp = interp1d(
            self._x_table, self._sigma_v_table, kind="cubic", fill_value="extrapolate"
        )

    def D_h_squared(self, s: float) -> float:
        """Helper function to compute the cross sections."""
        # Gamma_inv = self._lambda_S**2*pp.v_0**2 / (32 * np.pi * pp.higgs_mass**2)*np.sqrt(1-4*self._m1**2/pp.higgs_mass**2)
        return 1 / (
            (s - pp.higgs_mass**2) ** 2 + pp.higgs_mass**2 * (pp.Gamma_h_tot) ** 2
        )

    def sigma_v_cms(self, s: float) -> float:
        """Compute the cross-section times velocity in the center of mass frame."""
        return (
            2
            * self._lambda_S**2
            * pp.v_0**2
            / np.sqrt(s)
            * self.D_h_squared(s)
            * pp.Gamma_h(np.sqrt(s))
        )

    def _tabulate_sigma_v(self) -> np.ndarray:
        """Tabulate the cross-section times velocity."""
        sigma_v_values = []
        for x in tqdm(self._x_table):
            prefactor = x / (8 * self._m1**5 * kn(2, x) ** 2)
            s_min = (2 * self._m1) ** 2
            s_max = 1.215 * s_min  # should be adjusted

            def integrand(s: float) -> float:
                # return s * np.sqrt(s - 4 * self._m1**2) * kn(1, x * np.sqrt(s) / self._m1) * self.sigma_v_cms(s)*s/(2*s-4*self._m1**2)
                return (
                    np.sqrt(s)
                    * (s - 4 * self._m1**2)
                    * kn(1, x * np.sqrt(s) / self._m1)
                    * self.sigma_v_cms(s)
                    * s
                    / (2 * s - 4 * self._m1**2)
                )

            def transformed_integrand(log_s: float) -> float:
                s = np.exp(log_s)
                return (
                    integrand(s) * s
                )  # Jacobian of the transformation ds = s d(log_s)

            # Integration limits in the transformed variable
            log_s_min = np.log(s_min)
            log_s_max = np.log(s_max)

            # Perform the integration in the transformed variable
            integral, _ = quad(transformed_integrand, log_s_min, log_s_max)

            sigma_v_values.append(prefactor * integral)

            # integral, _ = quad(integrand, s_min, np.inf)
            # sigma_v_values.append(prefactor * integral)

            # # Plot the transformed integrand for debugging
            # log_s_values = np.linspace(log_s_min2, log_s_max, 100)
            # s_values = np.exp(log_s_values)
            # plt.plot(s_values, [integrand(s) for s in s_values])
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.xlabel('s')
            # plt.ylabel('Integrand')
            # plt.title(f'Transformed Integrand for x={x}')
            # plt.show()

        return np.array(sigma_v_values)

    def sigma_v(self, x: float) -> float:
        """Compute the thermal averaged cross-section times velocity via cubic interpolation from _sigma_v_table."""
        return self._sigma_v_interp(x)

    def rate(self, x: float, Y: float) -> float:
        """
        Compute the rate of the annihilation process -- the R.H.S. of the Boltzmann equation.

        Parameters
        ----------
        x : float
            The mass of scalar particle to the temperature ratio: m/T.
        Y : float
            The dark matter abundance.
        """

        rate = (
            self.sigma_v(x)
            * (Y_x_eq_massive(self._m1 / x, self._m1) ** 2 - Y**2)
            * s_ent(self._m1 / x) ** 2
        )

        return rate

