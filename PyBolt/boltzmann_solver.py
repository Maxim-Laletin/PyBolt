import numpy as np
import matplotlib.pyplot as plt
from . import constants as const
from .cosmology import H_t, H, s_ent, gtilda, Y_x_eq
from scipy.integrate import solve_ivp
import time
import logging
from typing import Callable


class Model:
    """
    A class that defines a DM model with all the neccessary parameters and processes that are relevant for the DM phase-space evolution,
    as well as a grid of x and q values that is used for the solution of the density and phase-space equations

    Parameters
    ----------
    m: float
        The mass of the heaviest particle in the reaction involving the DM particle (sets the characteristic scale and the relation for x = m/T)
        Note, that it is not necessarily the mass of the DM particle! [GeV]
    mDM: float
        The mass of DM particle [GeV]
    g_dof: float
        The number of dof of the DM particle [1]
    p_type: str
        The DM particle spin type (b = boson, f = fermion, m = Maxwell-Boltzmann particle)
    x: np.array
        The initial grid of x=m/T values [1]
    q: np.array
        The initial grid of DM momenta divided by T [1]

    The grid can be changed later.
    """

    def __init__(
        self,
        m: float,
        mDM: float,
        g_dof: float,
        p_type: str = "m",
        x: np.array = np.linspace(1.0, 10.0, 10),
        q: np.array = np.linspace(0.1, 20.0, 10),
    ):
        self._m = m
        self._mDM = mDM
        self._g = g_dof
        self._p_type = p_type

        if self._p_type not in {"f", "b", "m"}:
            print(
                "Incorrect particle type - should be either 'f' (fermion), 'b' (boson) or 'm' (classical). Setting the type to 'm' by default."
            )
            self._p_type = "m"

        self._x = x
        self._q = q

        self._f = np.zeros_like(
            [self._x, self._q], dtype=float
        )  # solution of the fBE - matrix of size Nx*Nq
        self._collision_terms = (
            []
        )  # list to store collision terms (as functions) provided by the user

    def getX(self) -> np.array:
        """ Returns the vector of x values in the grid """

        return self._x

    def getQ(self) -> np.array:
        """ Returns the vector of q values in the grid """

        return self._q

    def getSolution(self) -> np.array:
        """ Returns the solution of the fBE """

        return self._f

    def changeGrid(self, x_new: np.array, q_new: np.array) -> None:
        """ Changes the x and q grid for the model """

        self._x = x_new
        self._q = q_new
        self._f = np.zeros((np.size(x_new), np.size(q_new)), dtype=float)

    def equilibriumFunction(self, q: np.array) -> np.array:
        """ Computes the equilibrium function for the DM spin type and a given vector of q values """

        match self._p_type:
            case "b":
                return q**2 / (np.exp(q) - 1.0)
            case "f":
                return q**2 / (np.exp(q) + 1.0)
            case "m":
                return q**2 * np.exp(-q)

    def addCollisionTerm(self, CollTerm) -> None:
        """ Add a term to the collision integral """

        self._collision_terms.append(CollTerm)

    def _CI(self, x: float, q: float, f: float, feq: float) -> float:
        """ Full collision integral of the model (sum of all the collision terms provided by the user) """

        return sum(term(x, q, f, feq) for term in self._collision_terms)

    
    # ***************************************************************************************************
    def solve_fBE(self, f0: np.array, solver_options: dict) -> None:
        """
        The full Boltzmann equation solver function that takes an initial DM distribution (f0: np.array) and computes
        the distribution function for the x grid (x: np.array) and the collision term (a function of three variables: x,q and fq) in the given model

        Authors: Maxim Laletin, Michal Lukawski
        """

        x = self._x
        q = self._q

        # Sizes of x and q arrays
        sx = x.size
        sq = q.size

        # Step size in q
        dq = q[-1] - q[-2]

        # Equilibrium distribution
        f_eq = self.equilibriumFunction(q)

        def fBE_RHS(x: float, fq: np.array) -> np.array:
            """ The right-hand side of the full Boltzmann equation df/dx = RHS """

            eq = np.sqrt((self._mDM * x / self._m) ** 2 + q**2)

            dfdq = np.zeros_like(fq)
            # Four-point method numerical derivative
            dfdq[2:-2] = (-fq[4:] + 8 * fq[3:-1] - 8 * fq[1:-3] + fq[:-4]) / (12 * dq)
            """
            Below we assume that the distribution function at the edges of the q grid always behaves as an equilibrium one
            and use an analytical expression for the derivative (to avoid instabilities)
            """
            dfdq[:2] = (2 / q[:2] - q[:2] / eq[:2]) * fq[:2]
            dfdq[-2:] = (2 / q[-2:] - q[-2:] / eq[-2:]) * fq[-2:]

            # as defined by Eq. 9 in 2410.18186
            dfq_x = (
                gtilda(self._m / x) * (q * dfdq - 2 * fq)
                + q**2
                * (self._CI(x, q, fq, f_eq) / (2 * self._g * eq))
                / H_t(self._m / x)
            ) / x

            return dfq_x

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
        logging.info("Starting solve_fBE...")
        start_time = time.time()

        # Solving a system of ODEs for each momentum mode
        fBE_sol = solve_ivp(fBE_RHS, [x[0], x[-1]], f0, t_eval=x, **solver_options)

        if not fBE_sol.success:
            logging.error(f"Integration failed: {fBE_sol.message}")
        else:
            end_time = time.time()
            total_time = end_time - start_time
            logging.info(f"solve_fBE completed in {total_time:.2f} seconds")
            self._f = np.transpose(fBE_sol.y)

    # ***************************************************************************************************
    

    def solve_nBE(self, x: np.array, Rate: Callable[[float,float],float], Y0: float) -> np.array:
        """
        The comoving number density Boltzmann equation solver function that takes an array of x values, the Rate of the process under investigation and the initial comoving abundance Y0 to calculate an array of comoving densities Y for the corresponding values of x.

        Authors: Maxim Laletin, Adam Gomułka
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
        start_time = time.time()
        last_log_time = start_time
        log_interval = 1  # Log every 1 second
        x_start, x_end = x[0], x[-1]

        def YBE_RHS(x, Y):
            nonlocal last_log_time
            current_time = time.time()

            if current_time - last_log_time >= log_interval:
                elapsed_time = current_time - start_time
                progress = (x - x_start) / (x_end - x_start) * 100
                logging.info(
                    f"Progress: {x}/{x_end}, {progress:.2f}% - Elapsed time: {elapsed_time:.2f} seconds"
                )
                last_log_time = current_time

            return Rate(x, Y) * (H_t(self._m / x) * s_ent(self._m / x) * x) ** (-1)

        logging.info("Starting solve_nBE...")

        # Solve the differential equation
        sol = solve_ivp(
            YBE_RHS,
            [x_start, x_end],
            [Y0],
            t_eval=x,
            method="Radau",
            rtol=1e-6,
            atol=1e-8,
            vectorized=True,
        )

        if not sol.success:
            logging.error(f"Integration failed: {sol.message}")
        else:
            end_time = time.time()
            total_time = end_time - start_time
            logging.info(f"solve_nBE completed in {total_time:.2f} seconds")

        return sol.y[0] if sol.success else None


    # ***************************************************************************************************


    def getDensity(self) -> np.array:
        """
        Return the comoving density values from the solution of the fBE (by using a trapezoidal rule to integrate the distribution accordingly) [1]
        """

        Y_pde = (
            self._g
            * np.trapz(self._f, self._q, axis=1)
            * (self._m / self._x) ** 3
            / 2
            / np.pi**2
            / s_ent(self._m / self._x)
        )

        return Y_pde

    def plot2D(self):
        """ Create a 2D color plot to represent the solution of the fBE in x and q """

        mesh_q, mesh_x = np.meshgrid(self._q, self._x)

        plt.figure()
        plt.pcolormesh(mesh_q, mesh_x, self._f)
        plt.colorbar()
        plt.gca().set_title("fBE solution")
        plt.xlabel("q")
        plt.ylabel("x")
        plt.show()

    def plotFinalPDF(self):
        """ Plot the distribution function at the end of the evolution """

        Y_pde = (
            self._g
            * np.trapz(self._f[-1,:], self._q)
            * (self._m / self._x[-1]) ** 3
            / 2
            / np.pi**2
            / s_ent(self._m / self._x[-1])
        )

        plt.figure()
        plt.plot(self._q, self._f[-1, :])
        plt.plot(self._q, (Y_pde/Y_x_eq(self._m/self._x[-1]))*self.equilibriumFunction(self._q), "--")
        plt.xlabel(r"$q$")
        plt.ylabel(r"$q^2 f(q)$")
        plt.title("Final distribution function")
        plt.legend(["fBE", "Equilibrium (norm. to density)"], loc="upper right")
        plt.grid()
        plt.show()
