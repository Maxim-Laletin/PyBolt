import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d
from scipy.special import kn  # Bessel function
from scipy.integrate import quad, fixed_quad
from .cosmology import Y_x_eq, Y_x_eq_massive, h_s, nmeq, npheq
from .constants import e_g
from tqdm import tqdm

# Classes of processes
class Process(ABC):
    """
    Base class for all the processes.
    """
    
    def __init__(self, m1: float, g_1: float, coupling: float, **kwargs):
        self._m1 = m1
        self._g_1 = g_1
        self._coupling = coupling

    @abstractmethod
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
        
        pass
    
    @abstractmethod
    def collisionTerm(self, x: float, q: np.array, f: np.array, feq: np.array) -> np.array:
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
        
        pass


class DecayToX(Process): # 1 -> 2 decay where X is massless
    """
    A class for the 2-body decay processes of a fermion particle into another massive fermion and a massless scalar DM
    collisionTerm is used for the fBE solution, while the rate is used for the nBE solution.
    """
    def __init__(self, m1, g_1, coupling, 
                 m2: float,
                 Msquared_stripped: float
                ):
        super().__init__(m1 = m1, g_1 = g_1, coupling = coupling)
        self._Msquared = coupling**2*Msquared_stripped
        self._mu = m2 / m1  # ratio of masses [1]
        self._Gamma = self._Msquared*(1 - self._mu**2)/self._m1/16/np.pi # GeV
        

    def rate(self, x, Y):
        
        return (
            8
            * self._Gamma
            * self._m1**3
            * (kn(1, x) / x)
            * (1 - Y / Y_x_eq(self._m1 / x))
            / (2 * np.pi) ** 2
        )

    def collisionTerm(self, x, q, f, feq):

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


def lam_f(x: float, y: float, z: float) -> float:
        """ Auxiliary function for the calculation of the velocity-averaged cross section"""

        return (x - (y+z)**2)*(x - (y-z)**2) # [GeV**4]


class LeptonAnnihilationToAxionMB(Process): # l_i + l_j -> X + gamma_k 
    """
    A class for the simplified version of the process of annihilation of two leptons into an axion and photon. Axion (massless) is the particle of interest in this reaction. It is assumed that the leptons are described by a Maxwell-Boltzmann distribution. We also assume that the axion number of dof is 1. 
    """

    def __init__(self, m1, g_1, coupling):
        super().__init__(m1 = m1, g_1 = g_1, coupling = coupling)
        
        """
        Parameters
        ----------
        m1 : float
            The mass of the lepton
        g_1 : float
            The number of degrees of freedom of the annihilated particles.
        coupling : float
            The coupling constant for this process (C_l/f_a)
        """

    def sigma_ann(self, s: float) -> float:
        """ Annihilation cross section """

        return (self._coupling*e_g*self._m1)**2*np.atanh(np.sqrt(1-4*self._m1**2/s))/(s-4*self._m1**2)/4/np.pi # [GeV**(-2)]

    def sigmaV_ann(self, x: float) -> float:
        """ Velocity-averaged annihilation cross section (for MB distributions)"""

        integral, _ = quad(lambda s: lam_f(s,self._m1,self._m1)*self.sigma_ann(s)*kn(1,np.sqrt(s)*x/self._m1)/np.sqrt(s),4*self._m1**2,np.inf)

        return x*integral/kn(2,x)**2/8/self._m1**5 # [GeV**-2]
        

    def rate(self, x, Y):

        return nmeq(x, self._g_1, self._m1)**2*self.sigmaV_ann(x)*(1 - Y/Y_x_eq(self._m1 / x)) # [GeV**4]
        

    def collisionTerm(self, x, q, f, feq):

        # if x < 0.001:
        #     C_func = (self._g_1**2*(e_g*self._coupling)**2*self._m1**3)*(np.exp(-q)/2/(2*np.pi)**3)*(np.pi**2/6)/x   
        # else:
        integral = np.array([quad(lambda ek: ((2*ek*q_i - x**2)*np.atanh(np.sqrt(1 - x**2/ek/q_i)) - ek*q_i*np.sqrt(1 - x**2/ek/q_i))/(np.exp(ek)-1.0),x**2/q_i,np.inf)[0] for q_i in q])
            
        C_func = (self._g_1**2*(e_g*self._coupling)**2*self._m1**3)*(np.exp(-q)/q/x/2/(2*np.pi)**3)*integral

        return (1 - f/feq)*C_func


class PrimakoffScatteringMB(Process): # l_i + X -> l_j + gamma_k 
    """
    A class for the simplified version of the Primakoff scattering of axion on a lepton. Axion (massless) is the particle of interest in this reaction. It is assumed that the leptons are described by a Maxwell-Boltzmann distribution. We also assume that the axion number of dof is 1. 
    """

    def __init__(self, m1, g_1, coupling):
        super().__init__(m1 = m1, g_1 = g_1, coupling = coupling)
        
        """
        Parameters
        ----------
        m1 : float
            The mass of the lepton
        g_1 : float
            The number of degrees of freedom of the lepton.
        coupling : float
            The coupling constant for this process (C_l/f_a)
        """
        
    def sigma_prim(self, s: float) -> float:
        """ Primakoff scattering cross section """
        
        return (
            (e_g*self._coupling*self._m1)**2
            * (2*s**2*np.log(s/self._m1**2) - 3*s**2 + 4*self._m1**2*s - self._m1**4)
            / s**2/(s-self._m1**2)/32/np.pi
        )# [GeV**-2]

    def sigmaV_prim(self, x: float) -> float:
        """ Velocity-averaged cross section of the Primakoff scattering (for MB distributions)"""

        integral, _ = quad(lambda s: lam_f(s,self._m1,0.0)*self.sigma_prim(s)*kn(1,np.sqrt(s)*x/self._m1)/np.sqrt(s),self._m1**2,np.inf)
        
        return x**3*integral/kn(2,x)/16/self._m1**5 # [GeV**-2]


    def rate(self, x, Y):

        return 2*nmeq(x, self._g_1, self._m1)*npheq(self._m1/x)*self.sigmaV_prim(x)*(1 - Y/Y_x_eq(self._m1 / x)) # [GeV**4]
        

    def collisionTerm(self, x, q, f, feq):

        # norm = (0.5./g_ax).*(2*par(1)*par(2)^2)*mmu
        # Integral_s = @(s,x) 2*s.*log(s./x.^2) + 4*x.^2.*log(s) - 5*s + x.^4./s; 
        # y = norm*(q.*x)^(-1).*(exp(-q)./q/32/(2*pi)^3 ).*integral(@(ek) (Integral_s(x.^2 + 2*ek.*q + 2*sqrt(ek.^2 - x.^2).*q,x) - Integral_s(x.^2 + 2*ek.*q - 2*sqrt(ek.^2 - x.^2).*q,x))./(exp(ek)+1.0),x,Inf);

        integral_s = lambda s,x: 2*s*np.log(s/x**2) + 4*x**2*np.log(s) - 5*s + x**4/s
        
        integral = np.array([quad(lambda ek: (integral_s(x**2 + 2*ek*q_i + 2*np.sqrt(ek**2 - x**2)*q_i,x) - integral_s(x**2 + 2*ek*q_i - 2*np.sqrt(ek**2 - x**2)*q_i,x))/(np.exp(ek)+1.0),x,np.inf)[0] for q_i in q])
            
        C_func = (2*self._g_1**2*(e_g*self._coupling)**2*self._m1**3)*(np.exp(-q)/q/x/32/(2*np.pi)**3)*integral

        return (1 - f/feq)*C_func



class Annihilation:  # 2 -> 2 annihilation
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

    def collisionTerm(self, x, q, f, feq):
        # Not implemented yet

        return zeros(q.shape)
    