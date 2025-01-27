import numpy
from typing import Callable
from .cosmology import Y_x_eq
from scipy.integrate import quad, dblquad
from scipy.special import kn  # Bessel function


class Inelastic:  # 1->2 decay where X is massless
    """
    A class for the 2-body decay processes of a fermion particle into another massive fermion and a massless scalar DM
    collisionTerm is used for the fBE solution, while the rate is used for the nBE solution.
    """

    def __init__(self, m_X: float, m_k: float, m_i: float, m_j: float, g_X: float, sigma: Callable[[float],float]):
        """
        Parameters
        ----------
        m_X: float
            The mass of the X particle (unknown distribution)
        m_k: float
            The mass of the k particle
        ... 
        
        g_X: float
            The number of dof of the X particle [1]
        sigma: Callable[[float],float]
            The cross section of the reactions as a function of s 
        """
        self._mX = m_X
        self._mk = m_k
        self._mi = m_i
        self._mj = m_j
        self._gX = g_X
        self._sigma = sigma

        self._mmax = max(m_X, m_k, m_i, m_j) 

    #def integral_s(self, s_min: float,s_max: float) -> float:
  
      
    # Simplified collision term
    def collisionTerm(
        self, x: float, q: np.array, f: np.array, feq: np.array
    ) -> np.array:
        """
        The collision term below is the simplified version for the inelastic scattering X + SM -> SM + SM 

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

        def gamma(self, x: float,q: np.array) -> np.array:

          ex = np.sqrt(x**2 + q**2)

          ek_min =  0.5*( ex*( (self._mi+self._mj)**2 - self._mk**2 - self._mx**2) - q*np.sqrt(( (self._mx + self._mi + self._mj)**2 - self._mk**2)*( (self._mx - self._mi - self._mj)**2 - self._mk**2)) )/self._mx**2; # [1]
          if ek_min > 0.5*( (self._mi + self._mj)**2 - self._mx**2 - self._mk**2)/ex):
            ek_min = self.mk
          ek_max =  20*ek_min # [1]
          # ADD A BETTER CONDITION FOR THE UPPER LIMIT ek
          s_min = lambda ek: max( self._mx**2 + self._mk**2 + 2*ex*ek - 2*np.sqrt(ek**2 - self._mk**2)*q , (self._mi + self._mj)**2 )*(self._mmax/x)**2 # [GeV^2]
          s_max = lambda ek: ( self._mx**2 + self._mk**2 + 2*ex*ek + 2*np.sqrt(ek**2 - self._mk**2)*q )*(self._mmax/x)**2 # [GeV^2]
          
          #integral_ek, _ = quad(lambda ek: ek*np.exp(-ek*x/self._mX)*integral_s(s_min,s_max), ek_min, ek_max)
          integral_ek, _ = dblquad(lambda ek, s: 0.5*np.exp(-ek)*self._sigma(s)*np.sqrt( (s - (self._mx + self._mk)**2)*(s - (self._mx - self._mk)**2) ), ek_min, ek_max, s_min, s_max) # [GeV^2]

          return np.exp(-np.sqrt(x**2 + q**2))*integral_ek/(2*np.pi)**2 # [GeV^2]

        # CHECK THE DIMENSIONS AND THE CORRECTNESS
        return (1.0 - f/feq)*gamma(x,q)/2/self._gX/np.sqrt(self._m_X**2 + q**2) # [GeV]
    

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
