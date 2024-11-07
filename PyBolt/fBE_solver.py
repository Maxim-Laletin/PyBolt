import numpy as np
import matplotlib.pyplot as plt
import constants as const
from cosmology import H_t, H, s_ent, gtilda, Y_x_eq, Y_x_eq_massive, h_s
from scipy.integrate import quad, solve_ivp, fixed_quad
from scipy.interpolate import interp1d
import time
from tqdm import tqdm
import logging

# =================================================================================================================================

# Classes of processes
from scipy.special import kn # Bessel function

class DecayToX: # 1->2 decay where X is massless
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
        self._mu = m2/m1 # ratio of masses [1]
        self._g_1 = g_1
        self._Msquared = Msquared 
        self._Gamma = Gamma 

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

        # The following structure comes from the integration limits
        Elim1 = np.max([x*np.ones(q.shape), x*self._mu+q, x**2*(1 + 4*q**2/(x**2*(1-self._mu**2)) - self._mu**2 )/4/q],axis=0) # [1]
        Elim2 = np.max([x-q, x*self._mu*np.ones(q.shape), x**2/4/q],axis=0) # [1]

        return (2*self._g_1*self._Msquared*x/self._m1/8/np.pi/q**3)*(np.log((1+np.exp(-Elim1))/(1+np.exp(-Elim2))))*(f - feq)

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
        return 8*self._Gamma*self._m1**3*(kn(1,x)/x)*(1 - Y/Y_x_eq(self._m1/x))/(2*np.pi)**2

class Annihilation: # 2->2 annihilation
        """
        A class for computing Scalar Singlet Dark Matter (SSDM) annihilation.

        Author: Adam Gomułka
        """
        def __init__(self, m1: float, g_1: float, lambda_S: float, 
                     x_min: float = 10, x_max: float = 200, num_points: int = 100):
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
            self._sigma_v_interp = interp1d(self._x_table, self._sigma_v_table, kind='cubic', fill_value='extrapolate')

        def D_h_squared(self, s: float) -> float:
            """Helper function to compute the cross sections."""
            #Gamma_inv = self._lambda_S**2*pp.v_0**2 / (32 * np.pi * pp.higgs_mass**2)*np.sqrt(1-4*self._m1**2/pp.higgs_mass**2)
            return 1 / ((s - pp.higgs_mass**2)**2 + pp.higgs_mass**2 * (pp.Gamma_h_tot)**2)

        def sigma_v_cms(self, s: float) -> float:
            """Compute the cross-section times velocity in the center of mass frame."""
            return 2 * self._lambda_S**2 * pp.v_0**2 / np.sqrt(s) * self.D_h_squared(s) * pp.Gamma_h(np.sqrt(s))

        def _tabulate_sigma_v(self) -> np.ndarray:
            """Tabulate the cross-section times velocity."""
            sigma_v_values = []
            for x in tqdm(self._x_table):
                prefactor = x / (8 * self._m1**5 * kn(2, x)**2)
                s_min = (2 * self._m1)**2
                s_max = 1.215 * s_min # should be adjusted 
                def integrand(s: float) -> float:
                    #return s * np.sqrt(s - 4 * self._m1**2) * kn(1, x * np.sqrt(s) / self._m1) * self.sigma_v_cms(s)*s/(2*s-4*self._m1**2)
                    return np.sqrt(s) * (s - 4 * self._m1**2) * kn(1, x * np.sqrt(s) / self._m1) * self.sigma_v_cms(s)*s/(2*s-4*self._m1**2)

                def transformed_integrand(log_s: float) -> float:
                    s = np.exp(log_s)
                    return integrand(s) * s  # Jacobian of the transformation ds = s d(log_s)
                
                # Integration limits in the transformed variable
                log_s_min = np.log(s_min)
                log_s_max = np.log(s_max)

                # Perform the integration in the transformed variable
                integral, _ = quad(transformed_integrand, log_s_min, log_s_max)

                sigma_v_values.append(prefactor * integral)

                #integral, _ = quad(integrand, s_min, np.inf)
                #sigma_v_values.append(prefactor * integral)

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

            rate = self.sigma_v(x)*(Y_x_eq_massive(self._m1/x, self._m1)**2-Y**2)*s_ent(self._m1/x)**2

            return rate
# =================================================================================================================================


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
    def __init__(self, m: float, mDM: float, g_dof: float, p_type: str = 'm', x: np.array = np.linspace(1.0,10.0,10), q: np.array = np.linspace(0.1,20.,10)):
        self._m = m 
        self._mDM = mDM
        self._g = g_dof 
        self._p_type = p_type

        if self._p_type not in {'f','b','m'}:
            print("Incorrect particle type - should be either 'f' (fermion), 'b' (boson) or 'm' (classical). Setting the type to 'm' by default.")
            self._p_type = 'm'

        self._x = x 
        self._q = q 
         
        self._f = np.zeros_like([self._x, self._q],dtype=float) # solution of the fBE - matrix of size Nx*Nq
        self._collision_terms = [] # list to store collision terms (as functions) provided by the user

    def getX(self) -> np.array:
        #Returns the vector of x values in the grid
        
        return self._x

    def getQ(self) -> np.array:
        # Returns the vector of q values in the grid

        return self._q

    def getSolution(self) -> np.array:
        # Returns the solution of the fBE

        return self._f

    def changeGrid(self, x_new: np.array, q_new: np.array):
        # Changes the x and q grid for the model

        self._x = x_new
        self._q = q_new
        self._f = np.zeros((np.size(x_new),np.size(q_new)),dtype=float)

    def equilibriumFunction(self, q: np.array) -> np.array:
        # Computes the equilibrium function for the DM spin type and a given vector of q values

        match self._p_type:
            case 'b':
                return q**2/(np.exp(q)-1.0)
            case 'f':
                return q**2/(np.exp(q)+1.0)
            case 'm':
                return q**2*np.exp(-q)

    def addCollisionTerm(self, CollTerm):
        # Add a term to the collision integral

        self._collision_terms.append(CollTerm)

    def _CI(self, x: float, q: float, f: float, feq: float) -> float:
        # Full collision integral of the model (sum of all the collision terms provided by the user)

        return sum(term(x,q,f,feq) for term in self._collision_terms)


    # ***************************************************************************************************
    def solve_fBE(self, f0: np.array, solver_options: dict) -> np.array:
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
        dq = q[-1]-q[-2]
        
        # Equilibrium distribution
        f_eq = self.equilibriumFunction(q)
    
    
        def fBE_RHS(x: float, fq: np.array) -> np.array:
            # The right-hand side of the full Boltzmann equation df/dx = RHS

            eq = np.sqrt( (self._mDM*x/self._m)**2 + q**2)
            
            dfdq = np.zeros_like(fq)
            # Four-point method numerical derivative
            dfdq[2:-2] = (-fq[4:] + 8*fq[3:-1] - 8*fq[1:-3] + fq[:-4])/(12*dq) 
            """
            Below we assume that the distribution function at the edges of the q grid always behaves as an equilibrium one
            and use an analytical expression for the derivative (to avoid instabilities)
            """
            dfdq[:2] = (2/q[:2] - q[:2]/eq[:2])*fq[:2]
            dfdq[-2:] = (2/q[-2:] - q[-2:]/eq[-2:])*fq[-2:]
                
            #print(dfdq)
            
            dfq_x = ( gtilda(self._m/x)*( q*dfdq - 2*fq ) + q**2*(self._CI(x,q,fq,f_eq)/(2*self._g*eq ))/H_t(self._m/x) )/x

            #print(dfq_x)

            return dfq_x
    
        start_time = time.time()

        # Solving a system of ODEs for each momentum mode
        fBE_sol = solve_ivp(fBE_RHS,[x[0], x[-1]], f0, t_eval = x, **solver_options)

        end_time = time.time()
        elapsed_time = end_time - start_time

        #print(fBE_sol)
        if fBE_sol.success:
            print(f"fBE solved in {elapsed_time:.2f} seconds")
        
        self._f = np.transpose(fBE_sol.y)

    # ***************************************************************************************************
    
 
    def getDensity(self) -> np.array:
        # Return the comoving density values from the solution of the fBE (by using a trapezoidal rule to integrate the distribution accordingly) [1]
        
        Y_pde = self._g*np.trapz(self._f, self._q, axis=1)*(self._m/self._x)**3/2/np.pi**2/s_ent(self._m/self._x)
    
        return Y_pde

    
    def solve_nBE(self, x, Rate, Y0):
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
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
                logging.info(f"Progress: {x}/{x_end}, {progress:.2f}% - Elapsed time: {elapsed_time:.2f} seconds")
                last_log_time = current_time
            
            return Rate(x, Y) * (H_t(self._m/x) * s_ent(self._m/x) * x)**(-1)

        logging.info("Starting solve_nBE...")

        # Solve the differential equation
        sol = solve_ivp(
            YBE_RHS,
            [x_start, x_end],
            [Y0],
            t_eval=x,
            method='Radau',
            rtol=1e-6, 
            atol=1e-8,  
            vectorized=True  
        )

        if not sol.success:
            logging.error(f"Integration failed: {sol.message}")
        else:
            end_time = time.time()
            total_time = end_time - start_time
            logging.info(f"solve_nBE completed in {total_time:.2f} seconds")

        return sol.y[0] if sol.success else None

    
    def plot2D(self):
        # Create a 2D color plot to represent the solution of the fBE in x and q

        mesh_q, mesh_x = np.meshgrid(self._q, self._x)
        
        plt.figure()
        plt.pcolormesh(mesh_q, mesh_x, self._f)
        plt.colorbar()
        plt.gca().set_title('fBE solution')
        plt.xlabel('q')
        plt.ylabel('x')
        plt.show()


    def plotFinalPDF(self):
        # Plot the distribution function at the end of the evolution

        plt.figure()
        plt.plot(self._q, self._f[-1,:])
        plt.plot(self._q, self.equilibriumFunction(self._q),'--')
        plt.xlabel(r'$q$')
        plt.ylabel(r'$q^2 f(q)$')
        plt.title("Final distribution function")
        plt.legend(["fBE", "Equilibrium"],loc="upper right")
        plt.grid()
        plt.show()








