import numpy as np
import matplotlib.pyplot as plt
import constants as const
import particle_physics as pp
from cosmology import H_t, H, s_ent, gtilda, Y_x_eq, Y_x_eq_massive, h_s
from scipy.integrate import quad, solve_ivp, fixed_quad # ODE solver
from scipy.interpolate import interp1d
import time
from tqdm import tqdm
import logging

# =================================================================================================================================

# Classes of processes
from scipy.special import kn # Bessel function

class DecayToX: # 1->2 decay where X is massless
    """
    A class for the 2-body decay processes with 1 (massless) DM particle.
    collisionTerm is used for the fBE solution, while the rate is used for the nBE solution.

    Author: Maxim Laletin
    """
    def __init__(self, m1: float, m2: float, g_1: float, Msquared: float, Gamma: float):
        """
        Parameters  
        ----------
        m1: float
            The mass of the decaying particle
        m2: float
            The mass of the daughter particle (not DM)
        g_1: float
            The number of dof of the DM particle
        Msquared: float
            The value of the amplitude squared SUMMED over all final states (constant in this case)
        Gamma: float
            The value of the decay width (can be actually computed from Msquared)
        """
        self._m1 = m1 
        self._m2 = m2 
        self._mu = m2/m1 # ratio of masses
        self._g_1 = g_1
        self._Msquared = Msquared 
        self._Gamma = Gamma 

    def collisionTerm(self, x: float, q: List[float], f: List[float]) -> List[float]:
        """
        The collision term below is for the production of a scalar massless particle in the decay with the mother particle and the other daugther particle being fermions

        Parameters  
        ----------
        x: float 
            The inverse unit of temperature x = m/T
        q: List[float]
            The vector of momenta of particle X divided by T
        f: List[float]
            The distribution function of X particle (vector should be the same size as q)  
        """
        # The following structure comes from the integration limits
        Elim1 = np.max([x*np.ones(q.shape), x*self._mu+q, x**2*(1 + 4*q**2/(x**2*(1-self._mu**2)) - self._mu**2 )/4/q],axis=0)
        Elim2 = np.max([x-q, x*self._mu*np.ones(q.shape), x**2/4/q],axis=0)
        # Distribution function for x 
        fxeq = q**2/(np.exp(q)-1.0) 

        return (2*self._g_1*self._Msquared*x/self._m1/8/np.pi/q**3)*(np.log((1+np.exp(-Elim1))/(1+np.exp(-Elim2))))*(f - fxeq)

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

# Class of the solver
class Model: 
    def __init__(self, m, g_dof, p_type, x=np.linspace(1.0,10.0,10), q=np.linspace(0.1,20.,10)):
        self._m = m # m is not (!) the mass of the field, but rather the mass in the relation x = m/T
        self._g = g_dof # g_dof is the number of field's degrees of freedom
        self._p_type = p_type # particle's spin statistics (b = boson, f = fermion, m = Maxwell-Boltzmann particle)
        self._x = x # x should be a numpy vector (check!)
        self._q = q # q should be a numpy vector
        # We can change the grid (x,q) later 
        self._f = np.zeros_like([self._x, self._q]) # solution of fBE - matrix of size Nx*Nq
        #model._f = np.zeros((np.size(x),np.size(q)))

    def getX(self):
        return self._x

    def getQ(self):
        return self._q

    def getSolution(self):
        return self._f

    def changeGrid(self, x_new, q_new):
        self._x = x_new
        self._q = q_new
        self._f = np.zeros((np.size(x_new),np.size(q_new)))

    # Initial Collision Term of the model
    def _CI(self, x, q, f):
        return 0.0

    # Modify the Collision Term of the model
    def addCollisionTerm(self, CollTerm): # !!! CHECK IF THIS FUNCTION ASSIGNMENT AIN'T TOO SLOW !!!

        original_CI = self._CI

        def combined_CI(x,q,f):
            return original_CI(x,q,f) + CollTerm(x,q,f)
        
        self._CI = combined_CI

    # ***************************************************************************************************
    # Non-Adaptive ODE system method Solver for the Full Boltzmann Equation
    def solve_fBE(self, f0):
        # x and q are vectors
        # CI is the collision integral - should be a function that takes two values (x,q)
        # f0 should be a vector of initial values (of the same length as q)

        # Refine the model's values
        #model._x = x
        #model._q = q
        #model._f = np.zeros_like([x_new,q_new])
        x = self._x
        q = self._q
    
        # Sizes of x and q arrays
        sx = x.size
        sq = q.size
        
        # Size of q cell
        dq = q[-1]-q[-2]
        
        # Solution
        #f = np.zeros((sx,sq))
        
        # Equilibrium distribution
        f_eq = q**2/(np.exp(q)-1.0)
        #f_eq = lambda t: t**2/(np.exp(t)-1.0)
        #f_in = np.zeros(q.shape)
    
        # Order of the finite difference method
        order = 5
    
        def fBE_RHS(x, fq):
            dfq_x = np.zeros(sq)
            dfdq = np.zeros(sq)
            #for j in range(1,sq):
                #dfdq[j] = (fq[j]-fq[j-1])/dq
    
            for j in range(1,order//2):
                  dfdq[j] = (fq[j]-fq[j-1])/dq
    
            for j in range(order//2,sq-order//2):
                  dfdq[j] = (-fq[j+2] + 8*fq[j+1] - 8*fq[j-1] + fq[j-2])/12/dq
    
            for j in range(sq-order//2,sq):
                  dfdq[j] = fq[j-1]*(1.0 + dq/2 + dq**2/6 + dq**3/24 + dq**4/120) # exponential tail
                
            #dfq_x[j] = ( gtilda(m1/x)*( q[j]*dfdq[j] - 2*fq[j] ) + q[j]**2*CollInt_dec(x,q[j])*(fq[j] - f_eq[j])/H_t(m1/x) )/x
            dfq_x = ( gtilda(self._m/x)*( q*dfdq - 2*fq ) + q**2*(self._CI(x,q,fq)/(2*self._g*q ))/H_t(self._m/x) )/x
            # CHANGE q IN THE DENOMINATOR BELOW COLLISION TERM TO THE ENERGY OF THAT STATE

            return dfq_x
    
        
        fBE_sol = solve_ivp(fBE_RHS,[x[0], x[-1]], f0, t_eval = x).y
    
        self._f = np.transpose(fBE_sol)
    # ***************************************************************************************************
    
 

    # Get the density from the PDF (using trapezoidal rule)
    def getDensity(self):
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

        mesh_q, mesh_x = np.meshgrid(self._q, self._x)
        
        plt.figure()
        plt.pcolormesh(mesh_q, mesh_x, self._f)
        plt.colorbar()
        plt.gca().set_title('PDE solution')
        plt.xlabel('q')
        plt.ylabel('x')
        plt.show()


    def plotFinalPDF(self):

        plt.figure()
        plt.plot(self._q, self._f[-1,:])
        plt.plot(self._q, self._q**2/(np.exp(self._q) - 1.0),'--')
        plt.xlabel('q')
        plt.ylabel('q^2 f(q)')
        plt.legend(["PDF Final", "Equilibrium"],loc="upper right")
        plt.grid()
        plt.show()








