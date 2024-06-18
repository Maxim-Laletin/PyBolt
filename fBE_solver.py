import numpy as np
import matplotlib.pyplot as plt
from cosmology import H_t, H, s_ent, gtilda, Y_x_eq
from scipy.integrate import solve_ivp # ODE solver

# =================================================================================================================================

# Classes of processes
from scipy.special import kn # Bessel function

class DecayToX: # 1->2 decay where X is massless
    def __init__(process,m1,m2,g_1,Msquared,Gamma):
        process._m1 = m1 # mass of the decaying particle
        process._m2 = m2 # mass of the daughter particle
        process._mu = m2/m1 # ratio of masses !!! (CHECK THAT M1 AND M2 ARE SANE) !!!
        process._g_1 = g_1 # dof of the decaying particle
        process._Msquared = Msquared # squared amplitude of the decay (summed over all final states)
        process._Gamma = Gamma # !!! CAN BE ACTUALLY CALCULATED FROM MSQUARED

    def collisionTerm(proc,x,q,f):
        # Limits in the collision term
        Elim1 = np.max([x*np.ones(q.shape), x*proc._mu+q, x**2*(1 + 4*q**2/(x**2*(1-proc._mu**2)) - proc._mu**2 )/4/q],axis=0)
        Elim2 = np.max([x-q, x*proc._mu*np.ones(q.shape), x**2/4/q],axis=0)
        #Elim1 = np.max([x, x*mu+q, x**2*(1 + 4*q**2/(x**2*(1-mu**2)) - mu**2 )/4/q],axis=0)
        #Elim2 = np.max([x-q, x*mu, x**2/4/q],axis=0)
        
        fxeq = q**2/(np.exp(q)-1.0) # !!! NEED TO PROPERLY INCLUDE THE X PARTICLE SPIN STATISTICS !!!

        return (2*proc._g_1*proc._Msquared*x/proc._m1/8/np.pi/q**3)*(np.log((1+np.exp(-Elim1))/(1+np.exp(-Elim2))))*(f - fxeq)
        # !!! Check whether all the factors correspond to the definition: (left-hand side fBE ) = collisionTerm(x,q,f)/(2*g_a*E_a) !!!

    def rate(proc,x,Y):
        # with a Bessel function (MB distribution for decaying particle)
        return 8*proc._Gamma*proc._m1**3*(x*kn(1,x))*(1 - Y/Y_x_eq(proc._m1/x))/(2*np.pi)**2 # !!! CHECK THE NUMBERS OF DEGREES OF FREEDOM !!!
        

# =================================================================================================================================

# Class of the solver
class Model: 
    def __init__(model, m, g_dof, p_type, x=np.linspace(1.0,10.0,10), q=np.linspace(0.1,20.,10)):
        model._m = m # m is not (!) the mass of the field, but rather the mass in the relation x = m/T
        model._g = g_dof # g_dof is the number of field's degrees of freedom
        model._p_type = p_type # particle's spin statistics (b = boson, f = fermion, m = Maxwell-Boltzmann particle)
        model._x = x # x should be a numpy vector (check!)
        model._q = q # q should be a numpy vector
        # We can change the grid (x,q) later 
        model._f = np.zeros_like([model._x, model._q]) # solution of fBE - matrix of size Nx*Nq
        #model._f = np.zeros((np.size(x),np.size(q)))

    def getX(model):
        return model._x

    def getQ(model):
        return model._q

    def getSolution(model):
        return model._f

    def changeGrid(model,x_new,q_new):
        model._x = x_new
        model._q = q_new
        model._f = np.zeros((np.size(x_new),np.size(q_new)))

    # Initial Collision Term of the model
    def _CI(model,x,q,f):
        return 0.0

    # Modify the Collision Term of the model
    def addCollisionTerm(model,CollTerm): # !!! CHECK IF THIS FUNCTION ASSIGNMENT AIN'T TOO SLOW !!!

        original_CI = model._CI

        def combined_CI(x,q,f):
            return original_CI(x,q,f) + CollTerm(x,q,f)
        
        model._CI = combined_CI

    # ***************************************************************************************************
    # Non-Adaptive ODE system method Solver for the Full Boltzmann Equation
    def solve_fBE(model,f0):
        # x and q are vectors
        # CI is the collision integral - should be a function that takes two values (x,q)
        # f0 should be a vector of initial values (of the same length as q)

        # Refine the model's values
        #model._x = x
        #model._q = q
        #model._f = np.zeros_like([x_new,q_new])
        x = model._x
        q = model._q
    
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
            dfq_x = ( gtilda(model._m/x)*( q*dfdq - 2*fq ) + q**2*(model._CI(x,q,fq)/(2*model._g*q))/H_t(model._m/x) )/x
    
            return dfq_x
    
        
        fBE_sol = solve_ivp(fBE_RHS,[x[0], x[-1]], f0, t_eval = x).y
    
        model._f = np.transpose(fBE_sol)
    # ***************************************************************************************************
    
 

    # Get the density from the PDF (using trapezoidal rule)
    def getDensity(model):
        Y_pde = model._g*np.trapz(model._f,model._q, axis=1)*(model._m/model._x)**3/2/np.pi**2/s_ent(model._m/model._x)
    
        return Y_pde

    
    # Solve the standard nBE (using scipy.integrate.solve_ivp)
    def solve_nBE(model,x,Rate,Y0):
    
        # RHS of the Boltzmann equation
        #YBE_RHS = lambda t,y: 8*Gamma*m1**3*qintode(t)[0]*(1 - y/Y_x_eq(m1/t)/g_x)*((2*np.pi)**2*H(m1/t)*s_ent(m1/t)*t**3)**(-1)
        YBE_RHS = lambda x,Y: Rate(x,Y)*(H(model._m/x)*s_ent(model._m/x)*x**3)**(-1) 
    
        sol =  solve_ivp(YBE_RHS,[x[0], x[-1]], [Y0], t_eval = x)
    
        return sol.y[0]

    
    def plot2D(model):

        mesh_q,mesh_x = np.meshgrid(model._q,model._x)
        
        plt.figure()
        plt.pcolormesh(mesh_q,mesh_x,model._f)
        plt.colorbar()
        plt.gca().set_title('PDE solution')
        plt.xlabel('q')
        plt.ylabel('x')
        plt.show()


    def plotFinalPDF(model):

        plt.figure()
        plt.plot(model._q,model._f[-1,:])
        plt.plot(model._q,model._q**2/(np.exp(model._q) - 1.0),'--')
        plt.xlabel('q')
        plt.ylabel('q^2 f(q)')
        plt.legend(["PDF Final", "Equilibrium"],loc="upper right")
        plt.grid()
        plt.show()








