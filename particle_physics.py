### constants, functions and conventions for particle physics

import numpy as np
from scipy.interpolate import CubicSpline # interpolation

###### Constants

### Higgs sector

# Higgs mass
higgs_mass = 125.09 # GeV

# Higgs vacuum expectation value
v_0 = 246 #GeV

###### Functions

### Higgs decay widths

# Total
Gamma_h_tot = 4.042e-3 # GeV

# Partial
def Gamma_h(s):
    data = np.loadtxt('Hwidth.dat', skiprows=4)
    x, y = data[:, 0], data[:, 1]
    cs = CubicSpline(x, y, bc_type='natural')
    return cs(s)



    


