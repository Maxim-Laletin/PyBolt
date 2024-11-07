import numpy as np
from .constants import MPL, Zeta3
from scipy.interpolate import CubicSpline  # interpolation

# == Extracting the entropy degrees of freedom (taken from 1606.07494) ==

# | log10(T/MeV) | g_rho | g_rho/g_s |
dof_arr = np.array(
    [
        [0.00, 10.71, 1.00228],
        [0.50, 10.74, 1.00029],
        [1.00, 10.76, 1.00048],
        [1.25, 11.09, 1.00505],
        [1.60, 13.68, 1.02159],
        [2.00, 17.61, 1.02324],
        [2.15, 24.07, 1.05423],
        [2.20, 29.84, 1.07578],
        [2.40, 47.83, 1.06118],
        [2.50, 53.04, 1.04690],
        [3.00, 73.48, 1.01778],
        [4.00, 83.10, 1.00123],
        [4.30, 85.56, 1.00389],
        [4.60, 91.97, 1.00887],
        [5.00, 102.17, 1.00750],
        [5.45, 104.98, 1.00023],
    ]
)


gx = dof_arr[:, 0] - 3  # log10(T/GeV) points
gy = dof_arr[:, 1]  # g points
hy = dof_arr[:, 1] / dof_arr[:, 2]  # h points

# Constructing the cubic splines for interpolation
gy_spline = CubicSpline(gx, gy)
hy_spline = CubicSpline(gx, hy)

# Functions for density and entropy dofs
g_rho = lambda T: gy_spline(np.log10(T))
h_s = lambda T: hy_spline(np.log10(T))

# gx_probe = np.logspace(np.log10(gx[0]),np.log10(gx[-1]),100)
gx_probe = np.linspace(gx[0], gx[-1], 100)

# Creating a spline for log hs as a function of log T
hslog = CubicSpline(dof_arr[:, 0] - 3, np.log10(hy))

# Derivative of the spline
dloghdlogT = hslog.derivative()
# Function for gtilda
gtilda = lambda T: (1 / 3) * dloghdlogT(np.log10(T))

# ============== Cosmological quantities ==============

# Hubble parameter
H = lambda T: T**2 / (MPL / np.sqrt(8 * np.pi**3.0 * g_rho(T) / 90))  # GeV

# Reduced Hubble parameter \tilde{H}
H_t = lambda T: H(T) / (1 + gtilda(T))  # GeV

# Entropy density
s_ent = lambda T: h_s(T) * 4 * np.pi**2 * T**3 / 90
# GeV^3

# Comoving equilibrium number density of massless (relativistic) particles [per number of degrees of freedom (!)]
Y_x_eq = lambda T: 90.0 * Zeta3 / (h_s(T) * 4 * np.pi**4)  # 1
# Comoving equilibrium number density of massive non-relativistic particles [per number of degrees of freedom (!)]
Y_x_eq_massive = (
    lambda T, m: 45.0
    / (2 * np.pi**4 * h_s(T))
    * (np.pi / 8) ** (1 / 2)
    * (m / T) ** (3 / 2)
    * np.exp(-m / T)
)  # 1
