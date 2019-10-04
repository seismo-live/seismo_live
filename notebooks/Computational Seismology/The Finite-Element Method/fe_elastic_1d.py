# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# <div style='background-image: url("../../share/images/header.svg") ; padding: 0px ; background-size: cover ; border-radius: 5px ; height: 250px'>
#     <div style="float: right ; margin: 50px ; padding: 20px ; background: rgba(255 , 255 , 255 , 0.7) ; width: 50% ; height: 150px">
#         <div style="position: relative ; top: 50% ; transform: translatey(-50%)">
#             <div style="font-size: xx-large ; font-weight: 900 ; color: rgba(0 , 0 , 0 , 0.8) ; line-height: 100%">Computational Seismology</div>
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)">Finite Element Method - 1D Elastic Wave Equation</div>
#         </div>
#     </div>
# </div>

# <p style="width:20%;float:right;padding-left:50px">
# <img src=../../share/images/book.jpg>
# <span style="font-size:smaller">
# </span>
# </p>
#
#
# ---
#
# This notebook is part of the supplementary material 
# to [Computational Seismology: A Practical Introduction](https://global.oup.com/academic/product/computational-seismology-9780198717416?cc=de&lang=en&#), 
# Oxford University Press, 2016.
#
#
# ##### Authors:
# * David Vargas ([@dvargas](https://github.com/davofis))
# * Heiner Igel ([@heinerigel](https://github.com/heinerigel))

# ## Basic Equations
# This notebook presents a finite element code for the 1D elastic wave equation. Additionally, a solution using finite difference scheme is given for comparison.
#
# The problem of solving the wave equation
#
# \begin{equation}
# \rho(x) \partial_t^2 u(x,t) = \partial_x (\mu(x) \partial_x u(x,t)) + f(x,t)
# \end{equation}
#
# using the finite element method is done after a series of steps performed on the above equation.
#
# 1) We first obtain a weak form of the wave equation by integrating over the entire physical domain $D$ and at the same time multiplying by some basis $\varphi_{i}$. 
#
# 2) Integration by parts and implementation of the stress-free boundary condition is performed.
#
# 3) We approximate our unknown displacement field $u(x, t)$ by a sum over space-dependent basis functions $\varphi_i$ weighted by time-dependent coefficients $u_i(t)$.
#
# \begin{equation}
# u(x,t) \ \approx \ \overline{u}(x,t) \ = \ \sum_{i=1}^{n} u_i(t) \ \varphi_i(x)
# \end{equation}
#
# 4) Utilize the same basis functions used to expand $u(x, t)$ as test functions in the weak form, this is the Galerkin principle.
#
# 5) We can turn the continuous weak form into a system of linear equations by considering the approximated displacement field.
#
# \begin{equation}
# \mathbf{M}^T\partial_t^2 \mathbf{u} + \mathbf{K}^T\mathbf{u} = \mathbf{f}
# \end{equation}
#
# 6) For the second time-derivative, we use a standard finite-difference approximation. Finally, we arrive at the explicit time extrapolation scheme.
#
# \begin{equation}
# \mathbf{u}(t + dt) = dt^2 (\mathbf{M}^T)^{-1}[\mathbf{f} - \mathbf{K}^T\mathbf{u}] + 2\mathbf{u} - \mathbf{u}(t-dt).
# \end{equation}
#
# where $\mathbf{M}$ is known as the mass matrix, and $\mathbf{K}$ the stiffness matrix.
#
# 7) As interpolating functions, we choose interpolants such that $\varphi_{i}(x_{i}) = 1$ and zero elsewhere. Then, we transform the space coordinate into a local system. According to $\xi = x − x_{i}$ and $h_{i} = x_{i+1} − x_{i}$, we have:
#
# <p style="width:35%;float:right;padding-left:50px">
# <img src=fig_fe_basis_h.png>
# <span style="font-size:smaller">
# </span>
# </p>
#
# \begin{equation}
#  \varphi_{i}(\xi) =
#   \begin{cases}
#     \frac{\xi}{h_{i-1}} + 1  & \quad \text{if} \quad -h_{i-1} \le \xi \le 0\\
#     1 + \frac{\xi}{h_{i}}    & \quad \text{if} \quad 0 \le \xi \le h_{i}\\
#     0  & \quad elsewhere\\
#   \end{cases}
# \end{equation}
#
# with the corresponding derivatives
#
# \begin{equation}
#  \partial_{\xi}\varphi_{i}(\xi) =
#   \begin{cases}
#     \frac{1}{h_{i-1}}  & \quad \text{if} \quad -h_{i-1} \le \xi \le 0\\
#    -\frac{1}{h_{i}}    & \quad \text{if} \quad 0 \le \xi \le h_{i}\\
#     0  & \quad elsewhere\\
#   \end{cases}
# \end{equation}
#
# The figure on the left-hand side illustrates the shape of $\varphi_{i}(\xi)$ and $\partial_{\xi}\varphi_{i}(\xi)$ with varying $h$.
#
# Code implementation starts with the initialization of a particular setup of our problem. Then, we define the source that introduces perturbations following by initialization of the mass and stiffness matrices. Finally, time extrapolation is done.

# + {"code_folding": [0]}
# Import all necessary libraries, this is a configuration step for the exercise.
# Please run it before the simulation code!
import numpy as np
import matplotlib
# Show Plot in The Notebook
matplotlib.use("nbagg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -

# ### 1. Initialization of setup

# + {"code_folding": []}
# Initialization of setup
# ---------------------------------------------------------------
# Basic parameters
nt    = 2000    # Number of time steps
vs    = 3000    # Wave velocity [m/s] 
ro0   = 2500    # Density [kg/m^3]
nx    = 1000    # Number of grid points  
isx   = 500     # Source location [m] 
xmax  = 10000.  # Maximum length
eps   = 0.5     # Stability limit
iplot = 20      # Snapshot frequency

dx = xmax/(nx-1)           # calculate space increment
x  = np.arange(0, nx)*dx   # initialize space coordinates
x  = np.transpose(x)

h = np.diff(x)  # Element sizes [m]

# parameters
ro = x*0 + ro0
mu = x*0 + ro*vs**2

# time step from stabiity criterion
dt = 0.5*eps*dx/np.max(np.sqrt(mu/ro))
# initialize time axis
t   = np.arange(1, nt+1)*dt  

# ---------------------------------------------------------------
# Initialize fields
# ---------------------------------------------------------------
u    = np.zeros(nx)
uold = np.zeros(nx)
unew = np.zeros(nx)

p    = np.zeros(nx)
pold = np.zeros(nx)
pnew = np.zeros(nx)
# -

# ### 2. Source time function
#
# In 1D the propagating signal is an integral of the source time function. As we look for a Gaussian waveform, we initialize the source time function $f(t)$ using the first derivative of a Gaussian function.
#
# \begin{equation}
# f(t) = -\dfrac{2}{\sigma^2}(t - t_0)e^{-\dfrac{(t - t_0)^2}{\sigma^2}}
# \end{equation}
#  
# #### Exercise 1 
# Initialize a source time function called 'src'.  Use $\sigma = 20 dt$ as Gaussian width, and time shift $t_0 = 3\sigma$. Then, visualize the source in a given plot.

# + {"code_folding": []}
#################################################################
# INITIALIZE THE SOURCE TIME FUCTION HERE!
#################################################################

# Source vector
f = np.zeros(nx); f[isx:isx+1] = f[isx:isx+1] + 1.

#################################################################
# PLOT THE SOURCE TIME FUNCTION HERE!
#################################################################
# -

# ### 3. The Mass Matrix
# Having implemented the desired source, now we initialize the mass and stiffness matrices. In general, the mass matrix is given
#
# \begin{equation}
# M_{ij} = \int_{D} \rho \varphi_i \varphi_j dx = \int_{D_{\xi}} \rho \varphi_i \varphi_j d\xi
# \end{equation}
#
# next, the defined basis are introduced and some algebraic treatment is done to arrive at the explicit form of the mass matrix
#
# #### Exercise 2 
# Implement the mass matrix 
#
# \begin{equation}
# M_{ij} = \frac{\rho h}{6}
#  \begin{pmatrix}
#    \ddots  &    &    &    & 0\\
#    1 & 4 &  1 &    &  \\
#      & 1 &  4 &  1 &  \\
#      &   &  1 &  4 & 1\\
#    0 &   &    &    &  \ddots
#  \end{pmatrix} 
# \end{equation}
#
# Compute the inverse mass matrix and display your result to visually inspect how it looks like

# +
#################################################################
# IMPLEMENT THE MASS MATRIX HERE!
#################################################################


#################################################################
# COMPUTE THE INVERSE MASS MATRIX HERE!
#################################################################


#################################################################
# DISPLAY THE INVERSE MASS MATRIX HERE!
#################################################################


# -

# ### 4. The Stiffness matrix
# On the other hand, the general form of the stiffness matrix is
#
# \begin{equation}
# K_{ij} = \int_{D} \mu \partial_x\varphi_i \partial_x\varphi_j dx = \int_{D_{\xi}} \mu \partial_\xi\varphi_i \partial_\xi\varphi_j d\xi
# \end{equation} 
#
# at this point, the defined basis are introduced. Again, with the help of some algebraic treatment, we arrive at the explicit form of the stiffness matrix
#
# #### Exercise 3 
# Implement the stiffness matrix 
#
# \begin{equation}
# K_{ij} = \frac{\mu}{h}
#  \begin{pmatrix}
#    \ddots  &    &    &    & 0\\
#   -1 & 2 & -1 &    &  \\
#      &-1 &  2 & -1 &  \\
#      &   & -1 &  2 & -1\\
#    0 &   &    &    &  \ddots
#  \end{pmatrix} 
# \end{equation}
#
# Display the stiffness matrix to visually inspect how it looks like

# + {"code_folding": []}
#################################################################
# IMPLEMENT THE STIFFNESS MATRIX HERE!
#################################################################


#################################################################
# DISPLAY THE STIFFNESS MATRIX HERE!
#################################################################


# -

# ### 5. Finite differences matrices
# We implement a finite difference scheme in order to compare with the finite elements solution. 
#
# #### Exercise 4 
# Implement the finite differences matrices $M$ and $D$. Where $M$ is a diagonal mass matrix containing the inverse densities, and differentiation matrix 
#
# \begin{equation}
# D_{ij} = \frac{\mu}{dt^2}
#  \begin{pmatrix}
#    -2 & 1  &    &    &  \\
#     1 & -2 & 1  &    &  \\
#       &  & \ddots  &  & \\
#       &   & 1 &  -2 & 1\\
#       &   &    &  1 & -2
#  \end{pmatrix} 
# \end{equation}
#
# Display both matrices to visually inspect how they look like

# +
#################################################################
# INITIALIZE FINITE DIFFERENCES HERE!
#################################################################


#################################################################
# DISPLAY THE DIFFERENCES MATRICES HERE!
#################################################################


# -

# ### 6. Finite element solution 
#
# Finally we implement the finite element solution using the computed mass $M$ and stiffness $K$ matrices together with a finite differences extrapolation scheme
#
# \begin{equation}
# \mathbf{u}(t + dt) = dt^2 (\mathbf{M}^T)^{-1}[\mathbf{f} - \mathbf{K}^T\mathbf{u}] + 2\mathbf{u} - \mathbf{u}(t-dt).
# \end{equation}

# + {"code_folding": [29]}
# Initialize animated plot
# ---------------------------------------------------------------
plt.figure(figsize=(12,4))

line1 = plt.plot(x, u, 'k', lw=1.5, label='FEM')
line2 = plt.plot(x, p, 'r', lw=1.5, label='FDM')
plt.title('Finite elements 1D Animation', fontsize=16)
plt.ylabel('Amplitude', fontsize=12)
plt.xlabel('x (m)', fontsize=12)

plt.ion()   # set interective mode
plt.show()

# ---------------------------------------------------------------
# Time extrapolation
# ---------------------------------------------------------------
for it in range(nt):
    # --------------------------------------
    # Finite Element Method
    unew = (dt**2) * Minv @ (f*src[it] - K @ u) + 2*u - uold                         
    uold, u = u, unew
    
    # --------------------------------------
    # Finite Difference Method
    pnew = (dt**2) * Mf @ (D @ p + f/dx*src[it]) + 2*p - pold
    pold, p = p, pnew
     
    # --------------------------------------   
    # Animation plot. Display both solutions
    if not it % iplot:
        for l in line1:
            l.remove()
            del l
        for l in line2:
            l.remove()
            del l
        line1 = plt.plot(x, u, 'k', lw=1.5, label='FEM')
        line2 = plt.plot(x, p, 'r', lw=1.5, label='FDM')
        plt.legend()
        plt.gcf().canvas.draw()
# -


