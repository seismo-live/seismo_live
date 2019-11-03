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

# <div style='background-image: url("../../../share/images/header.svg") ; padding: 0px ; background-size: cover ; border-radius: 5px ; height: 250px'>
#     <div style="float: right ; margin: 50px ; padding: 20px ; background: rgba(255 , 255 , 255 , 0.7) ; width: 50% ; height: 150px">
#         <div style="position: relative ; top: 50% ; transform: translatey(-50%)">
#             <div style="font-size: xx-large ; font-weight: 900 ; color: rgba(0 , 0 , 0 , 0.8) ; line-height: 100%">Computational Seismology</div>
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)">Seismic Wavefield of a Double-Couple Point Source</div>
#         </div>
#     </div>
# </div>

# <p style="width:20%;float:right;padding-left:50px">
# <img src=../../../share/images/book.jpg>
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
# ##### Authors:
# * David Vargas ([@dvargas](https://github.com/davofis))
# * Heiner Igel ([@heinerigel](https://github.com/heinerigel))

# ## Basic Equations
#
# The fundamental analytical solution to the problem of a double couple point source in infinite homogeneous media (Aki and Richards - 2002) is implemented in this Ipython notebook. This solution of seimic waves in an infinite homogeneous medium provide fundamentamental information used as benchmark to understand kinematic properties of seismic sources, quasi-analytical solutions to wave propagation problems, and influence of earthquakes on crustal deformation. 
#
# Simulations of 3D elastic wave propagation need to be validated by the use of analytical solutions. In order to evaluate how healty a numerical solution is, one may recreate conditions for which analytical solutions exist with the aim of reproduce and compare the different results. In this sense, the fundamental solution for the  double couple point source offers an alternative to achieve this quality control.
#
# We want to find the displacement wavefield $\mathbf{u}(\mathbf{x},t)$ at some distance $\mathbf{x}$ from a seismic moment tensor source with $M_xz = M_zx = M_0$. According to (Aki and Richards 2002), the displacement $\mathbf{u}(\mathbf{x},t)$ due to a double-couple point source in an infinite, homogeneous, isotropic medium is
#
# \begin{align*}
# \mathbf{u}(\mathbf{x},t) &= \dfrac{1}{4\pi\rho} \mathbf{A}^N \dfrac{1}{r^4} \int_{{r}/{\alpha}}^{{r}/{\beta}} \tau M_o(t-\tau)d\tau +\\
# &+\dfrac{1}{4\pi\rho\alpha^2}\mathbf{A}^{IP}\dfrac{1}{r^2} M_o(t-{r}/{\alpha}) +\dfrac{1}{4\pi\rho\beta^2}\mathbf{A}^{IS}\dfrac{1}{r^2} M_o(t-{r}/{\beta})+\\
# &+\dfrac{1}{4\pi\rho\alpha^3}\mathbf{A}^{FP}\dfrac{1}{r} \dot M_o(t-{r}/{\alpha}) +\dfrac{1}{4\pi\rho\beta^3}\mathbf{A}^{FS}\dfrac{1}{r} \dot M_o(t-{r}/{\beta})
# \end{align*}
#
# where the radiation patterns $\mathbf{A}^N$ (near-field), $\mathbf{A}^{IP}$ (intermediate-field P wave), $\mathbf{A}^{IS}$ (intermediate-field S wave), $\mathbf{A}^{FP}$ (far-field P wave) and $\mathbf{A}^{FS}$ (far-field S wave) are:
#
# \begin{align*}
# \mathbf{A}^N &= 9sin(2\theta)cos(\phi)\hat{\mathbf{r}} - 6(cos(2\theta)cos(\phi)\hat{\mathbf{\theta}} - cos(\theta)sin(\phi))\hat{\mathbf{\phi}}\\
# \mathbf{A}^{IP} &= 4sin(2\theta)cos(\phi)\hat{\mathbf{r}} - 2(cos(2\theta)cos(\phi)\hat{\mathbf{\theta}} - cos(\theta)sin(\phi))\hat{\mathbf{\phi}}\\
# \mathbf{A}^{IS} &= -3sin(2\theta)cos(\phi)\hat{\mathbf{r}} + 3(cos(2\theta)cos(\phi)\hat{\mathbf{\theta}} - cos(\theta)sin(\phi))\hat{\mathbf{\phi}}\\ 
# \mathbf{A}^{FP} &= sin(2\theta)cos(\phi)\hat{\mathbf{r}}\\
# \mathbf{A}^{FS} &= cos(2\theta)cos(\phi)\hat{\mathbf{\theta}} - cos(\theta)sin(\phi)\hat{\mathbf{\phi}}
# \end{align*}
#
# The parameters one have to consider include: receiver coordinates $\mathbf{x}$, density of the medium $\rho$, S-Wave velocity $\beta$, p-wave velocity $\alpha$, and the desired time dependent seismic moment function $M_o(t)$.On the other hand, integrations limits are determined by the propagation time from source to receiver for both p-waves and s-waves ie. ${r}/{\beta}$ and ${r}/{\alpha}$ respectively.
#
# Similarly, the rotational motion $\mathbf{\Omega}$ (the curl of displacement) and the divergence of displacement can be written as
# \begin{align*}
# \mathbf{\Omega} & = \dfrac{1}{2} \nabla \times \mathbf{u} (\mathbf{x},t)
#  = \dfrac{-\mathbf{A}^R}{8 \pi \rho} \bigg[ \dfrac{3}{\beta^2 r^3} M_0 \bigg( t-\dfrac{r}{\beta} \bigg)  
#      + \dfrac{3}{\beta^3 r^2} \dot{M}_0\bigg( t-\dfrac{r}{\beta} \bigg) 
#      + \dfrac{1}{\beta^4 r} \ddot{M}_0\bigg( t-\dfrac{r}{\beta} \bigg)
#      \bigg] , \\[6pt]
# \nabla \cdot \mathbf{u} & = \dfrac{-\mathbf{A}^D}{4 \pi \rho} \bigg[ \dfrac{3}{\alpha^2 r^3} M_0 \bigg( t-\dfrac{r}{\alpha} \bigg)  
#      + \dfrac{3}{\alpha^3 r^2} \dot{M}_0\bigg( t-\dfrac{r}{\alpha} \bigg) 
#      + \dfrac{1}{\alpha^4 r} \ddot{M}_0\bigg( t-\dfrac{r}{\alpha} \bigg)
#      \bigg] ,
# \end{align*}
# where $\mathbf{A}^R = \cos \theta \sin \phi \, \boldsymbol{\hat{\theta}} + \cos \phi \cos 2\theta \, \boldsymbol{\hat{\phi}} $ is the radiantion patter of the three components of rotation (the radial component being zero) and $\mathbf{A}^D = \cos \phi \sin \theta \, \mathbf{\hat{r}}$ is the radiation pattern of the divergence.
#
#
# This a solution in spherical coordinates. Since we normally measure displacements in cartesian coordinates, it is necessary to implement a change of coordinates if we want to visualize the solution in cartesian coordinates. 
#

# + {"code_folding": []}
# Please run it before you start the simulation!
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.integrate import quad
from numpy import sin, cos, arccos, arctan,  pi, sign, sqrt, arctan2
from numpy import vectorize, linspace, asarray, outer, diff, savetxt
import numpy as np

# Show the plots in the Notebook.
plt.switch_backend("nbagg")


# -

# ## 2. Coordinate transformation methods

# + {"code_folding": []}
def sph2cart(r, th, phi):
    '''
    Transform spherical coordinates to cartesian
    '''
    x = r * sin(th) * cos(phi)
    y = r * sin(th) * sin(phi)
    z = r * cos(th)   
    return x, y, z

def sph2cart_vecProj(Ur, Uth, Uphi, th, phi):
    '''
    Transform spherical coordinates unit vectors to cartesian components
    '''
    xyz_r = outer(Ur,[sin(th) * cos(phi), sin(th) * sin(phi), cos(th)])
    xyz_th = outer(Uth,[cos(th) * cos(phi), cos(th) * sin(phi), -sin(th)])
    xyz_phi = outer(Uphi,[-sin(phi), cos(phi),  np.zeros(np.shape(th))])
    x = xyz_r[:,0] + xyz_th[:,0] + xyz_phi[:,0] 
    y = xyz_r[:,1] + xyz_th[:,1] + xyz_phi[:,1] 
    z = xyz_r[:,2] + xyz_th[:,2] + xyz_phi[:,2] 

    return x, y, z
    
def cart2sph(x, y, z):
    '''
    Transform cartesian coordinates to spherical
    '''
    r = sqrt(x**2 + y**2 + z**2)
    th = arccos(z/r)
    phi = arctan2(y,x)
    return r, th, phi 


# -

# ## 3. COMPUTE AKI & RICHARDS SOLUTION

# +
#%% Initialization of setup
# -----------------------------------------------------------------------------
x = 4000   # x receiver coordinate 
y = 0      # y receiver coodinate
z = 0   # z receiver coodinate

rho = 2500                # Density kg/m^3 
beta = 3000               # S-wave velocity
alpha = sqrt(3)*beta      # p-wave velocity

stf = 'gauss'             # Set the desired source time function 'heaviside' , 'gauss'
Trise = 0.05              # Rise time used in the source time function 
Mo =10E15                 # Scalar Moment 
    
r, th, phi = cart2sph(x, y, z)     # spherical receiver coordinates 


tmin = r/alpha - 3*Trise           # Minimum observation time 
tmax = r/beta + Trise + 5*Trise    # Maximum observation time 

# SOURCE TIME FUNCTION
# -----------------------------------------------------------------------------   
if stf == 'heaviside':
    M0 = lambda t: 0.5*Mo*0.5*(sign(t) + 1)
if stf == 'gauss':
    M0 = lambda t: Mo*(1 + erf(t/Trise))
#     M0 = lambda t: Mo*(-2 * t / Trise * np.exp(-(t/Trise)**2))

#******************************************************************************
# COMPUTE AKI & RICHARDS SOLUTION
#******************************************************************************
# Scalar factors int the AKI & RICHARDS solution
# -----------------------------------------------------------------------------
CN  = (1/(4 * pi * rho)) 
CIP = (1/(4 * pi * rho * alpha**2))
CIS = (1/(4 * pi * rho * beta**2))
CFP = (1/(4 * pi * rho * alpha**3))
CFS = (1/(4 * pi * rho * beta**3))

# Radiation patterns: near(AN), intermedia(AIP,AIS), and far(AFP,AFS) fields  
# -----------------------------------------------------------------------------
def AN(th, phi):    
    AN = [[9*sin(2*th)*cos(phi), -6*cos(2*th)*cos(phi), 6*cos(th)*sin(phi)]]
    return asarray(AN)
    
def AIP(th, phi):    
    AIP = [[4*sin(2*th)*cos(phi), -2*cos(2*th)*cos(phi), 2*cos(th)*sin(phi)]]
    return asarray(AIP)
    
def AIS(th, phi):    
    AIS = [-3*sin(2*th)*cos(phi), 3*cos(2*th)*cos(phi), -3*cos(th)*sin(phi)]
    return asarray(AIS)
    
def AFP(th, phi):    
    AFP = [sin(2*th)*cos(phi), 0, 0 ]
    return asarray(AFP)
    
def AFS(th, phi):    
    AFS = [0, cos(2*th)*cos(phi), -cos(th)*sin(phi)]
    return asarray(AFS)

def AR(th, phi):    
    AR = [0, cos(th)*sin(phi), cos(phi)*sin(2*th)]
    return asarray(AR)

def AS(th, phi):    
    AS = [cos(phi)*sin(2*th), 0, 0]
    return asarray(AS)

# Calculate integral in the right hand side of AKI & RICHARDS solution
# -----------------------------------------------------------------------------
integrand = lambda  tau, t: tau*M0(t - tau)

def integral(t):
    return quad(integrand, r/alpha, r/beta, args=(t))[0]

vec_integral = vectorize(integral)

# Assemble the total AKI & RICHARDS solution
# -----------------------------------------------------------------------------

# Displacement component

t = linspace(tmin, tmax, 1000) 
UN =   CN * (1/r**4) * outer(AN(th, phi), vec_integral(t))

UIP = CIP * (1/r**2) * outer(AIP(th, phi), M0(t - r/alpha)) 
UIS = CIS * (1/r**2) * outer(AIS(th, phi), M0(t - r/beta))
UI = UIP + UIS

t, dt = linspace(tmin, tmax, 1001, retstep=True) # diff() return N-1 size vector  
UFP = CFP * (1/r) * outer(AFP(th, phi), diff(M0(t - r/alpha))/dt)
UFS = CFS * (1/r) * outer(AFS(th, phi), diff(M0(t - r/beta))/dt)
UF = UFP + UFS
t = linspace(tmin, tmax, 1000) 

U = UN + UIP + UIS + UFP + UFS

UNr, UNth, UNphi = UN[0,:], UN[1,:], UN[2,:]  # spherical componets of the field u 
UIr, UIth, UIphi = UI[0,:], UI[1,:], UI[2,:]  # spherical componets of the field u 
UFr, UFth, UFphi = UF[0,:], UF[1,:], UF[2,:]  # spherical componets of the field u 
Ur, Uth, Uphi = U[0,:], U[1,:], U[2,:]  # spherical componets of the field u 
Ux, Uy, Uz = sph2cart_vecProj(Ur,Uth,Uphi,th, phi)

# Velocity component

t, dt = linspace(tmin, tmax, 1000, retstep=True) # 4 diff() return N size vector  

VNr = np.hstack([[0],np.diff(UNr)/dt]) 
VNth = np.hstack([[0],np.diff(UNth)/dt]) 
VNphi = np.hstack([[0],np.diff(UNphi)/dt])
VN = asarray([VNr, VNth, VNphi])

VIr = np.hstack([[0],np.diff(UIr)/dt]) 
VIth = np.hstack([[0],np.diff(UIth)/dt]) 
VIphi = np.hstack([[0],np.diff(UIphi)/dt])
VI = asarray([VIr, VIth, VIphi])

VFr = np.hstack([[0],np.diff(UFr)/dt]) 
VFth = np.hstack([[0],np.diff(UFth)/dt]) 
VFphi = np.hstack([[0],np.diff(UFphi)/dt])
VF = asarray([VFr, VFth, VFphi])

Vr = np.hstack([[0],np.diff(Ur)/dt]) 
Vth = np.hstack([[0],np.diff(Uth)/dt]) 
Vphi = np.hstack([[0],np.diff(Uphi)/dt])
V = asarray([Vr, Vth, Vphi])

Vx = np.hstack([[0],np.diff(Ux)/dt]) 
Vy = np.hstack([[0],np.diff(Uy)/dt]) 
Vz = np.hstack([[0],np.diff(Uz)/dt])  

t = linspace(tmin, tmax, 1000) 

# Rotation component (curl of displacement)

CRN = (-3/(8 * pi * rho * beta**2 * r**3)) 
CRI = (-3/(8 * pi * rho * beta**3 * r**2))
CRF = (-1/(8 * pi * rho * beta**4 * r))

t = linspace(tmin, tmax, 1000) 
RN = CRN * outer(AR(th, phi), M0(t - r/beta))
t, dt = linspace(tmin, tmax, 1001, retstep=True) # 4 diff() return N size vector  
RI = CRI * outer(AR(th, phi), diff(M0(t - r/beta))/dt)
t, dt = linspace(tmin, tmax, 1002, retstep=True) # 4 diff(diff()) return N size vector  
RF = CRF * outer(AR(th, phi), diff(diff(M0(t - r/beta)))/(dt**2))
t = linspace(tmin, tmax, 1000) 

R = RN + RI + RF

Rr, Rth, Rphi = R[0,:], R[1,:], R[2,:]  # spherical componets of the field u 
Rx, Ry, Rz = sph2cart_vecProj(Rr, Rth, Rphi,th, phi)    # spherical to cartesian coordinates

# Divergence component

CSN = (-3/(4 * pi * rho * alpha**2 * r**3)) 
CSI = (-3/(4 * pi * rho * alpha**3 * r**2))
CSF = (-1/(4 * pi * rho * alpha**4 * r))

t = linspace(tmin, tmax, 1000) 
SN = CSN * outer(AS(th, phi), M0(t - r/alpha))
t, dt = linspace(tmin, tmax, 1001, retstep=True) # 4 diff() return N size vector  
SI = CSI * outer(AS(th, phi), diff(M0(t - r/alpha))/dt)
t, dt = linspace(tmin, tmax, 1002, retstep=True) # 4 diff(diff()) return N size vector  
SF = CSF * outer(AS(th, phi), diff(diff(M0(t - r/alpha)))/(dt**2))
t = linspace(tmin, tmax, 1000) 

S = SN + SI + SF

# -

# ## 4. Plot displacement components 

# +
# Plotting
# -----------------------------------------------------------------------------
seis = [Ux, Uy, Uz, Ur, Uth, Uphi]  # Collection of seismograms
labels = ['$u_x(t)$','$u_y(t)$','$u_z(t)$','$u_r(t)$','$u_{\\theta}(t)$','$u_{\\phi}(t)$']
cols = ['b','r','k','g','c','m']

# Initialize animated plot
fig = plt.figure(figsize=(10,5), dpi=100)
fig.suptitle("Seismic displacement wavefield of a Double-Couple Point Source", fontsize=16)
# plt.ion() # set interective mode
# plt.show()

for i in range(6):              
    st = seis[i]
    ax = fig.add_subplot(2, 3, i+1)
    ax.plot(t, st, lw = 1.5, color=cols[i])  
    ax.set_xlabel('Time(s)')
    ax.text(tmin, 0.7*max(st), labels[i])
#     ax.set_ylim(-0.05,0.05)
    
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
# -

# ## 5. Plot displacement, velocity, rotation and divergence components

# +

titles = ['Near','Intermediate','Far','Total']
ylabels = ['Displacement (m)', 'Velocity (m/s)', 'Rotation (rad)','Divergence']


plt.figure(figsize=(16,14),dpi=200)

ax1 = plt.subplot(4,4,1)
ax1.plot(t, UN[0,:], linestyle="-", lw = 1.5, color='k',label="Disp-$r$") 
ax1.plot(t, UN[1,:], linestyle="-.",lw = 1.5, color='k',label="Disp-$\\theta$") 
ax1.plot(t, UN[2,:], linestyle=":", lw = 1.5, color='k',label="Disp-$\\phi$")
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.spines['bottom'].set_color('none')
ax1.set_ylabel(ylabels[0])
ax1.set_ylim(-5.e-2,5.e-2)
ax1.xaxis.set_ticks([])
ax1.yaxis.major.formatter.set_powerlimits((-1,2))
ax1.set_title(titles[0],fontsize=16)
ax1.legend(loc=2,prop={"size":9})

ax2 = plt.subplot(4,4,2)
ax2.plot(t, UI[0,:], linestyle="-", lw = 1.5, color='k',label="Disp-$r$") 
ax2.plot(t, UI[1,:], linestyle="-.",lw = 1.5, color='k',label="Disp-$\\theta$") 
ax2.plot(t, UI[2,:], linestyle=":", lw = 1.5, color='k',label="Disp-$\\phi$")
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.spines['bottom'].set_color('none')
ax2.spines['left'].set_color('none')
ax2.set_ylim(-5.e-2,5.e-2)
ax2.xaxis.set_ticks([])
ax2.yaxis.set_ticks([])
ax2.set_title(titles[1],fontsize=16)
ax2.legend(loc=2,prop={"size":9})

ax3 = plt.subplot(4,4,3)
ax3.plot(t, UF[0,:], linestyle="-", lw = 1.5, color='k',label="Disp-$r$") 
ax3.plot(t, UF[1,:], linestyle="-.",lw = 1.5, color='k',label="Disp-$\\theta$") 
ax3.plot(t, UF[2,:], linestyle=":", lw = 1.5, color='k',label="Disp-$\\phi$")
ax3.spines['right'].set_color('none')
ax3.spines['top'].set_color('none')
ax3.spines['bottom'].set_color('none')
ax3.spines['left'].set_color('none')
ax3.set_ylim(-5.e-2,5.e-2)
ax3.xaxis.set_ticks([])
ax3.yaxis.set_ticks([])
ax3.set_title(titles[2],fontsize=16)
ax3.legend(loc=2,prop={"size":9})

ax4 = plt.subplot(4,4,4)
ax4.plot(t, U[0,:], linestyle="-", lw = 1.5, color='k',label="Disp-$r$") 
ax4.plot(t, U[1,:], linestyle="-.",lw = 1.5, color='k',label="Disp-$\\theta$") 
ax4.plot(t, U[2,:], linestyle=":", lw = 1.5, color='k',label="Disp-$\\phi$")
ax4.spines['right'].set_color('none')
ax4.spines['top'].set_color('none')
ax4.spines['bottom'].set_color('none')
ax4.spines['left'].set_color('none')
ax4.set_ylim(-5.e-2,5.e-2)
ax4.xaxis.set_ticks([])
ax4.yaxis.set_ticks([])
ax4.set_title(titles[3],fontsize=16)
ax4.legend(loc=2,prop={"size":9})


ax5 = plt.subplot(4,4,5)
ax5.plot(t, VN[0,:], linestyle="-", lw = 1.5, color='k',label="Vel-$r$") 
ax5.plot(t, VN[1,:], linestyle="-.",lw = 1.5, color='k',label="Vel-$\\theta$") 
ax5.plot(t, VN[2,:], linestyle=":", lw = 1.5, color='k',label="Vel-$\\phi$")
ax5.spines['right'].set_color('none')
ax5.spines['top'].set_color('none')
ax5.spines['bottom'].set_color('none')
ax5.set_ylabel(ylabels[1])
ax5.set_ylim(-1e0,1e0)
ax5.xaxis.set_ticks([])
ax5.yaxis.major.formatter.set_powerlimits((-1,2))
ax5.legend(loc=2,prop={"size":9})

ax6 = plt.subplot(4,4,6)
ax6.plot(t, VI[0,:], linestyle="-", lw = 1.5, color='k',label="Vel-$r$") 
ax6.plot(t, VI[1,:], linestyle="-.",lw = 1.5, color='k',label="Vel-$\\theta$") 
ax6.plot(t, VI[2,:], linestyle=":", lw = 1.5, color='k',label="Vel-$\\phi$")
ax6.spines['right'].set_color('none')
ax6.spines['top'].set_color('none')
ax6.spines['bottom'].set_color('none')
ax6.spines['left'].set_color('none')
ax6.set_ylim(-1e0,1e0)
ax6.xaxis.set_ticks([])
ax6.yaxis.set_ticks([])
ax6.legend(loc=2,prop={"size":9})

ax7 = plt.subplot(4,4,7)
ax7.plot(t, VF[0,:], linestyle="-", lw = 1.5, color='k',label="Vel-$r$") 
ax7.plot(t, VF[1,:], linestyle="-.",lw = 1.5, color='k',label="Vel-$\\theta$") 
ax7.plot(t, VF[2,:], linestyle=":", lw = 1.5, color='k',label="Vel-$\\phi$")
ax7.spines['right'].set_color('none')
ax7.spines['top'].set_color('none')
ax7.spines['bottom'].set_color('none')
ax7.spines['left'].set_color('none')
ax7.set_ylim(-1e0,1e0)
ax7.xaxis.set_ticks([])
ax7.yaxis.set_ticks([])
ax7.legend(loc=2,prop={"size":9})

ax8 = plt.subplot(4,4,8)
ax8.plot(t, V[0,:], linestyle="-", lw = 1.5, color='k',label="Vel-$r$") 
ax8.plot(t, V[1,:], linestyle="-.",lw = 1.5, color='k',label="Vel-$\\theta$") 
ax8.plot(t, V[2,:], linestyle=":", lw = 1.5, color='k',label="Vel-$\\phi$")
ax8.spines['right'].set_color('none')
ax8.spines['top'].set_color('none')
ax8.spines['bottom'].set_color('none')
ax8.spines['left'].set_color('none')
ax8.set_ylim(-1e0,1e0)
ax8.xaxis.set_ticks([])
ax8.yaxis.set_ticks([])
ax8.legend(loc=2,prop={"size":9})

ax9 = plt.subplot(4,4,9)
ax9.plot(t, RN[0,:], linestyle="-", lw = 1.5, color='k',label="Rot-$r$") 
ax9.plot(t, RN[1,:], linestyle="-.",lw = 1.5, color='k',label="Rot-$\\theta$") 
ax9.plot(t, RN[2,:], linestyle=":", lw = 1.5, color='k',label="Rot-$\\phi$")
ax9.spines['right'].set_color('none')
ax9.spines['top'].set_color('none')
ax9.spines['bottom'].set_color('none')
ax9.set_ylabel(ylabels[2])
ax9.set_ylim(-2e-4,2e-4)
ax9.xaxis.set_ticks([])
ax9.yaxis.major.formatter.set_powerlimits((-1,2))
ax9.legend(loc=2,prop={"size":9})

ax10 = plt.subplot(4,4,10)
ax10.plot(t, RI[0,:], linestyle="-", lw = 1.5, color='k',label="Rot-$r$") 
ax10.plot(t, RI[1,:], linestyle="-.",lw = 1.5, color='k',label="Rot-$\\theta$") 
ax10.plot(t, RI[2,:], linestyle=":", lw = 1.5, color='k',label="Rot-$\\phi$")
ax10.spines['right'].set_color('none')
ax10.spines['top'].set_color('none')
ax10.spines['left'].set_color('none')
ax10.spines['bottom'].set_color('none')
ax10.set_ylim(-2e-4,2e-4)
ax10.xaxis.set_ticks([])
ax10.yaxis.set_ticks([])
ax10.legend(loc=2,prop={"size":9})

ax11 = plt.subplot(4,4,11)
ax11.plot(t, RF[0,:], linestyle="-", lw = 1.5, color='k',label="Rot-$r$") 
ax11.plot(t, RF[1,:], linestyle="-.",lw = 1.5, color='k',label="Rot-$\\theta$") 
ax11.plot(t, RF[2,:], linestyle=":", lw = 1.5, color='k',label="Rot-$\\phi$")
ax11.spines['right'].set_color('none')
ax11.spines['top'].set_color('none')
ax11.spines['left'].set_color('none')
ax11.spines['bottom'].set_color('none')
ax11.set_ylim(-2e-4,2e-4)
ax11.xaxis.set_ticks([])
ax11.yaxis.set_ticks([])
ax11.legend(loc=2,prop={"size":9})

ax12 = plt.subplot(4,4,12)
ax12.plot(t, R[0,:], linestyle="-", lw = 1.5, color='k',label="Rot-$r$") 
ax12.plot(t, R[1,:], linestyle="-.",lw = 1.5, color='k',label="Rot-$\\theta$") 
ax12.plot(t, R[2,:], linestyle=":", lw = 1.5, color='k',label="Rot-$\\phi$")
ax12.spines['right'].set_color('none')
ax12.spines['top'].set_color('none')
ax12.spines['left'].set_color('none')
ax12.spines['bottom'].set_color('none')
ax12.set_ylim(-2e-4,2e-4)
ax12.xaxis.set_ticks([])
ax12.yaxis.set_ticks([])
ax12.legend(loc=2,prop={"size":9})

ax13 = plt.subplot(4,4,13)
ax13.plot(t, SN[0,:], linestyle="-", lw = 1.5, color='k',label="Div-$r$") 
ax13.plot(t, SN[1,:], linestyle="-.",lw = 1.5, color='k',label="Div-$\\theta$") 
ax13.plot(t, SN[2,:], linestyle=":", lw = 1.5, color='k',label="Div-$\\phi$")
ax13.spines['right'].set_color('none')
ax13.spines['top'].set_color('none')
ax13.set_ylabel(ylabels[3])
ax13.set_xlabel('Time (s)')
ax13.set_ylim(-2e-4,2e-4)
ax13.yaxis.major.formatter.set_powerlimits((-1,2))
ax13.legend(loc=2,prop={"size":9})

ax14 = plt.subplot(4,4,14)
ax14.plot(t, SI[0,:], linestyle="-", lw = 1.5, color='k',label="Div-$r$") 
ax14.plot(t, SI[1,:], linestyle="-.",lw = 1.5, color='k',label="Div-$\\theta$") 
ax14.plot(t, SI[2,:], linestyle=":", lw = 1.5, color='k',label="Div-$\\phi$")
ax14.spines['right'].set_color('none')
ax14.spines['top'].set_color('none')
ax14.spines['left'].set_color('none')
ax14.set_ylim(-2e-4,2e-4)
ax14.set_xlabel('Time (s)')
ax14.yaxis.set_ticks([])
ax14.legend(loc=2,prop={"size":9})

ax15 = plt.subplot(4,4,15)
ax15.plot(t, SF[0,:], linestyle="-", lw = 1.5, color='k',label="Div-$r$") 
ax15.plot(t, SF[1,:], linestyle="-.",lw = 1.5, color='k',label="Div-$\\theta$") 
ax15.plot(t, SF[2,:], linestyle=":", lw = 1.5, color='k',label="Div-$\\phi$")
ax15.spines['right'].set_color('none')
ax15.spines['top'].set_color('none')
ax15.spines['left'].set_color('none')
ax15.set_ylim(-2e-4,2e-4)
ax15.set_xlabel('Time (s)')
ax15.yaxis.set_ticks([])
ax15.legend(loc=2,prop={"size":9})

ax16 = plt.subplot(4,4,16)
ax16.plot(t, S[0,:], linestyle="-", lw = 1.5, color='k',label="Div-$r$") 
ax16.plot(t, S[1,:], linestyle="-.",lw = 1.5, color='k',label="Div-$\\theta$") 
ax16.plot(t, S[2,:], linestyle=":", lw = 1.5, color='k',label="Div-$\\phi$")
ax16.spines['right'].set_color('none')
ax16.spines['top'].set_color('none')
ax16.spines['left'].set_color('none')
ax16.set_ylim(-2e-4,2e-4)
ax16.set_xlabel('Time (s)')
ax16.yaxis.set_ticks([])
ax16.legend(loc=2,prop={"size":9})


plt.suptitle("Seismic wavefield of a Double-Couple Point Source", fontsize=16)
plt.ion() # set interective mode


# plt.savefig('DC.png')

plt.show()

