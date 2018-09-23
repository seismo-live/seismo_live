# This is a python finite difference code to simulate waves in 2D linear isotropic medium
# SBP finite difference operators of interior accuracy 2, 4, 6 are used to approximate spatial derivatives
# Boundary conditions are imposed weakly using penalties (SATs)
# Time discretization is performed using the classical Runge-Kutta Method
# matplotlib notebook
# Created by: Kenneth Duru
#             LMU Munich, June, 2016.
#
#
import numpy as np
import matplotlib.pyplot as plt
import RK4_2D

plt.switch_backend("TkAgg")
plt.ion()


# ---------------------------------------------------------
# Summation-by-parts finite difference solver
#         for
# Elastic wave equation
#         in
# 2D Cartesian grid
# ---------------------------------------------------------

Lx = 50.0     # length of the domain (x-axis)
Ly = 50.0     # width of the domain (y-axis)
nx = 75       # grid points in x
ny = 75       # grid points in y
nt = 150      # number of time steps
dx = Lx/nx    # spatial step in x
dy = Ly/ny    # spatial step in y
isx = nx / 2  # source index x
isy = ny / 2  # source index y
ist = 100     # shifting of source time function
f0 = 100.0    # dominant frequency of source (Hz)
isnap = 20    # snapshot frequency
T = 1.0 / f0  # dominant period
nop = 5       # length of operator
nf = 3        # number of fields
cp = 6.0      # compresional wave speed
rho = 2.6702  # density

# extract Lame parameters
lam = rho*cp**2

dt = 0.5/cp*dx    # Time step


order = 6   # spatial order of accuracy

# Model type, available are "homogeneous", "fault_zone",
# "surface_low_velocity_zone", "random", "topography",
# "slab"
model_type = "homogeneous"

# Receiver locations
irx = np.array([30, 40, 50, 60, 70])
irz = np.array([5, 5, 5, 5, 5])
seis = np.zeros((len(irx), nt))

# Boundary reflection coefficients
r = np.array([1,0,0,0])

# Initialize pressure at different time steps and the second
# derivatives in each direction
F = np.zeros((nx, ny, nf))
Fnew = np.zeros((nx, ny, nf))

X = np.zeros((nx, ny))
Y = np.zeros((nx, ny))
              
p = np.zeros((nx, ny))             
# Initialize velocity model
Mat = np.zeros((nx, ny, 2))

if model_type == "homogeneous":
    Mat[:,:,0] += rho
    Mat[:,:,1] += lam

elif model_type == "random":
    pert = 0.4
    r_rho = 2.0 * (np.random.rand(nx, ny) - 0.5) * pert
    r_lambda = 2.0 * (np.random.rand(nx, ny) - 0.5) * pert
    
    Mat[:,:,0] += rho*(1.0 + r_rho)
    Mat[:,:,1] += lam*(1.0 + r_lambda)
    
    
    
sigma = 3.0                      # standard deviation of the initial Gaussian perturbation

for i in range(0, nx):
    for j in range(0, ny):
        F[i, j,0] = np.exp(-np.log(2)*(((i-isx)*dx)**2/(sigma) + ((j-isy)*dy)**2/(sigma)))
        X[i, j] = i*dx
        Y[i,j] = j*dy

v = 0.15
p = np.sqrt(F[:,:,0]**2)
#image = plt.imshow(p, interpolation='nearest', animated=True,
#                   vmin=-v, vmax=+v, cmap=plt.cm.RdBu)
image = plt.imshow(p, aspect='auto',extent=[0,Lx,Ly,0]
          ,cmap = 'viridis', vmin=0, vmax=+v, animated=True, interpolation='none')
# Plot the receivers
for x, z in zip(irx, irz):
    plt.text(x, z, '+')

plt.text(isx, isy, 'o')
plt.colorbar()
plt.xlabel('ix')
plt.ylabel('iy')


plt.show()


# required for seismograms
ir = np.arange(len(irx))

#################################################

# Time-stepping 
for it in range(nt):
    
    #4th order Runge-Kutta 
    RK4_2D.acoustics_RK4_2D(Fnew, F, Mat, nf, nx, ny, dx, dy, dt, order, r)
    
    # update fields
    F = Fnew
    
    #extract particle velocity for visualization
    p = F[:,:,0]

    # update time
    t = it*dt
    
    # Plot every isnap-th iteration
    if it % isnap == 0:                    # you can change the speed of the plot by increasing the plotting interval
        plt.title("time: %.2f" % t)
        image.set_data(p)
        plt.gcf().canvas.draw()
        plt.show()
        print(it)
    
    # Save seismograms
    seis[ir, it] = p[irz[ir], irx[ir]]
    
plt.ioff()
plt.figure(figsize=(12, 12))

plt.subplot(211)
ymax = seis.ravel().max()
time = np.arange(nt) * dt
for ir in range(len(seis)):
    plt.plot(time, seis[ir, :] + ymax * ir)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

plt.subplot(212)
ymax = seis.ravel().max()
for ir in range(len(seis)):
    plt.plot(time, seis[ir, :] + ymax * ir)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

plt.show()
