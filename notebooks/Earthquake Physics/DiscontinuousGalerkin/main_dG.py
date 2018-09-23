# Parameters initialization and plotting the simulation
import Lagrange
import numpy as np
import timeintegrate
import quadraturerules
import specdiff
import utils
import matplotlib.pyplot as plt

plt.switch_backend("TkAgg")
plt.ion()

# Show the plots in the Notebook.
# plt.switch_backend("nbagg")

iplot  = 10  

# at the moment, keep the length of the two domain the same
# physical domain x = [ax, bx] (km)
# for the first domain 
ax_1 = 0.0
bx_1 = 15.0

# physical domain x = [ax, bx] (km)
#for the second domain
ax_2 = bx_1
bx_2 = 30.0

# choose quadrature rules and the corresponding nodes
# we use Gauss-Legendre-Lobatto (Lobatto) or  Gauss-Legendre (Legendre) quadrature rule.

node = 'Lobatto'
#node = 'Legendre'

if node not in ('Lobatto', 'Legendre'):
     print('quadrature rule not implemented. choose node = Legendre or node = Lobatto')
     exit(-1)

# polynomial degree N: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
N = 3                  # Lagrange polynomial degree
NP = N+1               # quadrature nodes per element

if N < 1 or N > 12:
    print('polynomial degree not implemented. choose N>= 1 and N <= 12')
    exit(-1)


# degrees of freedom to resolve the wavefield    
deg_of_freedom1 = 300
deg_of_freedom2 = 300

# estimate the number of elements needed for a given polynomial degree and degrees of freedom
num_element1 = round(deg_of_freedom1/NP)
num_element2 = round(deg_of_freedom2/NP)

# Initialize the mesh
y_1 = np.zeros(NP*num_element1)
y_2 = np.zeros(NP*num_element2)


# Generate num_element dG elements in the interval [ax, bx]
x0_1 = np.linspace(ax_1,bx_1,num_element1+1)
dx_1 = np.diff(x0_1)                     # element sizes
    
x0_2 = np.linspace(ax_2,bx_2,num_element2+1)
dx_2 = np.diff(x0_2)                     # element sizes

# Generate Gauss quadrature nodes (psi): [-1, 1] and weights (w)
if node == 'Legendre':
    GL_return = quadraturerules.GL(N)
    psi = GL_return['xi']
    w = GL_return['weights'];
    
if node == 'Lobatto':
    gll_return = quadraturerules.gll(N)
    psi = gll_return['xi']
    w = gll_return['weights']

# Use the Gauss quadrature nodes (psi) generate the mesh (y)
for i in range (1,num_element1+1):
        for j in range (1,(N+2)):
            y_1[j+(N+1)*(i-1)-1] = dx_1[i-1]/2.0 * (psi[j-1] + 1.0) +x0_1[i-1]

for i in range (1,num_element2+1):
        for j in range (1,(N+2)):
            y_2[j+(N+1)*(i-1)-1] = dx_2[i-1]/2.0 * (psi[j-1] + 1.0) +x0_2[i-1]
          
deg_of_freedom1 = len(y_1) #same for both the domains
deg_of_freedom2 = len(y_2) #same for both the domains

# generate the spectral difference operator (D) in the reference element: [-1, 1]  
D = specdiff.derivative_GL(N, psi, w)


# Boundary condition reflection coefficients 
r0 = 0.0        # r=0:absorbing, r=1:free-surface, r=-1: clamped, 
rn = 0.0        # r=0:absorbing, r=1:free-surface, r=-1: clamped, 

# Initialize the wave-fields
L_1 = 0.5*(bx_1-ax_1)
L_2 = 0.5*(bx_2-ax_2)

delta_1 = 0.01*(bx_1-ax_1)
delta_2 = 0.01*(bx_2-ax_2)

x0_1 = 0.5*(bx_1+ax_1)
x0_2 = 0.35*(bx_2+ax_2)

#u_1 = np.sin(2*np.pi*y_1/L_1)           # Sine function (multiply by Zero to see the fault behaviour only
#u_2 = np.sin(2*np.pi*y_2/L_2)           # i.e. without wave propagration)
u_1 = 0.*np.exp(-(y_1-x0_1)**2/delta_1)         # Gaussian
u_2 =  0.0/np.sqrt(2.0*np.pi*delta_2**2)*np.exp(-(y_2-x0_2)**2/(2.0*delta_2**2))         # Gaussian

u_1 = np.transpose(u_1)
v_1 = np.zeros(len(u_1))
#U_1 = np.zeros(len(u_1))
#V_1 = np.zeros(len(u_1))
    
u_2 = np.transpose(u_2)
v_2 = np.zeros(len(u_2))
#U_2 = np.zeros(len(u_2))

# Values of friction coefficient (will be asked)

        
#f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row',figsize=(10,6))
#line1 = ax1.plot(y_1, u_1,'r')#,y_1,U_1,'k--')
#line2 = ax2.plot(y_2, u_2,'r')#,y_2,U_2,'k--')
#line3 = ax3.plot(y_1, v_1,'r')#,y_1,V_1,'k--')
#line4 = ax4.plot(y_2, v_2,'r')#,y_2,V_2,'k--')
    
#ax1.set_ylabel('velocity [m/s]')
#ax3.set_ylabel('stress [MPa]')
#ax3.set_xlabel('x [km]')
#ax4.set_xlabel('x [km]')

#f.subplots_adjust(wspace=0)

#plt.ion()
#plt.show()


# Initialize animated plot                                                                                                                                                                      
fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(2,1,1)
line1 = ax1.plot(y_1, u_1, 'r', y_1, u_1, 'k--')
plt.xlabel('x [km]')
plt.ylabel('velocity [m/s]')

ax2 = fig.add_subplot(2,1,2)
line2 = ax2.plot(y_1, v_1, 'r', y_1, v_1, 'k--')
plt.xlabel('x[km]')
plt.ylabel('stress[MPa]')
plt.ion()
plt.show()
    
#ax1.set_xlim([ax_1, bx_1])
#ax2.set_xlim([ax_2,bx_2])

ax1.set_xlim([-2.5,2.5])
ax2.set_xlim([-2.5,2.5])

# Material parameters
c1s = 3.464 + 0.0*np.sin(2.0*np.pi*y_1)             # shear wave speed (km/s)
rho1 = 2.67 + 0.0*y_1             # density          (g/cm^3)
mu1 = rho1 * c1s**2              # shear modulus    (GPa)
Z1s = rho1*c1s                   # shear impedance

c2s = 3.464 + 0.0*y_2             # shear wave speed (km/s)
rho2 = 2.67 + 0.0*y_2             # density          (g/cm^3)
mu2 = rho2 * c2s**2             # shear modulus    (GPa)
Z2s = rho2*c2s                  # shear impedance

   

# choose friction law: fric_law
# we use linear (LN: T = alpha*v, alpha >=0)
# Slip-weakening (SW)
# Rate-and-state friction law (RS)

fric_law = 'RS'

if fric_law in ('SW'):
    u_1 = 1.*np.exp(-(y_1-x0_1)**2/delta_1)         # Gaussian
    u_2 =  1.0*np.exp(-(y_2-x0_2)**2/(2.0*delta_2**2))         # Gaussian

    u_1 = np.transpose(u_1)
    v_1 = np.zeros(len(u_1))
    #U_1 = np.zeros(len(u_1))
    #V_1 = np.zeros(len(u_1))
    
u_2 = np.transpose(u_2)
v_2 = np.zeros(len(u_2))

if fric_law not in ('LN', 'SW', 'RS'):
     print('friction law not implemented. choose fric_law = SW or fric_law = LN')
     exit(-1)

if fric_law  in ('LN', 'SW'):
     alpha = 1e1000000                           # initial friction coefficient
     S = 0.0                                # initial slip (in m)
     Tau_0 = 81.24+0.1*0.36                 # initial load (81.24 in MPa), slight increase will unlock the fault
     W = 0.0

if fric_law  in ('RS'):
     alpha = 1e1000000                           # initial friction coefficient                                                                                                              
     S = 0.0                                # initial slip (in m)                                                                                                                             
     Tau_0 = 100.0                       # initial load (81.24 in MPa), slight increase will unlock the fault   
     W = 0.4367

Vd = [0]
Sd = [0]


# Time stepping parameters
cfl = 0.5                              # CFL number
dt = (cfl/(max(max(c1s),max(c2s))*(2*N+1)))*min(min(dx_1), min(dx_2))         # time-step (s)
t = 0.0                                # initial time
Tend = 2.5                             # final time (s)
n = 0           

T = [t]

    
for t in utils.drange (dt,Tend+dt,dt):
    n = n+1
    ADER_Wave_1D_GL_return = timeintegrate.ADER_Wave_1D_GL(u_1,v_1,u_2,v_2,S,W,D,NP,num_element1,num_element2,dx_1,dx_2,w,psi,t,r0,rn,dt,rho1,mu1,rho2,mu2,alpha,Tau_0,fric_law)
    u_1 = ADER_Wave_1D_GL_return['Hu_1']
    v_1 = ADER_Wave_1D_GL_return['Hv_1']
    u_2 = ADER_Wave_1D_GL_return['Hu_2']
    v_2 = ADER_Wave_1D_GL_return['Hv_2']
    S   = ADER_Wave_1D_GL_return['H_d']
    W   = ADER_Wave_1D_GL_return['H_psi']
    
    Vd.append(np.abs(u_1[-1]-u_2[0]))
    Sd.append(v_2[0]+0.0*Tau_0)
    T.append(t)

    print(np.abs(u_1[-1]-u_2[0]))
    # Analytical (Analytical for elastic sine waves i.e. without friction law. Hence, it is not needed here)
    #U_1 = 0.5*(np.sin(2*np.pi/L_1*(y_1+cs*(t+1*dt))) + np.sin(2*np.pi/L_1*(y_1-cs*(t+1*dt))))
    #U_2 = 0.5*(np.sin(2*np.pi/L_2*(y_2+cs*(t+1*dt))) + np.sin(2*np.pi/L_2*(y_2-cs*(t+1*dt))))
    #U_1 = 0.5*(np.exp(-(y_1+cs*(t+1*dt)-0.5)**2/0.01) + np.exp(-(y_1-cs*(t+1*dt)-0.5)**2/0.01)) #analytical gaussian
    #V_1 = 0.5*Zs*(np.sin(2*np.pi/L_1*(y_1+cs*(t+1*dt))) - np.sin(2*np.pi/L_1*(y_1-cs*(t+1*dt))))
    #V_2 = 0.5*Zs*(np.sin(2*np.pi/L_2*(y_2+cs*(t+1*dt))) - np.sin(2*np.pi/L_2*(y_2-cs*(t+1*dt))))
    
     # plotting                                                                                                                                                                                  
    if n % iplot == 0:
        for l in line1:
            l.remove()
            del l
        for l in line2:
            l.remove()
            del l

        # Display lines                                                                                                                                                                         
        line1 = ax1.plot(T, Vd, 'k-*')
        line2 = ax2.plot(T, Sd, 'k-*')
        plt.legend(iter(line2), ('Numerical'))
        plt.gcf().canvas.draw()

plt.ioff()
plt.show()


#    if (n-1) % iplot == 0:
#
#        
#        for l in line1:
#            l.remove()
#            del l               
#            for l in line2:
#                l.remove()
#                del l 
#            for l in line3:
#                l.remove()
#                del l               
#            for l in line4:
#                l.remove()
#                del l
#        # Display lines
#        line1 = ax1.plot(y_1, u_1, 'r')#,y_1,U_1,'k--')
#        line2 = ax2.plot(y_2, u_2, 'r')#,y_2,U_2,'k--')
#        line3 = ax3.plot(y_1, 0*Tau_0+v_1, 'r')#,y_1,V_1,'k--')
#        line4 = ax4.plot(y_2, 0*Tau_0+v_2, 'r')#,y_2,V_2,'k--')
#    plt.legend(iter(line2), ('Numerical', 'Analytic'))
#    plt.gcf().canvas.draw()
       
#plt.ioff()
#plt.show() 
