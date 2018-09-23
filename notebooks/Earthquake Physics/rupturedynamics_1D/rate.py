def elastic_rate(hv_l, hs_l, v_l, s_l, rho_l, mu_l, nx_l, dx_l, order_l, t_l, y_l, r0_l, r1_l, tau0_1_l, tau0_2_l, tauN_1_l, tauN_2_l,hv_r, hs_r, v_r, s_r, rho_r, mu_r, nx_r, dx_r, order_r, t_r, y_r, r0_r, r1_r, tau0_1_r, tau0_2_r, tauN_1_r, tauN_2_r, slip, psi, dslip, dpsi,friction_parameters):
    
    nx_r =  nx_l
    dx_r = dx_l
    order_r = order_l  
    t_r = t_l
    
    # we compute rates that will be used for Runge-Kutta time-stepping
    #
    import first_derivative_sbp_operators
    import numpy as np
    import boundarycondition
    import interface
    
    V_l = np.zeros((nx_l, 1))
    S_l = np.zeros((nx_l, 1))
    Vt_l = np.zeros((nx_l, 1))
    St_l = np.zeros((nx_l, 1))
    Vx_l = np.zeros((nx_l, 1))
    Sx_l = np.zeros((nx_l, 1))

    # initialize arrays for computing derivatives
    vx_l = np.zeros((nx_l, 1))
    sx_l = np.zeros((nx_l, 1))

    # compute first derivatives for velocity and stress fields
    first_derivative_sbp_operators.dx(vx_l, v_l, nx_l, dx_l, order_l)
    first_derivative_sbp_operators.dx(sx_l, s_l, nx_l, dx_l, order_l)

    # compute the elastic rates
    hv_l[:,:] = (1.0/rho_l)*sx_l
    hs_l[:,:] = mu_l*vx_l

    # impose boundary conditions using penalty: SAT

    impose_bcm(hv_l, hs_l, v_l, s_l, rho_l, mu_l, nx_l, dx_l, order_l, r0_l, tau0_1_l, tau0_2_l)
    
    V_r = np.zeros((nx_r, 1))
    S_r = np.zeros((nx_r, 1))
    Vt_r = np.zeros((nx_r, 1))
    St_r = np.zeros((nx_r, 1))
    Vx_r = np.zeros((nx_r, 1))
    Sx_r = np.zeros((nx_r, 1))

    # initialize arrays for computing derivatives
    vx_r = np.zeros((nx_r, 1))
    sx_r = np.zeros((nx_r, 1))

    # compute first derivatives for velocity and stress fields
    first_derivative_sbp_operators.dx(vx_r, v_r, nx_r, dx_r, order_r)
    first_derivative_sbp_operators.dx(sx_r, s_r, nx_r, dx_r, order_r)

    # compute the elastic rates
    hv_r[:,:] = (1.0/rho_r)*sx_r 
    hs_r[:,:] = mu_r*vx_r 

    # impose boundary conditions using penalty: SAT
    impose_bcp(hv_r, hs_r, v_r, s_r, rho_r, mu_r, nx_r, dx_r, order_r, r1_r, tauN_1_r, tauN_2_r)

    x =0

    I = np.zeros((nx_r, 1)) + 1.0
    
    interface.couple_friction(hv_l, hs_l, hv_r, hs_r, v_l, s_l, v_r, s_r, slip, psi, order_r, nx_l, dx_l, dx_r, rho_l*I, mu_l*I, rho_r*I, mu_r*I, x, friction_parameters, dslip, dpsi)



def mms(V, S, V_t, S_t, V_x, S_x, y, t, type_0):
    
    import numpy as np
    
    
    if type_0 in ('Gaussian'):
        
        delta = 0.015*(y[-1,0]-y[0,0])
        
        cs = 3.464
        rho = 2.6702
        
        Zs = rho*cs
        
        x0 = 0.5*(y[-1, 0]-y[0, 0])
        
        V[:,:]=0*(1/np.sqrt(2.0*np.pi*delta**2)*0.5*(np.exp(-(y+cs*(t)-x0)**2/(2.0*delta**2))\
                                             + np.exp(-(y-cs*(t)-x0)**2/(2.0*delta**2))))
        
        S[:,:]=0*(1/np.sqrt(2.0*np.pi*delta**2)*0.5*Zs*(np.exp(-(y+cs*(t)-x0)**2/(2.0*delta**2))\
                                                - np.exp(-(y-cs*(t)-x0)**2/(2.0*delta**2))))
    
        V_t[:,:] = 0
        S_t[:,:] = 0
        V_x[:,:] = 0
        S_x[:,:] = 0
        
        
    if type_0 in ('Sinusoidal'):
        
        delta = (y[-1,0]-y[0,0])
        
        ny = 20.5/delta*np.pi  
        nt = 2.5*np.pi
        fs = 9.33
        
        V[:,:]= np.cos(nt*t)*np.sin(ny*y+fs)
        
        S[:,:]= ny*np.sin(nt*t)*np.cos(ny*y-fs)
    
        V_t[:,:] = -nt*np.sin(nt*t)*np.sin(ny*y+fs)
        S_t[:,:] = nt*ny*np.cos(nt*t)*np.cos(ny*y-fs)
        
        V_x[:,:] = ny*np.cos(nt*t)*np.cos(ny*y+fs)
        S_x[:,:] = -ny*ny*np.sin(nt*t)*np.sin(ny*y-fs)
    
    
    

    
    
def g(V, t):

    import numpy as np
    
    V[:,:] = 0.0
    
    if (t <= 2.0 and t >= 0.0):
        V[:,:] = (np.sin(np.pi*t))**4



def impose_bcm(hv, hs, v, s, rho, mu, nx, dx, order, r, tau_1, tau_2):

    # enforces boundry condition for left boundry
    
    # impose boundary conditions
    import numpy as np
    import boundarycondition

    h11 = np.zeros((1,1))
    penaltyweights(h11, order, dx)

    mv = np.zeros((1,1))
    ms = np.zeros((1,1))

    v0 = v[0,:]
    s0 = s[0,:]

    # compute SAT terms
    boundarycondition.bcm(mv, ms, v0, s0, rho, mu, r)

    # penalize boundaries with the SAT terms
    hv[0,:] =  hv[0,:] - tau_1/h11*mv
    hs[0,:] =  hs[0,:] - tau_2/h11*ms


    

    
def impose_bcp(hv, hs, v, s, rho, mu, nx, dx, order, r, tau_1, tau_2):
    
        # enforces boundry condition for right boundry

    
    # impose boundary conditions
    import numpy as np
    import boundarycondition

    # penalty weights
    h11 = np.zeros((1,1))
    penaltyweights(h11, order, dx)

    pv = np.zeros((1,1))
    ps = np.zeros((1,1))

    vn = v[nx-1,:]
    sn = s[nx-1,:]

    # compute SAT terms                                                                                                                               
    boundarycondition.bcp(pv, ps, vn, sn, rho, mu, r)

    # penalize boundaries with the SAT terms           

    
    hv[nx-1,:] =  hv[nx-1,:] - tau_1/h11*pv
    hs[nx-1,:] =  hs[nx-1,:] - tau_2/h11*ps


def penaltyweights(h11, order, dx):

    # penalty weights                                                                                                                                    
    if order==2:
        h11[:] = 0.5*dx
    if order== 4:
        h11[:] = (17.0/48.0)*dx

    if order==6:
        h11[:] = 13649.0/43200.0*dx
