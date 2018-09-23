def elastic_RK4(rv_l, rs_l, v_l, s_l, rho_l, mu_l, nx_l, dx_l, order_l, y_l, t_l, dt_l, r0_l, r1_l, tau0_1_l, tau0_2_l, tauN_1_l, tauN_2_l, rv_r, rs_r, v_r, s_r, rho_r, mu_r, nx_r, dx_r, order_r, y_r, t_r, dt_r, r0_r, r1_r, tau0_1_r, tau0_2_r, tauN_1_r, tauN_2_r, rslip, rpsi, slip, psi,friction_parameters):
    
    nx_r =  nx_l
    dx_r = dx_l
    order_r = order_l  
    t_r = t_l
    dt_r = dt_l
    # fourth order Runge-Kutta time-stepping
    import rate
    import numpy as np

    # intialize arrays for Runge-Kutta stages
    k1v_l = np.zeros((nx_l, 1))
    k1s_l = np.zeros((nx_l, 1))
    k2v_l = np.zeros((nx_l, 1))
    k2s_l = np.zeros((nx_l, 1))
    k3v_l = np.zeros((nx_l, 1))
    k3s_l = np.zeros((nx_l, 1))
    k4v_l = np.zeros((nx_l, 1))
    k4s_l = np.zeros((nx_l, 1))
    
    k1v_r = np.zeros((nx_r, 1))
    k1s_r = np.zeros((nx_r, 1))
    k2v_r = np.zeros((nx_r, 1))
    k2s_r = np.zeros((nx_r, 1))
    k3v_r = np.zeros((nx_r, 1))
    k3s_r = np.zeros((nx_r, 1))
    k4v_r = np.zeros((nx_r, 1))
    k4s_r = np.zeros((nx_r, 1))
    
    k1slip = np.zeros((1, 1))
    k1psi  = np.zeros((1, 1))
    k2slip = np.zeros((1, 1))
    k2psi  = np.zeros((1, 1))
    k3slip = np.zeros((1, 1))
    k3psi  = np.zeros((1, 1))
    k4slip = np.zeros((1, 1))
    k4psi  = np.zeros((1, 1))
    
    rate.elastic_rate(k1v_l, k1s_l, v_l, s_l, rho_l, mu_l, nx_l, dx_l, order_l, t_l, y_l, r0_l, r1_l, tau0_1_l, tau0_2_l, tauN_1_l, tauN_2_l, k1v_r, k1s_r, v_r, s_r, rho_r, mu_r, nx_r, dx_r, order_r, t_r, y_r, r0_r, r1_r, tau0_1_r, tau0_2_r, tauN_1_r, tauN_2_r, slip, psi, k1slip, k1psi,friction_parameters)
    
    
    rate.elastic_rate(k2v_l, k2s_l, v_l+0.5*dt_l*k1v_l, s_l+0.5*dt_l*k1s_l, rho_l, mu_l, nx_l, dx_l, order_l, dt_l+0.5*dt_l, y_l, r0_l, r1_l, tau0_1_l, tau0_2_l, tauN_1_l, tauN_2_l,  k2v_r, k2s_r, v_r+0.5*dt_r*k1v_r, s_r+0.5*dt_r*k1s_r, rho_r, mu_r, nx_r, dx_r, order_r, dt_r+0.5*dt_r, y_r, r0_r, r1_r, tau0_1_r, tau0_2_r, tauN_1_r, tauN_2_r, 
                     slip+0.5*dt_l*k1slip, psi+0.5*dt_l*k1psi, k2slip, k2psi,friction_parameters)
    
    rate.elastic_rate(k3v_l, k3s_l, v_l+0.5*dt_l*k2v_l, s_l+0.5*dt_l*k2s_l, rho_l, mu_l, nx_l, dx_l, order_l, dt_l+0.5*dt_l, y_l, r0_l, r1_l, tau0_1_l, tau0_2_l, tauN_1_l, tauN_2_l, k3v_r, k3s_r, v_r+0.5*dt_r*k2v_r, s_r+0.5*dt_r*k2s_r, rho_r, mu_r, nx_r, dx_r, order_r, dt_r+0.5*dt_r, y_r, r0_r, r1_r, tau0_1_r, tau0_2_r, tauN_1_r, tauN_2_r, 
                     slip+0.5*dt_l*k2slip, psi+0.5*dt_l*k2psi, k3slip, k3psi,friction_parameters)
    
    rate.elastic_rate(k4v_l, k4s_l, v_l+dt_l*k3v_l, s_l+dt_l*k3s_l, rho_l, mu_l, nx_l, dx_l, order_l, dt_l+dt_l, y_l,  r0_l, r1_l, tau0_1_l, tau0_2_l, tauN_1_l, tauN_2_l,  k4v_r, k4s_r, v_r+dt_r*k3v_r, s_r+dt_r*k3s_r, rho_r, mu_r, nx_r, dx_r, order_r, dt_r+dt_r, y_r,  r0_r, r1_r, tau0_1_r, tau0_2_r, tauN_1_r, tauN_2_r,  
                     slip+ dt_l*k3slip, psi+dt_l*k3psi, k4slip, k4psi,friction_parameters)

    # update fields
    rv_l[:,:] = v_l + (dt_l/6.0)*(k1v_l + 2.0*k2v_l + 2.0*k3v_l + k4v_l)
    rs_l[:,:] = s_l + (dt_l/6.0)*(k1s_l + 2.0*k2s_l + 2.0*k3s_l + k4s_l)
    # update fields
    rv_r[:,:] = v_r + (dt_r/6.0)*(k1v_r + 2.0*k2v_r + 2.0*k3v_r + k4v_r)
    rs_r[:,:] = s_r + (dt_r/6.0)*(k1s_r + 2.0*k2s_r + 2.0*k3s_r + k4s_r)
    
    # update slip and state
    rslip[:,:] = slip + (dt_r/6.0)*(k1slip + 2.0*k2slip + 2.0*k3slip + k4slip)
    rpsi[:,:] = psi + (dt_r/6.0)*(k1psi + 2.0*k2psi + 2.0*k3psi + k4psi)


