def elastic_RK4_2D(DF_l, F_l, Mat_l, X_l, Y_l, t, nf, nx, ny, dx, dy, dt, order, r_l, source_parameter, DF_r, F_r, Mat_r, X_r, Y_r, r_r,friction_parameters, slip, psi, DF_slip, DF_psi, fric_law, FaultOutput0, Y0, mode):

    # fourth order Runge-Kutta time-stepping
    import rate2d
    import numpy as np

    # intialize arrays for Runge-Kutta stages
    k1_l = np.zeros((nx, ny, nf))
    k2_l = np.zeros((nx, ny, nf))
    k3_l = np.zeros((nx, ny, nf))
    k4_l = np.zeros((nx, ny, nf))
    
    k1_r = np.zeros((nx, ny, nf))
    k2_r = np.zeros((nx, ny, nf))
    k3_r = np.zeros((nx, ny, nf))
    k4_r = np.zeros((nx, ny, nf))
    
    k1slip = np.zeros((ny, 1))
    k1psi  = np.zeros((ny, 1))
    k2slip = np.zeros((ny, 1))
    k2psi  = np.zeros((ny, 1))
    k3slip = np.zeros((ny, 1))
    k3psi  = np.zeros((ny, 1))
    k4slip = np.zeros((ny, 1))
    k4psi  = np.zeros((ny, 1))

    rate2d.elastic_rate2d(k1_l, F_l, Mat_l, X_l, Y_l, t, nf, nx, ny, dx, dy, order, r_l, source_parameter, k1_r, F_r, Mat_r, X_r, Y_r, r_r, friction_parameters, slip, psi, k1slip, k1psi, fric_law, FaultOutput0, 0, Y0, mode)
    
    rate2d.elastic_rate2d(k2_l, F_l+0.5*dt*k1_l, Mat_l, X_l, Y_l, t+0.5*dt, nf, nx, ny, dx, dy, order, r_l, source_parameter,k2_r, F_r+0.5*dt*k1_r, Mat_r, X_r, Y_r, r_r, friction_parameters, slip+0.5*dt*k1slip, psi+0.5*dt*k1psi, k2slip, k2psi, fric_law, FaultOutput0, 1, Y0, mode)
    
    rate2d.elastic_rate2d(k3_l, F_l+0.5*dt*k2_l, Mat_l, X_l, Y_l, t+0.5*dt, nf, nx, ny, dx, dy, order, r_l, source_parameter,k3_r, F_r+0.5*dt*k2_r, Mat_r, X_r, Y_r, r_r, friction_parameters, slip+0.5*dt*k2slip, psi+0.5*dt*k2psi, k3slip, k3psi,fric_law, FaultOutput0, 2, Y0, mode)
    
    rate2d.elastic_rate2d(k4_l, F_l+dt*k3_l, Mat_l, X_l, Y_l, t+dt, nf, nx, ny, dx, dy, order, r_l, source_parameter,k4_r, F_r+dt*k3_r, Mat_r, X_r, Y_r, r_r, friction_parameters, slip+dt*k3slip, psi+dt*k3psi, k4slip, k4psi, fric_law, FaultOutput0, 3, Y0, mode)

    
    # update fields
    DF_l[:,:,:] = F_l[:,:,:]  + (dt/6.0)*(k1_l[:,:,:]  + 2.0*k2_l[:,:,:]  + 2.0*k3_l[:,:,:]  + k4_l[:,:,:] )
    DF_r[:,:,:] = F_r[:,:,:]  + (dt/6.0)*(k1_r[:,:,:]  + 2.0*k2_r[:,:,:]  + 2.0*k3_r[:,:,:]  + k4_r[:,:,:] )
    
    DF_slip[:,:] = slip[:,:]  + (dt/6.0)*(k1slip[:,:]  + 2.0*k2slip[:,:]  + 2.0*k3slip[:,:]  + k4slip[:,:])
    DF_psi[:,:]  = psi[:,:]   + (dt/6.0)*(k1psi[:,:]   + 2.0*k2psi[:,:]   + 2.0*k3psi[:,:]   + k4psi[:,:])
    
    FaultOutput0[:, 4] = DF_slip[:,0]
    FaultOutput0[:, 5] = DF_psi[:,0]