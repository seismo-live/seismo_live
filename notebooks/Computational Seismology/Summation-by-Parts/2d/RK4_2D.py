def elastic_RK4_2D(DF, F, Mat, X, Y, t, nf, nx, ny, dx, dy, dt, order, r, source_parameter):

    # fourth order Runge-Kutta time-stepping
    import rate2d
    import numpy as np

    # intialize arrays for Runge-Kutta stages
    k1 = np.zeros((nx, ny, nf))
    k2 = np.zeros((nx, ny, nf))
    k3 = np.zeros((nx, ny, nf))
    k4 = np.zeros((nx, ny, nf))

    rate2d.elastic_rate2d(k1, F, Mat, X, Y, t, nf, nx, ny, dx, dy, order, r, source_parameter)
    rate2d.elastic_rate2d(k2, F+0.5*dt*k1, Mat, X, Y, t+0.5*dt, nf, nx, ny, dx, dy, order, r, source_parameter)
    rate2d.elastic_rate2d(k3, F+0.5*dt*k2, Mat, X, Y, t+0.5*dt, nf, nx, ny, dx, dy, order, r, source_parameter)
    rate2d.elastic_rate2d(k4, F+dt*k3, Mat, X, Y, t+dt, nf, nx, ny, dx, dy, order, r, source_parameter)
    
    # update fields
    DF[:,:,:] = F[:,:,:]  + (dt/6.0)*(k1[:,:,:]  + 2.0*k2[:,:,:]  + 2.0*k3[:,:,:]  + k4[:,:,:] )
    

def acoustics_RK4_2D(DF, F, Mat, X, Y, t, nf, nx, ny, dx, dy, dt, order, r, source_parameter):

    # fourth order Runge-Kutta time-stepping                                                                                           
    import rate2d
    import numpy as np

    # intialize arrays for Runge-Kutta stages                                                                                          
    k1 = np.zeros((nx, ny, nf))
    k2 = np.zeros((nx, ny, nf))
    k3 = np.zeros((nx, ny, nf))
    k4 = np.zeros((nx, ny, nf))

    rate2d.acoustics_rate2d(k1, F, Mat, X, Y, t, nf, nx, ny, dx, dy, order, r, source_parameter)
    rate2d.acoustics_rate2d(k2, F+0.5*dt*k1, Mat, X, Y, t+0.5*dt, nf, nx, ny, dx, dy, order, r, source_parameter)
    rate2d.acoustics_rate2d(k3, F+0.5*dt*k2, Mat, X, Y, t+0.5*dt, nf, nx, ny, dx, dy, order, r, source_parameter)
    rate2d.acoustics_rate2d(k4, F+dt*k3, Mat, X, Y, t+dt, nf, nx, ny, dx, dy, order, r, source_parameter)

    # update fields                                                                                                                    
    DF[:,:,:] = F[:,:,:]  + (dt/6.0)*(k1[:,:,:]  + 2.0*k2[:,:,:]  + 2.0*k3[:,:,:]  + k4[:,:,:])
