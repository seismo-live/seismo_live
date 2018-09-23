def elastic_rate2d(D, F, Mat, X, Y, t, nf, nx, ny, dx, dy, order, r, source_parameter):
    # we compute rates that will be used for Runge-Kutta time-stepping
    #
    import first_derivative_sbp_operators
    import numpy as np
    import boundarycondition

    # initialize arrays for computing derivatives
    dxF = np.zeros((nf))
    dyF = np.zeros((nf))
    
    Ft = np.zeros((1))
    
    M = source_parameter[6]
    # compute the elastic rates
    for i in range(0, nx):
        for j in range(0, ny):
            
            x = X[i,j]
            y = Y[i,j]
            
            pointSource(Ft, dx, dy, i, j, x, y, t, source_parameter)
    
            # extract material parameters
            rho = Mat[i,j,0]
            lam = Mat[i,j,1]
            mu  = Mat[i,j,2]
            
            # compute spatial gradients of fields (particle velocity and stress)
            first_derivative_sbp_operators.dx2d(dxF, F, nx, i, j, dx, order)
            first_derivative_sbp_operators.dy2d(dyF, F, ny, i, j, dy, order)
            
            # momentum equation
            D[i,j,0] = 1.0/rho*(dxF[2] + dyF[4]) +  M[0]*Ft
            D[i,j,1] = 1.0/rho*(dxF[4] + dyF[3]) +  M[1]*Ft

            # Hooke's law
            D[i,j,2] = (2.0*mu + lam)*dxF[0] + lam*dyF[1] + M[2]*Ft
            D[i,j,3] = (2.0*mu + lam)*dyF[1] + lam*dxF[0] + M[3]*Ft
            D[i,j,4] = mu*(dyF[0] + dxF[1])               + M[4]*Ft
    

    # impose boundary conditions using penalty: SAT
    elastic_impose_bc(D, F, Mat, nx, ny, nf, dx, dy, order, r)


def acoustics_rate2d(D, F, Mat, X, Y, t, nf, nx, ny, dx, dy, order, r, source_parameter):
    # we compute rates that will be used for Runge-Kutta time-stepping                                                          
    #                                                                                                                           
    import first_derivative_sbp_operators
    import numpy as np
    import boundarycondition

    # initialize arrays for computing derivatives                                                                               
    dxF = np.zeros((nf))
    dyF = np.zeros((nf))
    
    Ft = np.zeros((1))
    
    # compute the elastic rates                                                                                                 
    for i in range(0, nx):
        for j in range(0, ny):
            
            x = X[i,j]
            y = Y[i,j]
            
            pointSource(Ft, dx, dy, i, j, x, y, t, source_parameter)

            # extract material parameters                                                                                       
            rho = Mat[i,j,0]
            lam = Mat[i,j,1]

            # compute spatial gradients of fields (particle velocity and stress)                                                
            first_derivative_sbp_operators.dx2d(dxF, F, nx, i, j, dx, order)
            first_derivative_sbp_operators.dy2d(dyF, F, ny, i, j, dy, order)

            # pressure equation           
            D[i,j,0] = lam*(dxF[1] + dyF[2]) + Ft

            # momentum equation
            D[i,j,1] = (1.0/rho)*dxF[0]
            D[i,j,2] = (1.0/rho)*dyF[0]

            

    # impose boundary conditions using penalty: SAT
    acoustics_impose_bc(D, F, Mat, nx, ny, nf, dx, dy, order, r)

def elastic_impose_bc(D, F, Mat, nx, ny, nf, dx, dy, order, r):
    # impose boundary conditions
    import numpy as np
    import boundarycondition

    # penalty weights
    if order==2:
        hx = 0.5*dx
        hy = 0.5*dy
        
    if order== 4:
        hx = (17.0/48.0)*dx
        hy = (17.0/48.0)*dy

    if order==6:
        hx = 13649.0/43200.0*dx
        hy = 13649.0/43200.0*dy
    
    BF0x = np.zeros((ny, nf))
    BFnx = np.zeros((ny, nf))

    BF0y = np.zeros((nx, nf))
    BFny = np.zeros((nx, nf))
      
    # compute SAT terms
    boundarycondition.bcm2dx(BF0x, F, Mat, nx, ny, r,'elastic')
    boundarycondition.bcp2dx(BFnx, F, Mat, nx, ny, r,'elastic')
    
    boundarycondition.bcm2dy(BF0y, F, Mat, nx, ny, r,'elastic')
    boundarycondition.bcp2dy(BFny, F, Mat, nx, ny, r,'elastic')

    # penalize boundaries with the SAT terms

    D[0,:,:] =  D[0,:,:] -  1.0/hx*BF0x[:,:]
    D[nx-1,:,:] =  D[nx-1,:,:] -  1.0/hx*BFnx[:,:]
    

    D[:,0,:] =  D[:,0,:] -  1.0/hy*BF0y[:,:]
    D[:,ny-1,:] =  D[:,ny-1,:] -  1.0/hy*BFny[:,:]
    
    

def acoustics_impose_bc(D, F, Mat, nx, ny, nf, dx, dy, order, r):
    # impose boundary conditions                                                                                                       
    import numpy as np
    import boundarycondition

    # penalty weights                                                                                                                  
    if order==2:
        hx = 0.5*dx
        hy = 0.5*dy

    if order== 4:
        hx = (17.0/48.0)*dx
        hy = (17.0/48.0)*dy

    if order==6:
        hx = 13649.0/43200.0*dx
        hy = 13649.0/43200.0*dy

    BF0x = np.zeros((ny, nf))
    BFnx = np.zeros((ny, nf))

    BF0y = np.zeros((nx, nf))
    BFny = np.zeros((nx, nf))

    # compute SAT terms                           
    boundarycondition.bcm2dx(BF0x, F, Mat, nx, ny, r, 'acoustics')
    boundarycondition.bcp2dx(BFnx, F, Mat, nx, ny, r, 'acoustics')   
    
    boundarycondition.bcm2dy(BF0y, F, Mat, nx, ny, r, 'acoustics')
    boundarycondition.bcp2dy(BFny, F, Mat, nx, ny, r, 'acoustics')

    # penalize boundaries with the SAT terms                                                                                         
    D[0,:,:] =  D[0,:,:] -  1.0/hx*BF0x[:,:]
    D[nx-1,:,:] =  D[nx-1,:,:] -  1.0/hx*BFnx[:,:]


    D[:,0,:] =  D[:,0,:] -  1.0/hy*BF0y[:,:]
    D[:,ny-1,:] =  D[:,ny-1,:] -  1.0/hy*BFny[:,:]
    

def pointSource(Ft, dx, dy, ix, jy, x, y, t, source_parameter):
    
    import numpy as np
    
    sigma_x = 2*dx
    sigma_y = 2*dy
    
    #source_parameter = [x0, y0, t0, T, M0, source_type]
    x0 = source_parameter[0]
    y0 = source_parameter[1]
    t0 = source_parameter[2]
    T  = source_parameter[3]
    M0 = source_parameter[4]
    source_type = source_parameter[5]
        
    
    g = 1./np.sqrt(2*np.pi*sigma_x)*1./np.sqrt(2*np.pi*sigma_x)*np.exp(-(x-x0)**2/(2*sigma_x**2))*np.exp(-(y-y0)**2/(2*sigma_y**2))
    
    if (source_type == 'Gaussian'):
        sigma0_t = T
    
        f = 1./np.sqrt(2*np.pi*sigma0_t)*np.exp(-(t-t0)**2/(2*sigma0_t**2))
    
    
        #Ft[:] = 1000.0*g*f
        
    if (source_type == 'Brune'):
        
    
        f = (t-t0)/T*np.exp(-(t-t0)/T)
    
    
 
    Ft[:] = M0*g*f
    

    

    
    
