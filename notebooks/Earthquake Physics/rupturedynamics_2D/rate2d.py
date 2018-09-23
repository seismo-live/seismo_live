def elastic_rate2d(D_l, F_l, Mat_l, X_l, Y_l, t, nf, nx, ny, dx, dy, order, r_l, source_parameter, D_r, F_r, Mat_r, X_r, Y_r, r_r, friction_parameters, slip, psi, dslip, dpsi,fric_law, FaultOutput0, stage, Y0, mode):
    # we compute rates that will be used for Runge-Kutta time-stepping
    #
    import first_derivative_sbp_operators
    import numpy as np
    import boundarycondition

    # initialize arrays for computing derivatives
    dxF_l = np.zeros((nf))
    dyF_l = np.zeros((nf))
    Ft_l = np.zeros((1))
    M_l = source_parameter[6]
    
    dxF_r = np.zeros((nf))
    dyF_r = np.zeros((nf))
    Ft_r = np.zeros((1))
    M_r = source_parameter[6]
    
    Fhat_l = np.zeros((ny, nf))
    Fhat_r = np.zeros((ny, nf))
    
    # compute the elastic rates
    for i in range(0, nx):
        for j in range(0, ny):
            
            x_l = X_l[i,j]
            y_l = Y_l[i,j]
            pointSource(Ft_l, dx, dy, i, j, x_l, y_l, t, source_parameter)
    
            # extract material parameters
            rho_l = Mat_l[i,j,0]
            lam_l = Mat_l[i,j,1]
            mu_l  = Mat_l[i,j,2]
            
            # compute spatial gradients of fields (particle velocity and stress)
            first_derivative_sbp_operators.dx2d(dxF_l, F_l, nx, i, j, dx, order)
            first_derivative_sbp_operators.dy2d(dyF_l, F_l, ny, i, j, dy, order)
            
            if mode == 'II':
                # momentum equation
                D_l[i,j,0] = 1.0/rho_l*(dxF_l[2] + dyF_l[4]) +  M_l[0]*Ft_l
                D_l[i,j,1] = 1.0/rho_l*(dxF_l[4] + dyF_l[3]) +  M_l[1]*Ft_l

                # Hooke's law
                D_l[i,j,2] = (2.0*mu_l + lam_l)*dxF_l[0] + lam_l*dyF_l[1] + M_l[2]*Ft_l
                D_l[i,j,3] = (2.0*mu_l + lam_l)*dyF_l[1] + lam_l*dxF_l[0] + M_l[3]*Ft_l
                D_l[i,j,4] = mu_l*(dyF_l[0] + dxF_l[1])               + M_l[4]*Ft_l
                
                
            if mode == 'III':
                # momentum equation
                D_l[i,j,0] = 1.0/rho_l*(dxF_l[1] + dyF_l[2]) +  M_l[0]*Ft_l
                
                # Hooke's law
                D_l[i,j,1] = mu_l*(dxF_l[0])               + M_l[1]*Ft_l
                D_l[i,j,2] = mu_l*(dyF_l[0])               + M_l[4]*Ft_l    
            
            #--------------------------------
                
            x_r = X_r[i,j]
            y_r = Y_r[i,j]
            pointSource(Ft_r, dx, dy, i, j, x_r, y_r, t, source_parameter)
    
            # extract material parameters
            rho_r = Mat_r[i,j,0]
            lam_r = Mat_r[i,j,1]
            mu_r  = Mat_r[i,j,2]
            
            # compute spatial gradients of fields (particle velocity and stress)
            first_derivative_sbp_operators.dx2d(dxF_r, F_r, nx, i, j, dx, order)
            first_derivative_sbp_operators.dy2d(dyF_r, F_r, ny, i, j, dy, order)
            
            if mode == 'II':
                # momentum equation
                D_r[i,j,0] = 1.0/rho_r*(dxF_r[2] + dyF_r[4]) +  M_r[0]*Ft_r
                D_r[i,j,1] = 1.0/rho_r*(dxF_r[4] + dyF_r[3]) +  M_r[1]*Ft_r

                # Hooke's law
                D_r[i,j,2] = (2.0*mu_r + lam_r)*dxF_r[0] + lam_r*dyF_r[1] + M_r[2]*Ft_r
                D_r[i,j,3] = (2.0*mu_r + lam_r)*dyF_r[1] + lam_r*dxF_r[0] + M_r[3]*Ft_r
                D_r[i,j,4] = mu_r*(dyF_r[0] + dxF_r[1])               + M_r[4]*Ft_r
                
            if mode == 'III':
                # momentum equation
                D_r[i,j,0] = 1.0/rho_r*(dxF_r[1] + dyF_r[2]) +  M_r[0]*Ft_r
                
                # Hooke's law
                D_r[i,j,1] = mu_r*(dxF_r[0])               + M_r[1]*Ft_l
                D_r[i,j,2] = mu_r*(dyF_r[0])               + M_r[4]*Ft_l    
            
    

    Y = Y_r[0, :]
    interfacecondition2d(F_l, F_r, Fhat_l, Fhat_r, Mat_l, Mat_r, nx, ny, friction_parameters, slip, psi, dslip, dpsi, fric_law, FaultOutput0, Y, t, stage, mode, Y0)
    
    # impose boundary conditions using penalty: SAT
    elastic_impose_bc(D_l, F_l, Fhat_l, Mat_l, nx, ny, nf, dx, dy, order, r_l,  'left', mode)
    elastic_impose_bc(D_r, F_r, Fhat_r, Mat_r, nx, ny, nf, dx, dy, order, r_r, 'right', mode)

    

    
def interfacecondition2d(F_l, F_r, Fhat_l, Fhat_r, Mat_l, Mat_r, nx, ny, friction_parameters, slip, psi, dslip, dpsi, fric_law, FaultOutput0, Y, t, stage, mode, Y0): 
    import numpy as np
    import interfacedata
    # Generate the hat-variables
    
    #vx_l = np.zeros((ny))
    #vx_r = np.zeros((ny))
    
    #vy_l = np.zeros((ny))
    #vy_r = np.zeros((ny))
    
    #Tx_l = np.zeros((ny))
    #Tx_r = np.zeros((ny))
    
    #Ty_l = np.zeros((ny))
    #Ty_r = np.zeros((ny))
    
    vx_m = np.zeros((1))
    vy_m = np.zeros((1))
    
    Tx_m = np.zeros((1))
    Ty_m = np.zeros((1))
    
    vx_p = np.zeros((1))
    vy_p = np.zeros((1))
    
    Tx_p = np.zeros((1))
    Ty_p = np.zeros((1))
    
    friction_parameters_index = [12]
    
    for j in range(0, ny):
                
        r = np.sqrt((Y[j]-Y0)**2)

        F = 0.0
        if(r < 3.0):
            F = np.exp(r**2/(r**2 - 9.0))

        G = 0.0
        if (t > 0.0 and t < 1.0):
            G = np.exp((t-1.0)**2/(t*(t-2.0)))
            
        if (t >= 1.0):
            G = 1.0
        
        tau = friction_parameters[2, j]
        if fric_law == 'RS':
            #tau = friction_parameters[2, j]
            tau = friction_parameters[2, j] + 25.0*F*G
        
        rho_r = Mat_r[0,j,0]
        lam_r = Mat_r[0,j,1]
        mu_r  = Mat_r[0,j,2]
        
        rho_l = Mat_l[-1,j,0]
        lam_l = Mat_l[-1,j,1]
        mu_l  = Mat_l[-1,j,2]
            
            
        twomulam_r = 2*mu_r + lam_r
        twomulam_l = 2*mu_l + lam_l
        
        if mode == 'II':
            vx_l = F_l[-1, j, 0]
            vx_r = F_r[0, j, 0]
    
            vy_l = F_l[-1, j, 1]
            vy_r = F_r[0, j, 1]
    
            Tx_l = F_l[-1, j, 2]
            Tx_r = F_r[0, j, 2]
    
            Ty_l = F_l[-1, j, 4]
            Ty_r = F_r[0, j, 4]
        
            return_int = interfacedata.interface_condition(vx_l, Tx_l, vx_r, Tx_r,rho_l,twomulam_l,rho_r,twomulam_r)
    
            vx_m = return_int['V_m']
            vx_p = return_int['V_p']
            Tx_m = return_int['T_m']
            Tx_p = return_int['T_p']
        
            Fhat_l[j, 0] = vx_m
            Fhat_r[j, 0] = vx_p
        
            Fhat_l[j, 2] = Tx_m
            Fhat_r[j, 2] = Tx_p
        
            sigma_n = np.max([0.0, -(Tx_m+friction_parameters[8,j])])
        
            friction_parameters_index = [friction_parameters[0,j] , friction_parameters[1,j], tau, friction_parameters[3,j] ,friction_parameters[4,j] ,friction_parameters[5,j] ,friction_parameters[6,j] ,friction_parameters[7,j] ,sigma_n ,friction_parameters[9,j] ,friction_parameters[10,j] , friction_parameters[11,j]]
        
        
        
            return_int = interfacedata.friction_law(vy_l, Ty_l, vy_r, Ty_r, slip[j], psi[j], rho_l, mu_l, rho_r, mu_r, friction_parameters_index, fric_law)
            vy_m = return_int['V_m']
            vy_p = return_int['V_p']
            Ty_m = return_int['T_m']
            Ty_p = return_int['T_p']
        
            Fhat_l[j, 1] = vy_m
            Fhat_r[j, 1] = vy_p
        
            Fhat_l[j, 3] = Ty_m
            Fhat_r[j, 3] = Ty_p
            
        
        if mode == 'III':
            
            vy_l = F_l[-1, j, 0]
            vy_r = F_r[0, j, 0]
    
            Ty_l = F_l[-1, j, 1]
            Ty_r = F_r[0, j, 1]
            
            vx_p = 0
            vx_m = 0
            Tx_m = 0
            Ty_m = 0
            
        
            sigma_n = np.max([0.0, -(friction_parameters[8,j])])
        
            friction_parameters_index = [friction_parameters[0,j] , friction_parameters[1,j], tau, friction_parameters[3,j] ,friction_parameters[4,j] ,friction_parameters[5,j] ,friction_parameters[6,j] ,friction_parameters[7,j] ,sigma_n ,friction_parameters[9,j] ,friction_parameters[10,j] , friction_parameters[11,j]]
        
        
        
            return_int = interfacedata.friction_law(vy_l, Ty_l, vy_r, Ty_r, slip[j], psi[j], rho_l, mu_l, rho_r, mu_r, friction_parameters_index, fric_law)
            vy_m = return_int['V_m']
            vy_p = return_int['V_p']
            Ty_m = return_int['T_m']
            Ty_p = return_int['T_p']
        
            Fhat_l[j, 0] = vy_m
            Fhat_r[j, 0] = vy_p
        
            Fhat_l[j, 1] = Ty_m
            Fhat_r[j, 1] = Ty_p
        
        
        #friction_parameters = [fric_law, alpha, Tau_0, L0, f0, a, b, V0, sigma_n, alp_s, alp_d, D_c]    
        #                        0         1      2     3   4  5  6   7    8       9      10    11 
        L0 = friction_parameters[3,j]
        f0 = friction_parameters[4,j]
        a  = friction_parameters[5,j]
        b  = friction_parameters[6,j]
        V0 = friction_parameters[7,j]
        
        
    
        vv = np.abs(vy_p-vy_m)
    
        dslip[j,:] = vv
        
        dpsi[j,:] = b*V0/L0*np.exp(-(psi[j,:]-f0)/b) - vv*b/L0
        
        
        if stage == 0:
            FaultOutput0[j, 0] = np.abs(vx_p-vx_m)
            FaultOutput0[j, 1] = np.abs(vy_p-vy_m)
            FaultOutput0[j, 2] = Tx_m+friction_parameters[8,j]
            FaultOutput0[j, 3] = Ty_m+friction_parameters[2,j]
        
        
        
    
    
    
    
    
def elastic_impose_bc(D, F, Fhat, Mat, nx, ny, nf, dx, dy, order, r, side, mode):
    # impose boundary conditions
    import numpy as np
    import boundarycondition
    import interface

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
    if side == 'left' :
        boundarycondition.bcm2dx(BF0x, F, Mat, nx, ny, r, mode)
        interface.Interface_Fault(BFnx, F, Fhat, Mat, nx, ny, mode, side)
        
    if side == 'right' : 
        boundarycondition.bcp2dx(BFnx, F, Mat, nx, ny, r,mode)
        interface.Interface_Fault(BF0x, F, Fhat, Mat, nx, ny, mode, side)
    
    boundarycondition.bcm2dy(BF0y, F, Mat, nx, ny, r, mode)
    boundarycondition.bcp2dy(BFny, F, Mat, nx, ny, r, mode)
        
   

    # penalize boundaries with the SAT terms
    #if side == 'left' :
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
    

    

    
    
