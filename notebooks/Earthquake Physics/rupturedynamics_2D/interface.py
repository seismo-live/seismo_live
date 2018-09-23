def couple_friction(Hu_1,Hv_1,Hu_2,Hv_2, u_1,v_1,u_2,v_2, S, psi, order, nx, dx1, dx2, rho1, mu1,rho2,mu2, x, friction_parameters, dslip, dpsi):
    
    import numpy as np
    import interfacedata
    
    #Hu_1 = np.zeros(len(u_1)) 
    #Hv_1 = np.zeros(len(v_1)) 
     
    #Hu_2 = np.zeros(len(u_2)) 
    #Hv_2 = np.zeros(len(v_2)) 
    

    
    # extract local wave field
    u0 = u_1[nx-1,:]
    v0 = v_1[nx-1,:]

    u1 = u_2[0,:]
    v1 = v_2[0,:]
    
    # extract local material parameters
    mu_0 = mu1[nx-1,:]
    rho_0 = rho1[nx-1,:]

    mu_1 = mu2[0,:]
    rho_1 = rho2[0,:]

    
    # compute local wave speeds and shear impedance 
    c1s = np.sqrt(mu_0/rho_0)
    z1s = rho_0 * c1s

    c2s = np.sqrt(mu_1/rho_1)
    z2s = rho_1 * c2s


    # generate interface data obeying frictional laws
    friction_law_return = interfacedata.friction_law(u0, v0, u1, v1, S, psi, rho_0, mu_0, rho_1, mu_1,friction_parameters)
    V_m = friction_law_return['V_m']
    V_p = friction_law_return['V_p']
    T_m = friction_law_return['T_m']
    T_p = friction_law_return['T_p']
    
    #friction_parameters = [fric_law, alpha, Tau_0, L0, f0, a, b, V0, sigma_n, alp_s, alp_d, D_c]    
    #                        0         1      2     3   4  5  6   7    8       9      10    11 
    L0 = friction_parameters[3]
    f0 = friction_parameters[4]
    b = friction_parameters[6]
    V0 = friction_parameters[7]
    a = friction_parameters[5]
    
    vv = np.abs(V_p-V_m)
    
    dslip[:,:] = vv
    dpsi[:,:] = b*V0/L0*np.exp(-(psi-f0)/b) - vv*b/L0
    
    
    p0 = 0.5 * (z1s*u0+v0)
    p1 = 0.5 * (z1s*V_m+T_m)
       
    q1 = 0.5 * (z2s*u1-v1)
    q0 = 0.5 * (z2s*V_p-T_p)

    hx1 = np.zeros((1,1))
    hx2 = np.zeros((1,1))
    
    penaltyweights(hx1, order, dx1)
    penaltyweights(hx2, order, dx2)
    
    #interface flux penalty functions 
    
    Hu_1[nx-1,:] = Hu_1[nx-1,:] - 2/(hx1*rho_0)* (p0 - p1)
    Hv_1[nx-1,:] = Hv_1[nx-1,:] - mu_0*2/(z1s*hx1)*(p0 - p1)
               
    Hu_2[0,:] = Hu_2[0,:] - 2/(rho_1*hx2)*(q1 - q0)
    Hv_2[0,:] = Hv_2[0,:] +2*mu_1/(z2s*hx2)*(q1 - q0)



def penaltyweights(h11, order, dx):

    # penalty weights                                                                                                                                    
    if order==2:
        h11[:,:] = 0.5*dx
    if order== 4:
        h11[:,:] = (17.0/48.0)*dx

    if order==6:
        h11[:,:] = 13649.0/43200.0*dx

        
        
def Interface_Fault(BF, F, Fhat, Mat, nx, ny, mode, side):
    import numpy as np
    for j in range(ny):
        
        if side == 'left':
            if mode == "II":
                vx = F[nx-1, j, 0]
                vy = F[nx-1, j, 1]
                sxx = F[nx-1, j, 2]
                sxy = F[nx-1, j, 4]

                Vx = Fhat[j, 0]
                Vy = Fhat[j, 1]
                Sxx = Fhat[j, 2]
                Sxy = Fhat[j, 3]

                # extract material parameters
                rho = Mat[nx-1, j, 0]
                Lambda = Mat[nx-1, j, 1]
                mu = Mat[nx-1, j, 2]

                # compute wave speeds
                cp = np.sqrt((2.0*mu + Lambda)/rho)
                cs = np.sqrt(mu/rho)

                # compute impedances
                zp = rho*cp
                zs = rho*cs
            
                px = 0.5*(zp*vx + sxx)
                py = 0.5*(zs*vy + sxy)

                Px = 0.5*(zp*Vx + Sxx)
                Py = 0.5*(zs*Vy + Sxy)

                # compute SAT terms to be used in imposing bou
                BF[j,0] = 1.0/rho*(px - Px)
                BF[j,1] = 1.0/rho*(py - Py)
                BF[j,2] = (2.0*mu + Lambda)/zp*(px - Px)
                BF[j,3] = Lambda/zp*(px - Px)
                BF[j,4] = mu/zs*(py - Py)
                
            if mode == "III":
                vz = F[nx-1, j, 0]
                sxz = F[nx-1, j, 1]
                syz = F[nx-1, j, 2]

                Vz = Fhat[j, 0]
                Sxz = Fhat[j, 1]
                Syz = Fhat[j, 2]

                # extract material parameters
                rho = Mat[nx-1, j, 0]
                Lambda = Mat[nx-1, j, 1]
                mu = Mat[nx-1, j, 2]

                # compute wave speeds
                cs = np.sqrt(mu/rho)

                # compute impedances
                zs = rho*cs
            
                px = 0.5*(zs*vz + sxz)
                
                Px = 0.5*(zs*Vz + Sxz)
                

                # compute SAT terms to be used in imposing bou
                BF[j,0] = 1.0/rho*(px - Px)
                BF[j,1] = mu/zs*(px - Px)
                BF[j,2] = 0
                



        if side == "right":
            if mode == "II":
                vx = F[0, j, 0]
                vy = F[0, j, 1]
                sxx = F[0, j, 2]
                sxy = F[0, j, 4]

                Vx = Fhat[j, 0]
                Vy = Fhat[j, 1]
                Sxx = Fhat[j, 2]
                Sxy = Fhat[j, 3]

                # extract material parameters
                rho = Mat[0, j, 0]
                Lambda = Mat[0, j, 1]
                mu = Mat[0, j, 2]

                # compute wave speeds
                cp = np.sqrt((2.0*mu + Lambda)/rho)
                cs = np.sqrt(mu/rho)

                # compute impedances
                zp = rho*cp
                zs = rho*cs
            
                px = 0.5*(zp*vx - sxx)
                py = 0.5*(zs*vy - sxy)

                Px = 0.5*(zp*Vx - Sxx)
                Py = 0.5*(zs*Vy - Sxy)

                # compute SAT terms to be used in imposing bou
                BF[j,0] = 1.0/rho*(px - Px)
                BF[j,1] = 1.0/rho*(py - Py)
                BF[j,2] = -(2.0*mu + Lambda)/zp*(px - Px)
                BF[j,3] = -Lambda/zp*(px - Px)
                BF[j,4] =  -mu/zs*(py - Py)
                
                
                ######
                
            if mode == "III":
                vz = F[0, j, 0]
                sxz = F[0, j, 1]
                syz = F[0, j, 2]

                Vz = Fhat[j, 0]
                Sxz = Fhat[j, 1]
                Syz = Fhat[j, 2]

                # extract material parameters
                rho = Mat[0, j, 0]
                Lambda = Mat[0, j, 1]
                mu = Mat[0, j, 2]

                # compute wave speeds
                cs = np.sqrt(mu/rho)

                # compute impedances
                zs = rho*cs
            
                px = 0.5*(zs*vz - sxz)
                
                Px = 0.5*(zs*Vz - Sxz)
                

                # compute SAT terms to be used in imposing bou
                BF[j,0] = 1.0/rho*(px - Px)
                BF[j,1] = -mu/zs*(px - Px)
                BF[j,2] = 0
                
    



