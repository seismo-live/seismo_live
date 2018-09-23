def couple_GL(u,v,w,NP,num_elements, dx, rho, mu,x):
    #
    import quadraturerules
    import Lagrange
    import interfacedata
    import numpy as np
    

    Hu = np.zeros(len(u)) 
    Hv = np.zeros(len(v)) 
    
    PH_0 = np.zeros(NP)
    PH_1 = np.zeros(NP)
    for i in range(1,NP+1):
        PH_0[i-1] = Lagrange.lagrange_basis(NP,i,1,x)
        PH_1[i-1] = Lagrange.lagrange_basis(NP,i,-1,x)
    
    PH_0 = np.transpose(PH_0)
    PH_1 = np.transpose(PH_1)
    
    for j in range(1,num_elements):

        # extract local element degrees of freedom for the wave field
        u_m = u[(j-1)*NP:NP+(j-1)*NP]
        v_m = v[(j-1)*NP:NP+(j-1)*NP]
        
        u_p = u[(j-1)*NP+NP:2*NP+(j-1)*NP]
        v_p = v[(j-1)*NP+NP:2*NP+(j-1)*NP]

        # evaluate the Lagrange interpolant of the wave field at the element boundaries
        u0 = Lagrange.interpol(NP,1,x,u_m)
        v0 = Lagrange.interpol(NP,1,x,v_m)
        
        u1 = Lagrange.interpol(NP,-1,x,u_p)
        v1 = Lagrange.interpol(NP,-1,x,v_p)

        # extract local element degrees of freedom for the material parameters
        mu_m = mu[(j-1)*NP:NP+(j-1)*NP]
        rho_m = rho[(j-1)*NP:NP+(j-1)*NP]

        mu_p = mu[(j-1)*NP+NP:2*NP+(j-1)*NP]
        rho_p = rho[(j-1)*NP+NP:2*NP+(j-1)*NP]

        # evaluate the Lagrange interpolant of the the material parameters at the element boundaries
        mu_0 = Lagrange.interpol(NP,1,x,mu_m)
        rho_0 = Lagrange.interpol(NP,1,x,rho_m)

        mu_1 = Lagrange.interpol(NP,-1,x,mu_p)
        rho_1 = Lagrange.interpol(NP,-1,x,rho_p)

        
        # compute local wave speeds and shear impedance
        c1s = np.sqrt(mu_0/rho_0)
        z1s = rho_0 * c1s

        c2s = np.sqrt(mu_1/rho_1)
        z2s = rho_1 * c2s

        # generate interface data obeying locked interface conditions
        interface_condition_return = interfacedata.interface_condition(u0, v0, u1, v1, rho_0, mu_0, rho_1, mu_1)
        V_m = interface_condition_return['V_m']
        V_p = interface_condition_return['V_p']
        T_m = interface_condition_return['T_m']
        T_p = interface_condition_return['T_p'] 

        p0 = 0.5 * (z1s*u0+v0)
        p1 = 0.5 * (z1s*V_m+T_m)
       
        q1 = 0.5 * (z2s*u1-v1)
        q0 = 0.5 * (z2s*V_p-T_p)

        #interface flux penalty functions 
        Hu[(j-1)*NP:NP+(j-1)*NP] = Hu[(j-1)*NP:NP+(j-1)*NP] + 2/(dx[0:NP])* (p0 - p1)*(PH_0/w)
        Hv[(j-1)*NP:NP+(j-1)*NP] = Hv[(j-1)*NP:NP+(j-1)*NP] + 2/(z1s*dx[0:NP])*(p0 - p1)*(PH_0/w)
               
        Hu[(j-1)*NP+NP:2*NP+(j-1)*NP] = Hu[(j-1)*NP+NP:2*NP+(j-1)*NP] + 2/(dx[0:NP])*(q1 - q0)*(PH_1/w)
        Hv[(j-1)*NP+NP:2*NP+(j-1)*NP] = Hv[(j-1)*NP+NP:2*NP+(j-1)*NP] -2/(z2s*dx[0:NP])*(q1 - q0)*(PH_1/w)
        
    return {'Hu':Hu,'Hv':Hv}

def couple_friction_GL(u_1,v_1,u_2,v_2,S, psi, w,NP, dx1, dx2, rho1,mu1,rho2,mu2,x,alpha,Tau_0,fric_law):
    import quadraturerules
    import Lagrange
    import numpy as np
    import interfacedata
    
    Hu_1 = np.zeros(len(u_1)) 
    Hv_1 = np.zeros(len(v_1)) 
     
    Hu_2 = np.zeros(len(u_2)) 
    Hv_2 = np.zeros(len(v_2)) 
    
    PH_0 = np.zeros(NP)
    PH_1 = np.zeros(NP)
    
    for i in range(1,NP+1):
        PH_0[i-1] = Lagrange.lagrange_basis(NP,i,1,x)
        PH_1[i-1] = Lagrange.lagrange_basis(NP,i,-1,x)
    
    PH_0 = np.transpose(PH_0)
    PH_1 = np.transpose(PH_1)

    
    # extract local element degrees of freedom for the wave field
    u_m = u_1[-NP:]
    v_m = v_1[-NP:]
        
    u_p = u_2[0:NP]
    v_p = v_2[0:NP]

    # evaluate the Lagrange interpolant of the wave field at the element boundaries
    u0 = Lagrange.interpol(NP,1,x,u_m)
    v0 = Lagrange.interpol(NP,1,x,v_m)
        
    u1 = Lagrange.interpol(NP,-1,x,u_p)
    v1 = Lagrange.interpol(NP,-1,x,v_p)
    

    # extract local element degrees of freedom for the material parameters
    mu_m = mu1[-NP:];
    rho_m = rho1[-NP:];

    mu_p = mu2[0:NP];
    rho_p = rho2[0:NP];

    # evaluate the Lagrange interpolant of the the material parameters at the element boundaries
    mu_0 = Lagrange.interpol(NP,1,x,mu_m)
    rho_0 = Lagrange.interpol(NP,1,x,rho_m)

    mu_1 = Lagrange.interpol(NP,-1,x,mu_p)
    rho_1 = Lagrange.interpol(NP,-1,x,rho_p)

    
    # compute local wave speeds and shear impedance 
    c1s = np.sqrt(mu_0/rho_0)
    z1s = rho_0 * c1s

    c2s = np.sqrt(mu_1/rho_1)
    z2s = rho_1 * c2s

    # generate interface data obeying frictional laws
    friction_law_return = interfacedata.friction_law(u0, v0, u1, v1, S, psi, rho_0, mu_0, rho_1, mu_1, alpha,Tau_0,fric_law)
    V_m = friction_law_return['V_m']
    V_p = friction_law_return['V_p']
    T_m = friction_law_return['T_m']
    T_p = friction_law_return['T_p']

    
    vv = np.abs(V_p-V_m)
    
    
    p0 = 0.5 * (z1s*u0+v0)
    p1 = 0.5 * (z1s*V_m+T_m)
       
    q1 = 0.5 * (z2s*u1-v1)
    q0 = 0.5 * (z2s*V_p-T_p)
    
    #interface flux penalty functions 
    Hu_1[-NP:] = Hu_1[-NP:] + 2/(dx1[-1])* (p0 - p1)*(PH_0/w)
    Hv_1[-NP:] = Hv_1[-NP:] + 2/(z1s*dx1[-1])*(p0 - p1)*(PH_0/w)
               
    Hu_2[0:NP] = Hu_2[0:NP] + 2/(dx2[0])*(q1 - q0)*(PH_1/w)
    Hv_2[0:NP] = Hv_2[0:NP] -2/(z2s*dx2[0])*(q1 - q0)*(PH_1/w)

    return {'Hu_1':Hu_1,'Hv_1':Hv_1,'Hu_2':Hu_2,'Hv_2':Hv_2,'vv':vv}
