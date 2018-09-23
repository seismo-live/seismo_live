def wave_dG(u, v, D, NP, num_element, dx, w, r0, rn, rho, mu,  x, t):
    
    import spatialdiff
    import boundary
    import interface
    import numpy as np

    # compute spatial derivatives
    ux = spatialdiff.Dx(u,D,NP,num_element,dx)
    vx = spatialdiff.Dx(v,D,NP,num_element,dx)

    # compute flux fluctuations with appropriate penalty weights
    # first for inter-element boundaries
    couple_GL_return = interface.couple_GL(u, v, w, NP, num_element, dx, rho, mu, x)
    Cu = couple_GL_return['Hu']
    Cv = couple_GL_return['Hv']

    # first for physical boundaries
    BC_GL_return =  boundary.BC_GL(u, v, w, dx, r0, rn, rho, mu,  x, t, NP)
    Bu = BC_GL_return['Hu']
    Bv = BC_GL_return['Hv']

    # left element boundary 
    BClGL_return = boundary.BCl_GL(u, v, w, dx, r0, rho, mu,  x, t, NP)
    Bul = BClGL_return['Hu_l']
    Bvl = BClGL_return['Hv_l']

    # right element boundary  
    BCrGL_return = boundary.BCr_GL(u, v, w, dx, rn, rho, mu,  x, t, NP)
    Bur = BCrGL_return['Hu_r']
    Bvr = BCrGL_return['Hv_r']
    
    # generate the right handside that will be used in the time-stepping scheme:
    # spatial approximations of the PDE with weak enforcement of inter-element and physical boundary conditions
    Hu = 1.0/rho*(vx - Bul - Bur -Cu)
    Hv = mu*(ux - Bvl - Bvr - Cv)
    
    return {'Hu':Hu,'Hv':Hv}




def wave_1D_Friction(u_1, v_1,u_2,v_2,S, psi, D, NP, num_element1, num_element2, dx1, dx2, w, r0, rn, rho1,mu1,rho2,mu2, x, t,alpha,Tau_0,fric_law):
    
    import spatialdiff
    import boundary
    import interface
    import numpy as np

    # compute spatial derivatives
    ux_1 = spatialdiff.Dx(u_1,D,NP,num_element1,dx1)
    vx_1 = spatialdiff.Dx(v_1,D,NP,num_element1,dx1)
    
    ux_2 = spatialdiff.Dx(u_2,D,NP,num_element2,dx2)
    vx_2 = spatialdiff.Dx(v_2,D,NP,num_element2,dx2)
    
    # compute flux fluctuations
    # first for inter-element boundaries
    couple_GL_return = interface.couple_GL(u_1, v_1, w, NP, num_element1, dx1, rho1, mu1, x)
    Cu_1 = couple_GL_return['Hu']
    Cv_1 = couple_GL_return['Hv']
    
    # first for inter-element boundaries 
    couple_GL_return = interface.couple_GL(u_2, v_2, w, NP, num_element2, dx2, rho2, mu2, x)
    Cu_2 = couple_GL_return['Hu']
    Cv_2 = couple_GL_return['Hv']
    
    # left element boundary 
    BClGL_return = boundary.BCl_GL(u_1,v_1, w, dx1, r0, rho1, mu1,  x, t, NP)
    Bul_1 = BClGL_return['Hu_l']
    Bvl_1 = BClGL_return['Hv_l']

    # right element boundary  
    BCrGL_return = boundary.BCr_GL(u_2,v_2, w, dx2, rn, rho2, mu2,  x, t, NP)
    Bur_2 = BCrGL_return['Hu_r']
    Bvr_2 = BCrGL_return['Hv_r']
    
    # frictional inter-element boundaries  
    couple_friction_return = interface.couple_friction_GL(u_1,v_1,u_2,v_2,S,psi,w,NP,dx1,dx2, rho1,mu1,rho2,mu2, x,alpha,Tau_0,fric_law)
    Bur_1 = couple_friction_return['Hu_1']
    Bvr_1 = couple_friction_return['Hv_1']
    Bul_2 = couple_friction_return['Hu_2']
    Bvl_2 = couple_friction_return['Hv_2']
    vv = couple_friction_return['vv']
    
    # generate the right handside
    Hu_1 = 1.0/rho1*(vx_1 - Bul_1 - Bur_1 - Cu_1)
    Hv_1 = mu1*(ux_1 - Bvl_1 - Bvr_1 - Cv_1)
        
    Hu_2 = 1.0/rho2*(vx_2 - Bul_2 - Bur_2 - Cu_2)
    Hv_2 = mu2*(ux_2 - Bvl_2 - Bvr_2 - Cv_2)
    
    H_d = vv

    
    H_psi = 0.0
    
    L0 = 1.0 # 0.02
    f0 = 0.6
    b = 0.012
    V0 = 1.0e-6
    a = 0.008

    H_psi = b*V0/L0*np.exp(-(psi-f0)/b) - vv*b/L0

    return {'Hu_1':Hu_1,'Hv_1':Hv_1,'Hu_2':Hu_2,'Hv_2':Hv_2,'H_d':H_d,'H_psi':H_psi}

