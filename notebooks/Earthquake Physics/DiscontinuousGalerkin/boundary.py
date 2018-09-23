# This fuction computes flux penalization at the physical boundaries
# left boundary
def BCl_GL(u,v,w,dx,r0,rho,mu,x,t,NP):
    
    import boundarydata
    import numpy as np
    import Lagrange
    
    # extract local solutions in the first element
    u1 = u[0:NP]
    v1 = v[0:NP]
    
    # extrapolate solutions to the boundaries
    NP = len(x)
    
    u_0 = Lagrange.interpol(NP, -1, x, u1)   
    v_0 = Lagrange.interpol(NP, -1, x, v1) 
    
    mu_p = mu[0:NP];
    rho_p = rho[0:NP];

    mu_1 = Lagrange.interpol(NP,-1,x,mu_p)
    rho_1 = Lagrange.interpol(NP,-1,x,rho_p)

    cs = np.sqrt(mu_1/rho_1)
    zs = rho_1 * cs

    # initialize basis vector
    PH_0 = np.zeros(NP)

    # evaluate basis vector at the boundary
    for i in range (1,NP+1):
        PH_0[i-1] = Lagrange.lagrange_basis(NP,i,-1,x)   
        
        
    PH_0 = np.transpose(PH_0)
    
    # generate boundary data
    return_BC0 = boundarydata.BC0(u_0,v_0,r0,rho_p,mu_p,0,0)
    u0 = return_BC0['BCu']
    v0 = return_BC0['BCv']
    
    # incoming characteristics
    q = 0.5*(zs*u_0-v_0)

    # incoming characteristics data
    Q0 = 0.5*(zs*u0 - v0)
   


    Hu_l = np.zeros(len(u))
    Hv_l = np.zeros(len(v))

    # penalize incoming characteristics against data 
    Hu_l[0:NP] = Hu_l[0:NP] + 2/(dx[0]) * (q-Q0)*(PH_0/w)
    Hv_l[0:NP] = Hv_l[0:NP] - 2/(zs*dx[0])*(q-Q0)*(PH_0/w)
    
    return {'Hu_l':Hu_l,'Hv_l':Hv_l}

# right boundary
def BCr_GL(u,v,w,dx, rn,rho,mu,x,t,NP):
    import boundarydata
    import numpy as np
    import Lagrange
    
    # exract local solutions in the last element
    uK = u[-NP:]
    vK = v[-NP:]
  
    mu_m = mu[-NP:];
    rho_m = rho[-NP:];

    mu_0 = Lagrange.interpol(NP,1,x,mu_m)
    rho_0 = Lagrange.interpol(NP,1,x,rho_m)

    cs = np.sqrt(mu_0/rho_0)
    zs = rho_0 * cs

    # extrapolate solutions to the boundaries
    NP = len(x)
    
    u_L = Lagrange.interpol(NP, 1, x, uK)    
    v_L = Lagrange.interpol(NP, 1, x, vK)  

    # initialize basis vector
    PH_1 = np.zeros(NP)

    # evaluate basis vector at the boundary
    for i in range (1,NP+1):  
        PH_1[i-1] = Lagrange.lagrange_basis(NP,i,1,x)    
        
    PH_1 = np.transpose(PH_1)
    

    # generate boundary data
    return_BCn = boundarydata.BCn(u_L,v_L,rn,rho_0,mu_0,0,0)
    un = return_BCn['BCu']
    vn = return_BCn['BCv']

    # incoming characteristics
    p = 0.5*(zs*u_L+v_L)
    
    # incoming characteristics data
    Pn = 0.5*(zs*un + vn)

    Hu_r = np.zeros(len(u))
    Hv_r = np.zeros(len(v))
   
    # penalize incoming characteristics against data     
    Hu_r[-NP:] = Hu_r[-NP:] + 2/(dx[-1])*(p-Pn)*(PH_1/w)
    Hv_r[-NP:] = Hv_r[-NP:] + 2/(zs*dx[-1])*(p-Pn)*(PH_1/w);
    
    return {'Hu_r':Hu_r,'Hv_r':Hv_r}


# FUNCTION BC_GL
# This fuction computes flux penalization at the physical boundaries
def BC_GL(u,v,w,dx,r0,rn,rho,mu,x,t,NP):
    import boundarydata
    import numpy as np
    import Lagrange
    # extract local solutions in the first element
    u1 = u[0:NP];
    v1 = v[0:NP];
    
    # exract local solutions in the last element
    uK = u[-NP:];
    vK = v[-NP:];
  
    cs = np.sqrt(mu/rho);
    Zs = rho*cs;
    
    # extrapolate solutions to the boundaries
    NP = len(x);
    
    u_0 = Lagrange.interpol(NP, -1, x, u1);   
    v_0 = Lagrange.interpol(NP, -1, x, v1);  
 
    u_L = Lagrange.interpol(NP, 1, x, uK);    
    v_L = Lagrange.interpol(NP, 1, x, vK);   
    
    PH_0 = np.zeros(NP)
    PH_1 = np.zeros(NP)
    
    for i in range (1,NP+1):
        PH_0[i-1] = Lagrange.lagrange_basis(NP,i,-1,x);    
        PH_1[i-1] = Lagrange.lagrange_basis(NP,i,1,x);    
        
    PH_0 = np.transpose(PH_0);
    PH_1 = np.transpose(PH_1);

    mu_m = mu[-NP:];
    rho_m = rho[-NP:];

    mu_p = mu[0:NP];
    rho_p = rho[0:NP];

    mu_0 = Lagrange.interpol(NP,1,x,mu_m)
    rho_0 = Lagrange.interpol(NP,1,x,rho_m)

    mu_1 = Lagrange.interpol(NP,-1,x,mu_p)
    rho_1 = Lagrange.interpol(NP,-1,x,rho_p)

    c1s = np.sqrt(mu_0/rho_0)
    z1s = rho_0 * c1s

    c2s = np.sqrt(mu_1/rho_1)
    z2s = rho_1 * c2s

    Hu = np.zeros(len(u));
    Hv = np.zeros(len(v));
    
    return_BCn = boundarydata.BCn(u_L,v_L,rn,rho_0,mu_0,0,0);
    un = return_BCn['BCu']
    vn = return_BCn['BCv']

    return_BC0 = boundarydata.BC0(u_0,v_0,r0,rho_1,mu_1,0,0);
    u0 = return_BC0['BCu']
    v0 = return_BC0['BCv']
    
    Pn = 0.5*(z2s*un + vn);
    Q0 = 0.5*(z1s*u0 - v0);
    
    p = 0.5*(z2s*u_L+v_L);
    q = 0.5*(z1s*u_0-v_0);
    
    Hu[0:NP] = Hu[0:NP] + 2/(dx[0]) * (q-Q0)*(PH_0/w);
    Hv[0:NP] = Hv[0:NP] - 2/(z1s*dx[0])*(q-Q0)*(PH_0/w);
    
    Hu[-NP:] = Hu[-NP:] + 2/(dx[-1])*(p-Pn)*(PH_1/w)
    Hv[-NP:] = Hv[-NP:] + 2/(z2s*dx[-1])*(p-Pn)*(PH_1/w);
    
    return {'Hu':Hu,'Hv':Hv}
