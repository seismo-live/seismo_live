def interface_condition(v_m,t_m,v_p,t_p,rho_m,mu_m,rho_p,mu_p):
    #
    # generate interface data for a locked interface
    
    import numpy as np

    # local wave speeds and impedances
    
    cs_m = np.sqrt(mu_m/rho_m)     # left block
    Zs_m = rho_m * cs_m            # left block
    
    cs_p = np.sqrt(mu_p/rho_p)     # right block
    Zs_p = rho_p * cs_p            # right block

    # outgoing characteristics that must be preserved
    q_m = Zs_m * v_m - t_m
    p_p = Zs_p * v_p + t_p

    eta_s = Zs_m * Zs_p/(Zs_m + Zs_p)      # half harmonic mean of shear impedance
    
    Phi = eta_s * (1.0/Zs_p * p_p - 1.0/Zs_m * q_m)  # stress transfer functional
    
    vv = 0.0                               # vanishing slip-rate (no slip condition)

    # hat-variables for tractions
    T_m = Phi                 # left block    
    T_p = Phi                 # right block

    # hat-variables for particle velocities
    V_m = (p_p - T_p)/Zs_p - vv      # left block
    V_p = (q_m + T_m)/Zs_m + vv      # right block
   
    
    return {'V_m':V_m, 'V_p':V_p, 'T_m':T_m, 'T_p':T_p}

def friction_law(v_m,t_m,v_p,t_p,S,psi,rho_m,mu_m,rho_p,mu_p,friction_parameters):
    #
    # generate interface data for an interface governed by a slip-weaking friction law
    
    import numpy as np

    # local wave speeds and impedances
    
    cs_m = np.sqrt(mu_m/rho_m)     # left block
    Zs_m = rho_m * cs_m            # left block
    
    cs_p = np.sqrt(mu_p/rho_p)     # right block
    Zs_p = rho_p * cs_p            # right block

    # outgoing characteristics that must be preserved
    q_m = Zs_m * v_m - t_m
    p_p = Zs_p * v_p + t_p

    eta_s = Zs_m * Zs_p/(Zs_m + Zs_p)                # half harmonic mean of shear impedance
    
    Phi = eta_s * (1.0/Zs_p * p_p - 1.0/Zs_m * q_m)  # stress transfer functional
    
    #friction_parameters = [fric_law, alpha, Tau_0, L0, f0, a, b, V0, sigma_n, alp_s, alp_d, D_c]    
    #                        0         1      2     3   4  5  6   7    8       9      10    11 
    Tau_0 = friction_parameters[2]
    fric_law = friction_parameters[0]
    alpha = friction_parameters[1]
    
    if(fric_law in 'locked'):

        vv = 0.0                               # vanishing slip-rate (no slip condition)

        # hat-variables for tractions
        T_m = Phi                 # left block    
        T_p = Phi                 # right block
        
        # hat-variables for particle velocities
        V_m = (p_p - T_p)/Zs_p - vv      # left block
        V_p = (q_m + T_m)/Zs_m + vv      # right block
   
        
        return {'V_m':V_m, 'V_p':V_p, 'T_m':T_m, 'T_p':T_p}
    
    if(fric_law in 'SW'):
        
        Tau_lock = Phi + Tau_0
        Tau_str = TauStrength(S,friction_parameters)
    
 
        if Tau_lock >= Tau_str:                          #fault is sliping
            # solve for slip rate (vv) and total traction (Tau_h)
            slip_weakening_return = slip_weakening(Phi,Tau_0, Tau_str, eta_s)
            Tau_h = slip_weakening_return['Tau_h']
            vv = slip_weakening_return['vv']
            
        # tractions on the fault
            T_m = Tau_h - Tau_0 
            T_p = Tau_h - Tau_0
            
            # particle velocity on the fault
            V_p = (q_m + T_m)/Zs_m + vv
            V_m = (p_p - T_p)/Zs_p - vv
            
        else:                                        #fault is locked 
            vv = 0.0                       # no slip

        # traction on the fault (same as stress transfer functional)
            T_m = Phi
            T_p = Phi

        # particle velocity on the fault
            V_p = (q_m + T_m)/Zs_m + vv
            V_m = (p_p - T_p)/Zs_p - vv

    if(fric_law in 'LN'):
        coeff = 1/(eta_s/alpha + 1)
        T_m = coeff *Phi    #alpha/(eta_s + alpha)* Phi
        T_p = coeff *Phi    #alpha/(eta_s + alpha)* Phi
        vv =  1.0/(eta_s + alpha)* Phi
        V_p = (q_m + T_m)/Zs_m + vv
        V_m = (p_p - T_p)/Zs_p - vv


    if(fric_law in 'RS'):

        # initialize sliprate for the solver
        V = np.abs(v_p - v_m)
        
        if (V > Phi):
            V = 0.5*Phi/eta_s
        
        # solve for slip rate (vv) and total traction (Tau_h)
        rate_and_state_return = rate_and_state(V,Phi,Tau_0,psi,eta_s,friction_parameters)
        
        Tau_h = rate_and_state_return['Tau_h']
        vv = rate_and_state_return['vv']
        
        # tractions on the fault                                                                                                          
        T_m = Tau_h - Tau_0
        T_p = Tau_h - Tau_0

        # particle velocity on the fault                                                                                                   
        V_p = (q_m + T_m)/Zs_m + vv
        V_m = (p_p - T_p)/Zs_p - vv

        
    return {'V_m':V_m, 'V_p':V_p, 'T_m':T_m, 'T_p':T_p}


def TauStrength(S, friction_parameters):
    import numpy as np
    #friction_parameters = [fric_law, alpha, Tau_0, L0, f0, a, b, V0, sigma_n, alp_s, alp_d, D_c]    
    #                        0         1      2     3   4  5  6   7    8       9      10    11 
    alp_s = friction_parameters[9]                           # stastic friction
    alp_d = friction_parameters[10]                           # dynamic friction
    nor_str = friction_parameters[8]                         # normal stress
    D_c = friction_parameters[11]                              # critical slip
    
    fric_coeff = alp_s - (alp_s-alp_d) * min(S,D_c)/D_c  # friction coefficient
    
    T_str = fric_coeff*nor_str                                # fault strength
    
    return T_str

def slip_weakening(phi,Tau_0, T_str, eta):
    # solve for slip-rate (vv):  
    vv = ((phi+Tau_0) - T_str)/eta
    
    # solve for total traction on the fault (Tau_h):
    Tau_h = (phi+Tau_0)-eta*vv
    
    return {'Tau_h':Tau_h,'vv':vv}


def rate_and_state(V, phi,Tau_0, psi, eta, friction_parameters):
    
    #L0 = 0.02
    #f0 = 0.6
    #b = 0.012
    #friction_parameters = [fric_law, alpha, Tau_0, L0, f0, a, b, V0, sigma_n, alp_s, alp_d, D_c]    
    #                        0         1      2     3   4  5  6   7    8       9      10    11 
    sigma_n = friction_parameters[8]
    V0 = friction_parameters[7]
    a = friction_parameters[5]
    
    #V0 = 1.0e-6        # reference velocity
    #a = 0.008          # direct effect
    #sigma_n = 120.0    # normal stress 
    
    PHI = phi+Tau_0

    # solve for slip-rate (vv):  
    vv = regula_falsi(V, PHI,eta,sigma_n,psi,V0,a)
    
    # solve for total traction on the fault (Tau_h):                                                                                             
    Tau_h = PHI - eta*vv
    #Tau_h = 0.0

    return {'Tau_h':Tau_h,'vv':vv}

def regula_falsi(V,Phi,eta,sigma_n,psi,V0,a):
    # a nonlinear solver for slip-rate using Regula-Falsi                                                                                      
    #solve: V + sigma_n/eta*f(V,psi) - Phi/eta = 0                                                                                              
    
    import numpy as np
    import math

    err = 1.0
    tol = 1e-12
    
    Vl = 0.0               #lower bound                                                                                                        
    Vr = Phi/eta           #upper bound                                                                                                        
    
    k = 1
    maxit = 5000
    
    fv = V + a*sigma_n/eta*math.asinh(0.5*V/V0*np.exp(psi/a)) - Phi/eta
    fl = Vl + a*sigma_n/eta*math.asinh(0.5*Vl/V0*np.exp(psi/a)) - Phi/eta
    fr = Vr + a*sigma_n/eta*math.asinh(0.5*Vr/V0*np.exp(psi/a)) - Phi/eta
    
    if (abs(fv) <= tol): # check if the guess is a solution                                                                                
        V = V
        return
    
    elif (abs(fl) <= tol): # check if the lower bound is a solution                                                                          
        V = Vl
        return

    elif (abs(fr) <= tol): # check if the upper bound is a solution                                                                         
        V = Vr
        return

    else:
        # else solve for V iteratively using regula-falsi                                                                                           
        while((k <= maxit) and (err >= tol)):
            V1 = V
            
            if (fv*fl > tol):
                Vl = V
                V = Vr - (Vr-Vl)*fr/(fr-fl)
                fl = fv
            elif (fv*fr > tol):
                Vr = V
                V = Vr -(Vr-Vl)*fr/(fr-fl)
                fr = fv
                
            fv = V + a*sigma_n/eta*math.asinh(0.5*V/V0*np.exp(psi/a)) - Phi/eta
                
            err = abs(V-V1)
            k = k + 1
                
    if (k >= maxit):
        print('error:=', err, 'max iteration:=', k, 'maximum iteration reached. solution didnt convergence!')
                    
    return V
                
